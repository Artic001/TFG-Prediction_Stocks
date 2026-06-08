"""
train_generalista.py
====================
Trains a generalist LSTM and XGBoost model with 20 US stocks.
Uses Exp.4 features (OHLCV + Technical + Macro + Market).

Since all stocks are US, the macro data (Fed, CPI, VIX...) and
the market index (^GSPC) are identical for all. This makes it
possible to train a single model that learns general market patterns.

The resulting model can predict any US stock without retraining.

Execution (once, offline):
    python train_generalista.py

Generated files:
    model_lstm_generalista.keras
    model_xgb_generalista.pkl
    scaler_generalista.pkl
    threshold_generalista.json
"""

import warnings
import os
import json
import numpy as np
import pandas as pd
import requests
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import yfinance as yf
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

tf.get_logger().setLevel("ERROR")


# CONFIGURATION

# 20 representative US stocks from diverse sectors
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",   # Large-cap technology
    "TSLA", "AMZN", "AMD",  "INTC",  "QCOM",   # Tech / Growth
    "KO",   "PG",   "JNJ",  "PEP",   "WMT",    # Stable consumer
    "JPM",  "BAC",  "GS",                      # Financial
    "XOM",  "CVX",                             # Energy
]

START_DATE  = "2010-07-01"
END_DATE    = "2026-03-01"
HORIZON     = 30
WINDOW_SIZE = 20
BATCH_SIZE  = 64
SEED        = 42
OPTUNA_TRIALS = 30

# Output files
OUT_LSTM      = "models/model_lstm_generalista.keras"
OUT_XGB       = "models/model_xgb_generalista.pkl"
OUT_SCALER    = "models/scaler_generalista.pkl"
OUT_THRESHOLD = "models/threshold_generalista.json"

# Create models/ directory if it does not exist
os.makedirs("models", exist_ok=True)

# US macro data (identical for all stocks)
FRED_API_KEY = "d1579a90b1b46f86b9b802630f4c5fda"
FRED_US = [
    ("DFF",          "fed_rate",     "Fed Funds Rate"),
    ("T10Y2Y",       "yield_curve",  "Yield Curve 10y-2y"),
    ("CPIAUCSL",     "cpi",          "CPI USA"),
    ("T10YIE",       "inflation_exp","Implicit inflation 10y"),
    ("VIXCLS",       "vix",          "VIX"),
    ("BAMLH0A0HYM2", "hy_spread",    "High Yield Spread USA"),
    ("UNRATE",       "unemployment", "Unemployment USA"),
    ("INDPRO",       "indprod",      "Industrial production USA"),
]
MACRO_TRANSFORMS = {
    "fed_rate"     : ("diff",          "fed_rate_chg"),
    "yield_curve"  : ("identity",      "yield_curve"),
    "cpi"          : ("pct_change_12", "cpi_yoy"),
    "inflation_exp": ("identity",      "inflation_exp"),
    "vix"          : ("log",           "vix_log"),
    "hy_spread"    : ("diff",          "hy_spread_chg"),
    "unemployment" : ("diff",          "unemployment_chg"),
    "indprod"      : ("pct_change",    "indprod_chg"),
}

# US market index (same for all stocks)
MARKET_INDEX    = "^GSPC"
SECONDARY_INDEX = "^STOXX50E"

np.random.seed(SEED)
tf.random.set_seed(SEED)

LSTM_CONFIG = {
    "epochs"  : 150,
    "units"   : 64,
    "dropout" : 0.5,
    "l2"      : 3e-3,
    "patience": 15,
}


# DATA LOADING
def load_stock(ticker):
    print(f"  [STOCK] Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def fetch_fred(series_id, col_name):
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_API_KEY}"
           f"&file_type=json&observation_start={START_DATE}"
           f"&observation_end={END_DATE}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    s = pd.Series(
        {pd.Timestamp(o["date"]): float(o["value"])
         for o in r.json()["observations"] if o["value"] != "."},
        name=col_name,
    )
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def load_macro_us(stock_index):
    """Load US macro data - identical for all stocks."""
    print(f"  [MACRO] Downloading FRED US series...")
    frames = {}
    for sid, col, desc in FRED_US:
        try:
            frames[col] = fetch_fred(sid, col)
            print(f"           OK {sid}")
        except Exception as e:
            print(f"           ERROR {sid}: {e}")

    transformed = []
    for col, s in {c: frames[c].dropna() for c in frames}.items():
        tr, new_name = MACRO_TRANSFORMS.get(col, ("identity", col))
        if tr == "diff":            t = s.diff().rename(new_name)
        elif tr == "pct_change_12": t = s.pct_change(12).rename(new_name)
        elif tr == "pct_change":    t = s.pct_change().rename(new_name)
        elif tr == "log":           t = np.log(s.clip(lower=1e-9)).rename(new_name)
        else:                       t = s.rename(new_name)
        transformed.append(t)
    if "vix" in frames:
        transformed.append(frames["vix"].dropna().pct_change().rename("vix_chg"))

    df_t = pd.concat(transformed, axis=1).replace([np.inf, -np.inf], np.nan)
    daily = pd.date_range(df_t.index.min(), df_t.index.max(),
                          freq="D").tz_localize(None)
    macro_daily = df_t.reindex(daily).ffill()
    stock_idx   = pd.to_datetime(stock_index).tz_localize(None).normalize()
    return macro_daily.reindex(stock_idx).replace([np.inf, -np.inf], np.nan)


def load_market_us(stock_index):
    """Load US market context - S&P500 for all stocks."""
    print(f"  [MARKET] Downloading S&P500 and STOXX50...")

    def dl(ticker):
        raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        c = raw["Close"].dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        return c

    idx_close = dl(MARKET_INDEX)
    sec_close = dl(SECONDARY_INDEX)

    out = pd.DataFrame(index=idx_close.index)
    out["index_ret_1d"] = np.log(idx_close / idx_close.shift(1))
    out["index_ret_5d"] = np.log(idx_close / idx_close.shift(5))
    out["index_vol"]    = out["index_ret_1d"].rolling(20).std()
    sec = sec_close.reindex(idx_close.index, method="ffill")
    out["stoxx_ret_1d"] = np.log(sec / sec.shift(1))
    out["stoxx_vol"]    = out["stoxx_ret_1d"].rolling(20).std()
    out["index_corr"]   = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)

    stock_idx = pd.to_datetime(stock_index).tz_localize(None).normalize()
    return out.reindex(stock_idx, method="ffill"), idx_close


# FEATURES
def build_base_features(df):
    C, O, H, L, V = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    rng = (H - L).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["log_ret_1d"]    = np.log(C / C.shift(1))
    out["log_ret_5d"]    = np.log(C / C.shift(5))
    out["log_ret_10d"]   = np.log(C / C.shift(10))
    out["range_pct"]     = (H - L) / C
    out["body_pct"]      = (C - O).abs() / rng
    out["overnight_gap"] = (O / C.shift(1)) - 1
    out["vol_ratio"]     = V / V.rolling(20).mean()
    out["realized_vol"]  = out["log_ret_1d"].rolling(10).std()
    out["close_sma20"]   = (C / C.rolling(20).mean()) - 1
    out["dir_5d"]        = (out["log_ret_1d"] > 0).rolling(5).mean()
    return out.replace([np.inf, -np.inf], np.nan)


def build_technical_indicators(df):
    C, H, L, V = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    ema21 = C.ewm(span=21, adjust=False).mean()
    ema50 = C.ewm(span=50, adjust=False).mean()
    out["ema_ratio_21_50"] = (ema21 / ema50) - 1
    tr    = pd.concat([H-L, (H-C.shift(1)).abs(),
                       (L-C.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    up    = H - H.shift(1); down = L.shift(1) - L
    dm_p  = up.where((up > down) & (up > 0), 0.0)
    dm_m  = down.where((down > up) & (down > 0), 0.0)
    di_p  = 100 * dm_p.rolling(14).mean() / (atr14 + 1e-9)
    di_m  = 100 * dm_m.rolling(14).mean() / (atr14 + 1e-9)
    out["adx"] = (100*(di_p-di_m).abs()/(di_p+di_m+1e-9)).rolling(14).mean()/100
    delta = C.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = (100 - 100/(1 + gain/(loss+1e-9)))/100
    out["roc_10"] = C.pct_change(10)
    sma20 = C.rolling(20).mean(); std20 = C.rolling(20).std()
    out["bb_width"] = 4*std20/(sma20+1e-9)
    out["atr_14"]   = atr14/(C+1e-9)
    mfv = ((C-L)-(H-C))/(H-L+1e-9)*V
    out["cmf"]       = mfv.rolling(20).sum()/(V.rolling(20).sum()+1e-9)
    obv = (np.sign(C.diff())*V).fillna(0).cumsum()
    out["obv_slope"] = (obv/(obv.rolling(50).std()+1e-9)).diff(10)/10
    max50 = H.rolling(50).max(); min50 = L.rolling(50).min()
    out["price_position"] = (C-min50)/(max50-min50+1e-9)
    out["dist_max_50"]    = (max50-C)/(C+1e-9)
    return out.replace([np.inf, -np.inf], np.nan)


def add_correlation(df_stock, market, idx_close):
    sidx = pd.to_datetime(df_stock.index).tz_localize(None).normalize()
    ia   = idx_close.reindex(sidx, method="ffill")
    sr   = np.log(df_stock["Close"] / df_stock["Close"].shift(1))
    ir   = np.log(ia / ia.shift(1))
    corr = sr.rolling(20).corr(ir)
    corr.index = sidx
    market = market.copy()
    market["index_corr"] = corr.values
    return market


def build_all_features(df, macro, market):
    base = build_base_features(df)
    tech = build_technical_indicators(df)
    for f in [base, tech, macro, market]:
        f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
    combined = (base.join(tech, how="inner")
                    .join(macro, how="left")
                    .join(market, how="left"))
    return combined.replace([np.inf, -np.inf], np.nan)


def build_target(df):
    C = df["Close"]
    future_mean = C.shift(-HORIZON).rolling(HORIZON).mean().shift(-(HORIZON-1))
    past_mean   = C.rolling(HORIZON).mean()
    return (future_mean > past_mean).astype(int).rename("target")


# COMBINED DATASET
def build_combined_dataset():
    """
    Downloads and processes all stocks.
    Returns X_raw, y_raw as concatenated numpy arrays.
    Macro and market index are downloaded ONCE and reused for all stocks.
    """
    print("\n" + "="*60)
    print("  BUILDING COMBINED DATASET")
    print("="*60)

    # Download macro and market once using S&P500 as reference index
    print("\n[1/3] Downloading S&P500 as reference index...")
    spx = yf.download(MARKET_INDEX, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)["Close"]
    spx.index = pd.to_datetime(spx.index).tz_localize(None).normalize()

    print("\n[2/3] Downloading US macro data (once)...")
    macro_ref = load_macro_us(spx.index)

    print("\n[3/3] Downloading US market context (once)...")
    market_ref, idx_close = load_market_us(spx.index)

    # Process each stock
    all_X = []
    all_y = []
    ticker_info = []

    print(f"\n[STOCKS] Processing {len(TICKERS)} stocks...")
    for i, ticker in enumerate(TICKERS):
        print(f"\n  [{i+1}/{len(TICKERS)}] {ticker}")
        try:
            df = load_stock(ticker)

            # Align macro and market to stock index
            stock_idx = df.index
            macro = macro_ref.reindex(
                pd.to_datetime(stock_idx).tz_localize(None).normalize()
            )
            market = market_ref.reindex(
                pd.to_datetime(stock_idx).tz_localize(None).normalize(),
                method="ffill"
            )

            # Add stock-specific correlation with S&P500
            market = add_correlation(df, market, idx_close)

            # Build features and target
            features = build_all_features(df, macro, market)
            target   = build_target(df)

            idx      = features.index.intersection(target.index)
            features = features.loc[idx].ffill().bfill().dropna()
            target   = target.loc[features.index]
            features = features.iloc[:-HORIZON]
            target   = target.iloc[:-HORIZON]

            if len(features) < 500:
                print(f"    SKIP - too few rows ({len(features)})")
                continue

            all_X.append(features.values.astype(np.float32))
            all_y.append(target.values.astype(np.float32))
            ticker_info.append({
                "ticker":    ticker,
                "n_samples": len(features),
                "pct_up":    float(target.mean()),
            })
            print(f"    OK - {len(features)} samples | "
                  f"upward: {target.mean():.1%}")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if not all_X:
        raise ValueError("No stocks processed successfully.")

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    n_features = X_combined.shape[1]

    print(f"\n{'='*60}")
    print(f"  COMBINED DATASET")
    print(f"  Stocks processed : {len(ticker_info)}")
    print(f"  Total samples    : {X_combined.shape[0]}")
    print(f"  Features         : {n_features}")
    print(f"  % upward global  : {y_combined.mean():.1%}")
    print(f"{'='*60}")

    return X_combined, y_combined, n_features, ticker_info


# SPLIT AND SCALING
def split_and_scale(X_raw, y_raw):
    """
    Temporal 70/15/15 split on the combined dataset.
    Scaler is fitted ONLY on train set.
    """
    n = len(X_raw)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_tr = X_raw[:n_train]
    X_vl = X_raw[n_train : n_train + n_val]
    X_te = X_raw[n_train + n_val:]
    y_tr = y_raw[:n_train]
    y_vl = y_raw[n_train : n_train + n_val]
    y_te = y_raw[n_train + n_val:]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_vl_s  = scaler.transform(X_vl)
    X_te_s  = scaler.transform(X_te)

    print(f"\n[SPLIT] Train: {X_tr_s.shape} | "
          f"Val: {X_vl_s.shape} | Test: {X_te_s.shape}")

    return {
        "X_train": X_tr_s, "y_train": y_tr,
        "X_val":   X_vl_s, "y_val":   y_vl,
        "X_test":  X_te_s, "y_test":  y_te,
    }, scaler


def make_sequences(data):
    """Creates sequences of WINDOW_SIZE days for the LSTM."""
    def seq(X, y):
        return (np.array([X[i-WINDOW_SIZE:i] for i in range(WINDOW_SIZE, len(X))]),
                np.array([y[i] for i in range(WINDOW_SIZE, len(y))]))

    Xtr, ytr = seq(data["X_train"], data["y_train"])
    Xvl, yvl = seq(data["X_val"],   data["y_val"])
    Xte, yte = seq(data["X_test"],  data["y_test"])

    print(f"[SEQS]  Train: {Xtr.shape} | Val: {Xvl.shape} | Test: {Xte.shape}")
    return {
        "X_train": Xtr, "y_train": ytr,
        "X_val":   Xvl, "y_val":   yvl,
        "X_test":  Xte, "y_test":  yte,
    }


# LSTM TRAINING
def train_lstm(data_seq, n_features):
    print(f"\n{'='*60}")
    print(f"  TRAINING GENERALIST LSTM")
    print(f"  units={LSTM_CONFIG['units']} | "
          f"dropout={LSTM_CONFIG['dropout']} | "
          f"l2={LSTM_CONFIG['l2']}")
    print(f"{'='*60}")

    cfg = LSTM_CONFIG
    model = Sequential([
        LSTM(
            units=cfg["units"],
            input_shape=(WINDOW_SIZE, n_features),
            kernel_regularizer=regularizers.l2(cfg["l2"]),
            recurrent_regularizer=regularizers.l2(cfg["l2"]),
            dropout=cfg["dropout"],
            recurrent_dropout=0.2,
        ),
        Dropout(cfg["dropout"]),
        Dense(16, activation="relu",
              kernel_regularizer=regularizers.l2(cfg["l2"])),
        Dropout(cfg["dropout"] * 0.5),
        Dense(1, activation="sigmoid"),
    ], name="LSTM_Generalist_30d")

    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    cbs = [
        EarlyStopping(monitor="val_auc", mode="max",
                      patience=cfg["patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_auc", mode="max",
                          factor=0.5, patience=7,
                          min_lr=1e-6, verbose=0),
    ]

    model.fit(
        data_seq["X_train"], data_seq["y_train"],
        validation_data=(data_seq["X_val"], data_seq["y_val"]),
        epochs=cfg["epochs"],
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1,
    )
    return model


# XGBOOST TRAINING
def train_xgb(data_flat):
    print(f"\n{'='*60}")
    print(f"  TRAINING GENERALIST XGBOOST (Optuna {OPTUNA_TRIALS} trials)")
    print(f"{'='*60}")

    X_tr, y_tr = data_flat["X_train"], data_flat["y_train"]
    X_vl, y_vl = data_flat["X_val"],   data_flat["y_val"]
    spw = float((y_tr == 0).sum()) / float((y_tr == 1).sum() + 1e-9)

    def objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 600),
            "max_depth"       : trial.suggest_int("max_depth", 3, 6),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma"           : trial.suggest_float("gamma", 0.0, 2.0),
            "scale_pos_weight": spw,
            "objective"       : "binary:logistic",
            "eval_metric"     : "logloss",
            "random_state"    : SEED,
            "n_jobs"          : -1,
            "verbosity"       : 0,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        pred = m.predict(X_vl)
        return balanced_accuracy_score(y_vl, pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_params
    best["scale_pos_weight"] = spw
    best["objective"]        = "binary:logistic"
    best["eval_metric"]      = "logloss"
    best["random_state"]     = SEED
    best["n_jobs"]           = -1
    best["verbosity"]        = 0

    print(f"  Best BalAcc val: {study.best_value:.4f}")

    model = xgb.XGBClassifier(**best)
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    return model


# OPTIMAL THRESHOLD
def find_threshold(predict_fn, data_val):
    """Finds the optimal threshold on the validation set."""
    y_prob = predict_fn(data_val["X_val"])
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.35, 0.66, 0.01):
        pred = (y_prob >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        ratio = pred.mean()
        if ratio < 0.10 or ratio > 0.90:
            continue
        ba = balanced_accuracy_score(data_val["y_val"], pred)
        if ba > best_ba:
            best_ba, best_t = ba, t
    print(f"  Optimal threshold: {best_t:.2f} | BalAcc val: {best_ba:.4f}")
    return float(best_t)


# FINAL EVALUATION
def evaluate(predict_fn, data, threshold, model_name):
    y_prob = predict_fn(data["X_test"])
    y_pred = (y_prob >= threshold).astype(int)
    y_test = data["y_test"]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0

    metrics = {
        "model"            : model_name,
        "roc_auc"          : round(roc_auc_score(y_test, y_prob), 4),
        "accuracy"         : round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "f1"               : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision"        : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"           : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "specificity"      : specificity,
        "threshold"        : threshold,
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS {model_name} GENERALIST")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<22}: {v}")
    print(f"{'='*60}")
    return metrics


# MAIN
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  GENERALIST MODEL TRAINING")
    print(f"  {len(TICKERS)} US stocks | Exp.4 | Horizon {HORIZON}d")
    print("="*60)

    # 1. Build combined dataset
    X_raw, y_raw, n_features, ticker_info = build_combined_dataset()

    # 2. Split and scale
    data_flat, scaler = split_and_scale(X_raw, y_raw)

    # 3. Sequences for LSTM
    data_seq = make_sequences(data_flat)

    # 4. Train LSTM
    lstm_model = train_lstm(data_seq, n_features)

    # 5. LSTM threshold
    print("\n[THRESHOLD LSTM]")
    def lstm_predict(X):
        return lstm_model.predict(X, verbose=0).flatten()
    threshold_lstm = find_threshold(lstm_predict, data_seq)

    # 6. Evaluate LSTM
    metrics_lstm = evaluate(lstm_predict, data_seq, threshold_lstm, "LSTM")

    # 7. Train XGBoost
    xgb_model = train_xgb(data_flat)

    # 8. XGBoost threshold
    print("\n[THRESHOLD XGBoost]")
    def xgb_predict(X):
        return xgb_model.predict_proba(X)[:, 1]
    threshold_xgb = find_threshold(xgb_predict, data_flat)

    # 9. Evaluate XGBoost
    metrics_xgb = evaluate(xgb_predict, data_flat, threshold_xgb, "XGBoost")

    # 10. Save models, scaler and thresholds
    print(f"\n[SAVING MODELS]")
    lstm_model.save(OUT_LSTM)
    print(f"  OK -> {OUT_LSTM}")

    joblib.dump(xgb_model, OUT_XGB)
    print(f"  OK -> {OUT_XGB}")

    joblib.dump(scaler, OUT_SCALER)
    print(f"  OK -> {OUT_SCALER}")

    thresholds = {
        "lstm": threshold_lstm,
        "xgb":  threshold_xgb,
    }
    with open(OUT_THRESHOLD, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  OK -> {OUT_THRESHOLD}")

    # 11. Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Stocks trained   : {len(ticker_info)}")
    print(f"  Total samples    : {X_raw.shape[0]}")
    print(f"  Features         : {n_features}")
    print(f"  LSTM  AUC        : {metrics_lstm['roc_auc']}")
    print(f"  XGBoost AUC      : {metrics_xgb['roc_auc']}")
    print(f"\n  Saved files:")
    print(f"    {OUT_LSTM}")
    print(f"    {OUT_XGB}")
    print(f"    {OUT_SCALER}")
    print(f"    {OUT_THRESHOLD}")
    print(f"\n  You can now use predict_generalista.py to make predictions")
    print(f"  for any US stock without retraining.")
    print(f"{'='*60}")