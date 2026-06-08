"""
neuron_search.py
================
Architecture search for the LSTM model across all 4 experiments.
Tests 5 configurations varying units, dropout and L2 regularization.
Used to justify the selected architecture in the TFG memory.

Execution:
    python neuron_search.py --ticker AAPL --exp 1
    python neuron_search.py --ticker AAPL --exp 2
    python neuron_search.py --ticker AAPL --exp 3
    python neuron_search.py --ticker AAPL --exp 4
"""

import argparse
import warnings
import os
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, confusion_matrix,
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

tf.get_logger().setLevel("ERROR")

# CONFIGURATION
START_DATE  = "2010-07-01"
END_DATE    = "2026-03-01"
HORIZON     = 30
WINDOW_SIZE = 20
BATCH_SIZE  = 64
SEED        = 1

from dotenv import load_dotenv
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

np.random.seed(SEED)
tf.random.set_seed(SEED)

# 5 configurations to compare
CONFIGS = [
    {"name": "A - Small",       "units": 16,  "dropout": 0.2, "l2": 1e-3},
    {"name": "B - Medium",      "units": 32,  "dropout": 0.3, "l2": 2e-3},
    {"name": "C - Current 48",  "units": 48,  "dropout": 0.4, "l2": 2e-3},
    {"name": "D - Current 64",  "units": 64,  "dropout": 0.5, "l2": 3e-3},
    {"name": "E - Extra Large", "units": 128, "dropout": 0.5, "l2": 3e-3},
]

# Current config per experiment
CURRENT_CONFIG = {1: "C - Current 48", 2: "C - Current 48",
                  3: "D - Current 64", 4: "D - Current 64"}

FRED_US = [
    ("DFF",          "fed_rate"),
    ("T10Y2Y",       "yield_curve"),
    ("CPIAUCSL",     "cpi"),
    ("T10YIE",       "inflation_exp"),
    ("VIXCLS",       "vix"),
    ("BAMLH0A0HYM2", "hy_spread"),
    ("UNRATE",       "unemployment"),
    ("INDPRO",       "indprod"),
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

# DATA LOADING
def load_stock(ticker):
    print(f"[DATA] Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    print(f"       {len(df)} sessions ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def fetch_fred(series_id, col_name):
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_API_KEY}"
           f"&file_type=json&observation_start={START_DATE}&observation_end={END_DATE}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    s = pd.Series(
        {pd.Timestamp(o["date"]): float(o["value"])
         for o in r.json()["observations"] if o["value"] != "."},
        name=col_name,
    )
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def load_macro(stock_index):
    print("[MACRO] Downloading FRED US...")
    frames = {}
    for sid, col in FRED_US:
        try:
            frames[col] = fetch_fred(sid, col)
        except Exception:
            pass

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
    daily = pd.date_range(df_t.index.min(), df_t.index.max(), freq="D").tz_localize(None)
    macro_daily = df_t.reindex(daily).ffill()
    sidx = pd.to_datetime(stock_index).tz_localize(None).normalize()
    return macro_daily.reindex(sidx).replace([np.inf, -np.inf], np.nan)


def load_market(stock_index):
    print("[MARKET] Downloading S&P500 and STOXX50...")
    def dl(ticker):
        raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        c = raw["Close"].dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        return c

    idx_close = dl("^GSPC")
    sec_close = dl("^STOXX50E")

    out = pd.DataFrame(index=idx_close.index)
    out["index_ret_1d"] = np.log(idx_close / idx_close.shift(1))
    out["index_ret_5d"] = np.log(idx_close / idx_close.shift(5))
    out["index_vol"]    = out["index_ret_1d"].rolling(20).std()
    sec = sec_close.reindex(idx_close.index, method="ffill")
    out["stoxx_ret_1d"] = np.log(sec / sec.shift(1))
    out["stoxx_vol"]    = out["stoxx_ret_1d"].rolling(20).std()
    out["index_corr"]   = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)

    # Add correlation stock vs index
    sidx = pd.to_datetime(stock_index).tz_localize(None).normalize()
    return out.reindex(sidx, method="ffill"), idx_close


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


def build_technical(df):
    C, H, L, V = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    ema21 = C.ewm(span=21, adjust=False).mean()
    ema50 = C.ewm(span=50, adjust=False).mean()
    out["ema_ratio_21_50"] = (ema21 / ema50) - 1
    tr    = pd.concat([H-L, (H-C.shift(1)).abs(), (L-C.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    up = H - H.shift(1); down = L.shift(1) - L
    dm_p = up.where((up > down) & (up > 0), 0.)
    dm_m = down.where((down > up) & (down > 0), 0.)
    di_p = 100 * dm_p.rolling(14).mean() / (atr14 + 1e-9)
    di_m = 100 * dm_m.rolling(14).mean() / (atr14 + 1e-9)
    out["adx"]    = (100*(di_p-di_m).abs()/(di_p+di_m+1e-9)).rolling(14).mean()/100
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


def build_features(df, exp, macro=None, market=None):
    base = build_base_features(df)
    if exp == 1:
        return base
    tech = build_technical(df)
    for f in [base, tech]:
        f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
    combined = base.join(tech, how="inner")
    if exp == 2:
        return combined.replace([np.inf, -np.inf], np.nan)
    if macro is not None:
        macro.index = pd.to_datetime(macro.index).tz_localize(None).normalize()
        combined = combined.join(macro, how="left")
    if exp == 3:
        return combined.replace([np.inf, -np.inf], np.nan)
    if market is not None:
        market.index = pd.to_datetime(market.index).tz_localize(None).normalize()
        combined = combined.join(market, how="left")
    return combined.replace([np.inf, -np.inf], np.nan)


def build_target(df):
    C = df["Close"]
    future_mean = C.shift(-HORIZON).rolling(HORIZON).mean().shift(-(HORIZON - 1))
    past_mean   = C.rolling(HORIZON).mean()
    return (future_mean > past_mean).astype(int).rename("target")


# DATA PREPARATION
def prepare_data(df, exp, macro=None, market=None):
    features = build_features(df, exp, macro, market)
    target   = build_target(df)

    idx      = features.index.intersection(target.index)
    features = features.loc[idx].ffill().bfill().dropna()
    target   = target.loc[features.index]
    features = features.iloc[:-HORIZON]
    target   = target.iloc[:-HORIZON]

    X_raw = features.values
    y_raw = target.values

    n_train = int(len(X_raw) * 0.70)
    n_val   = int(len(X_raw) * 0.15)

    X_tr = X_raw[:n_train]
    X_vl = X_raw[n_train:n_train + n_val]
    X_te = X_raw[n_train + n_val:]
    y_tr = y_raw[:n_train]
    y_vl = y_raw[n_train:n_train + n_val]
    y_te = y_raw[n_train + n_val:]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_vl_s  = scaler.transform(X_vl)
    X_te_s  = scaler.transform(X_te)

    def make_seqs(X, y):
        return (np.array([X[i-WINDOW_SIZE:i] for i in range(WINDOW_SIZE, len(X))]),
                np.array([y[i] for i in range(WINDOW_SIZE, len(y))]))

    X_train, y_train = make_seqs(X_tr_s, y_tr)
    X_val,   y_val   = make_seqs(X_vl_s, y_vl)
    X_test,  y_test  = make_seqs(X_te_s, y_te)

    X_tot   = np.concatenate([X_tr_s, X_vl_s, X_te_s], axis=0)
    X_today = X_tot[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, X_raw.shape[1])

    print(f"[PREP] Exp.{exp} | {X_raw.shape[1]} features | "
          f"Train:{X_train.shape} Val:{X_val.shape} Test:{X_test.shape}")
    print(f"       % upward test: {y_test.mean():.2%}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "n_features": X_raw.shape[1],
        "X_today": X_today,
    }


# MODEL
def build_model(n_features, cfg):
    dense_units = max(8, cfg["units"] // 4)
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
        Dense(dense_units, activation="relu",
              kernel_regularizer=regularizers.l2(cfg["l2"])),
        Dropout(cfg["dropout"] * 0.5),
        Dense(1, activation="sigmoid"),
    ], name=f"LSTM_{cfg['name'].replace(' ', '_').replace('-','')}")

    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_and_eval(data, cfg):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = build_model(data["n_features"], cfg)
    cbs = [
        EarlyStopping(monitor="val_auc", mode="max",
                      patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_auc", mode="max",
                          factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=150, batch_size=BATCH_SIZE, callbacks=cbs, verbose=0,
    )
    epochs_run = len(history.history["loss"])

    # Threshold
    y_prob_val = model.predict(data["X_val"], verbose=0).flatten()
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.35, 0.66, 0.01):
        pred = (y_prob_val >= t).astype(int)
        if len(np.unique(pred)) < 2: continue
        if pred.mean() < 0.10 or pred.mean() > 0.90: continue
        ba = balanced_accuracy_score(data["y_val"], pred)
        if ba > best_ba:
            best_ba, best_t = ba, t

    # Evaluate
    y_prob = model.predict(data["X_test"], verbose=0).flatten()
    y_pred = (y_prob >= best_t).astype(int)
    y_test = data["y_test"]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    spec = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0

    return {
        "config"    : cfg["name"],
        "units"     : cfg["units"],
        "dropout"   : cfg["dropout"],
        "l2"        : cfg["l2"],
        "params"    : model.count_params(),
        "roc_auc"   : round(roc_auc_score(y_test, y_prob), 4),
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "bal_acc"   : round(balanced_accuracy_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "specificity": spec,
        "threshold" : round(best_t, 2),
        "epochs"    : epochs_run,
    }


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--exp",    type=int, required=True, choices=[1, 2, 3, 4])
    args   = parser.parse_args()
    ticker = args.ticker.upper()
    exp    = args.exp

    exp_names = {1: "OHLCV", 2: "OHLCV+Technical",
                 3: "OHLCV+Technical+Macro", 4: "OHLCV+Technical+Macro+Market"}

    print(f"\n{'='*65}")
    print(f"  ARCHITECTURE SEARCH - Exp.{exp} {exp_names[exp]} | {ticker} | {HORIZON}d")
    print(f"  {len(CONFIGS)} configurations | Current: {CURRENT_CONFIG[exp]}")
    print(f"{'='*65}")

    df     = load_stock(ticker)
    macro  = load_macro(df.index)  if exp >= 3 else None
    if exp >= 4:
        market_df, idx_close = load_market(df.index)
        market_df = add_correlation(df, market_df, idx_close)
    else:
        market_df = None

    data = prepare_data(df, exp, macro, market_df)

    results = []
    current = CURRENT_CONFIG[exp]
    for i, cfg in enumerate(CONFIGS, 1):
        marker = " <<<" if cfg["name"] == current else ""
        print(f"\n[{i}/{len(CONFIGS)}] {cfg['name']}{marker}")
        print(f"         units={cfg['units']} | dropout={cfg['dropout']} | l2={cfg['l2']}")
        m = train_and_eval(data, cfg)
        results.append(m)
        print(f"         -> AUC={m['roc_auc']:.4f} | Acc={m['accuracy']:.4f} | "
              f"Epochs={m['epochs']} | Params={m['params']:,}")

    # Summary table
    print(f"\n{'='*95}")
    print(f"  RESULTS — Exp.{exp} {exp_names[exp]} | {ticker} | {HORIZON}d")
    print(f"{'='*95}")
    print(f"  {'Config':<22} {'Units':>6} {'Drop':>6} {'L2':>7} "
          f"{'Params':>8} {'AUC':>8} {'Acc':>8} {'BalAcc':>8} {'Prec UP':>8} {'Prec DW':>8}")
    print(f"  {'-'*91}")
    for m in results:
        marker = " <<<" if m["config"] == current else ""
        print(f"  {m['config']:<22} {m['units']:>6} {m['dropout']:>6.1f} {m['l2']:>7.4f} "
              f"{m['params']:>8,} {m['roc_auc']:>8.4f} {m['accuracy']:>8.4f} "
              f"{m['bal_acc']:>8.4f} {m['precision']:>8.4f} {m['specificity']:>8.4f}{marker}")
    print(f"{'='*95}")

    best = max(results, key=lambda x: x["roc_auc"])
    print(f"\n  Best AUC : {best['config']} (AUC={best['roc_auc']:.4f})")
    print(f"  Current  : {current}")
    if best["config"] == current:
        print(f"  -> Current configuration is OPTIMAL for Exp.{exp}")
    else:
        diff = abs(best["roc_auc"] - next(r["roc_auc"] for r in results if r["config"] == current))
        print(f"  -> Difference with current: {diff:.4f} AUC")
    print(f"{'='*95}\n")