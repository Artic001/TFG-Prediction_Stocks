"""
predict_generalista.py
======================
Make predictions for any US stock using pre-trained models trained on 20 stocks. No retraining required.

Prerequisite:
Have run train_generalist.py at least once.

Execution:
    python predict_generalista.py --ticker AAPL
    python predict_generalista.py --ticker MSFT
    python predict_generalista.py --ticker V
"""

import argparse
import warnings
import os
import json
import numpy as np
import pandas as pd
import joblib
import requests

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_MILDL"] = "3"

import yfinance as yf
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report,
)

tf.get_logger().setLevel("ERROR")


# CONFIGURATION

START_DATE  = "2010-07-01"
END_DATE    = "2026-03-01"
HORIZON     = 30
WINDOW_SIZE = 20

FRED_API_KEY    = "d1579a90b1b46f86b9b802630f4c5fda"
MARKET_INDEX    = "^GSPC"
SECONDARY_INDEX = "^STOXX50E"

FRED_US = [
    ("DFF", "fed_rate", "Fed Funds Rate"),
    ("T10Y2Y", "yield_curve", "10Y–2Y Yield Curve"),
    ("CPIAUCSL", "cpi", "US CPI"),
    ("T10YIE", "inflation_exp", "10Y Breakeven Inflation"),
    ("VIXCLS", "vix", "VIX"),
    ("BAMLH0A0HYM2", "hy_spread", "US High Yield Spread"),
    ("UNRATE", "unemployment", "US Unemployment Rate"),
    ("INDPRO", "indprod", "US Industrial Production"),
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

# Pretrained model files
MODEL_LSTM      = "models/model_lstm_generalista.keras"
MODEL_XGB       = "models/model_xgb_generalista.pkl"
MODEL_SCALER    = "models/scaler_generalista.pkl"
MODEL_THRESHOLD = "models/threshold_generalista.json"


# LOAD PRETRAINED MODELS

def load_pretrained_modelos():
    """Load the models saved on disk."""
    for f in [MODEL_LSTM, MODEL_XGB, MODEL_SCALER, MODEL_THRESHOLD]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Not found '{f}'. "
                f"Run first: python train_generalista.py"
            )

    print("[MODELS] Loading pre-trained models...")
    lstm    = tf.keras.models.load_model(MODEL_LSTM)
    xgb     = joblib.load(MODEL_XGB)
    scaler  = joblib.load(MODEL_SCALER)
    with open(MODEL_THRESHOLD) as f:
        thresholds = json.load(f)

    print(f"  OK -> {MODEL_LSTM}")
    print(f"  OK -> {MODEL_XGB}")
    print(f"  OK -> {MODEL_SCALER}")
    print(f"  OK -> {MODEL_THRESHOLD}")
    print(f"  Threshold LSTM   : {thresholds['lstm']:.2f}")
    print(f"  Threshold XGBoost: {thresholds['xgb']:.2f}")
    return lstm, xgb, scaler, thresholds


# DATA LOADING (identical to train_generalista.py)

def load_stock(ticker):
    print(f"\n[DATA] Downloading {ticker} ({START_DATE} -> {END_DATE})...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    print(f"       {len(df)} sessions "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
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


def load_macro(stock_index):
    print("[MACRO] Downloading macro US...")
    frames = {}
    for sid, col, _ in FRED_US:
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
    daily = pd.date_range(df_t.index.min(), df_t.index.max(),
                          freq="D").tz_localize(None)
    macro_daily = df_t.reindex(daily).ffill()
    stock_idx   = pd.to_datetime(stock_index).tz_localize(None).normalize()
    return macro_daily.reindex(stock_idx).replace([np.inf, -np.inf], np.nan)


def load_market(stock_index):
    print("[MARKET] Downloading S&P500 and STOXX50...")

    def dl(ticker):
        raw = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
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


# DATA PREPARATION FOR PREDICTION

def prepare_ticker(ticker, scaler):
    """
    Process a ticker and return data ready for the model.
    Uses the pre-trained scaler. NO fitting is done, only transform.
    """
    df     = load_stock(ticker)
    macro  = load_macro(df.index)
    market, idx_close = load_market(df.index)
    market = add_correlation(df, market, idx_close)

    features = build_all_features(df, macro, market)
    target   = build_target(df)

    idx      = features.index.intersection(target.index)
    features = features.loc[idx].ffill().bfill().dropna()
    target   = target.loc[features.index]
    features = features.iloc[:-HORIZON]
    target   = target.iloc[:-HORIZON]

    X_raw = features.values.astype(np.float32)
    y_raw = target.values.astype(np.float32)
    dates = features.index

    # Scale using the pre-trained scaler (no fitting)
    X_scaled = scaler.transform(X_raw)

    # 70/15/15 split for fair evaluation
    n       = len(X_scaled)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_te = X_scaled[n_train + n_val:]
    y_te = y_raw[n_train + n_val:]
    d_te = dates[n_train + n_val:]

    print(f"[PREP] {ticker} | {len(features)} samples totals | "
          f"Test: {len(X_te)} samples | "
          f"Test set uplift: {y_te.mean():.1%}")

    # LSTM sequences (test)
    X_seq = np.array([X_te[i-WINDOW_SIZE:i]
                      for i in range(WINDOW_SIZE, len(X_te))])
    y_seq = y_te[WINDOW_SIZE:]
    d_seq = d_te[WINDOW_SIZE:]

    # Real prediction for today - last 20 days from the full dataset.
    X_tot = np.concatenate([
        X_scaled[:n_train], X_scaled[n_train:n_train+n_val], X_te
    ], axis=0)
    X_today_lstm = X_tot[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, X_scaled.shape[1])
    X_today_xgb  = X_tot[-1:, :]

    return {
        "X_flat"     : X_te,
        "y_flat"     : y_te,
        "X_seq"      : X_seq,
        "y_seq"      : y_seq,
        "dates"      : d_seq,
        "ticker"     : ticker,
        "X_today_lstm": X_today_lstm,
        "X_today_xgb" : X_today_xgb,
    }


# INVERSION SIGNAL

def get_signal(metrics):
    auc  = metrics["roc_auc"]
    prec = metrics["precision"]
    rec  = metrics["recall"]
    spec = metrics["specificity"]

    if auc < 0.55:
        return "INSUFFICIENT"
    elif rec > spec and prec > 0.55 and spec > 0.50:
        return "BUY"
    elif spec > rec and spec > 0.55:
        return "DO NOT BUY"
    else:
        return "INSUFFICIENT"


def print_signal(ticker, metrics_lstm, metrics_xgb):
    signal_lstm = get_signal(metrics_lstm)
    signal_xgb  = get_signal(metrics_xgb)

    W = 60
    print(f"\n{'='*W}")
    print(f"  GENERALIST SIGNAL - {ticker} | Horizon: 30 days")
    print(f"{'='*W}")
    print(f"  LSTM    : {signal_lstm:<15} "
          f"(AUC={metrics_lstm['roc_auc']:.3f} | "
          f"Prec={metrics_lstm['precision']:.3f} | "
          f"Spec={metrics_lstm['specificity']:.3f})")
    print(f"  XGBoost : {signal_xgb:<15} "
          f"(AUC={metrics_xgb['roc_auc']:.3f} | "
          f"Prec={metrics_xgb['precision']:.3f} | "
          f"Spec={metrics_xgb['specificity']:.3f})")
    print(f"{'-'*W}")

    # Combined signal from both models
    signals = [signal_lstm, signal_xgb]
    if signals.count("BUY") == 2:
        consensus = "BUY (consensus)"
    elif signals.count("DO NOT BUY") == 2:
        consensus = "DO NOT BUY (consensus)"
    elif "INSUFFICIENT" in signals:
        consensus = "INSUFFICIENT"
    else:
        consensus = "DIVERGENT (LSTM and XGBoost do not agree)"

    print(f"  Consensus : {consensus}")
    print(f"{'='*W}")
    print(f"  WARNING: This is an indicative prediction. It is not investment advice.")
    print(f"{'='*W}\n")


# EVALUATION

def evaluate_model(y_prob, y_true, threshold, model_name, prob_today=None):
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    spec   = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0

    pred_today = None
    if prob_today is not None:
        pred_today = "UP" if prob_today >= threshold else "DOWN"

    metrics = {
        "model"            : model_name,
        "roc_auc"          : round(roc_auc_score(y_true, y_prob), 4),
        "accuracy"         : round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "f1"               : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision"        : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"           : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "specificity"      : spec,
        "threshold"        : threshold,
        "prob_today"        : round(prob_today, 4) if prob_today is not None else None,
        "pred_today"        : pred_today,
    }

    print(f"\n  --- {model_name} ---")
    print(classification_report(y_true, y_pred,
                                 target_names=["Downside", "Rise"]))
    print(f"  TN={cm[0,0]:>4}  FP={cm[0,1]:>4}")
    print(f"  FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<22}: {v}")
    return metrics


# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediction with the pre-trained generalist model."
    )
    parser.add_argument("--ticker", type=str, required=True,
                        help="Ticker US (ex: AAPL, V, DIS, NFLX)")
    args   = parser.parse_args()
    ticker = args.ticker.upper()

    print(f"\n{'='*60}")
    print(f"  GENERALIST PREDICTION - {ticker}")
    print(f"  Pre-trained model with 20 US stocks | Exp.4 | 30d")
    print(f"{'='*60}")

    # 1. Load models
    lstm, xgb_model, scaler, thresholds = load_pretrained_modelos()

    # 2. Prepare ticker data
    data = prepare_ticker(ticker, scaler)

    print(f"\n{'='*60}")
    print(f"  TEST EVALUATION - {ticker}")
    print(f"{'='*60}")

    # 3. Prediction LSTM
    if len(data["X_seq"]) > 0:
        prob_lstm      = lstm.predict(data["X_seq"], verbose=0).flatten()
        prob_today_lstm = float(lstm.predict(data["X_today_lstm"], verbose=0).flatten()[0])
        metrics_lstm   = evaluate_model(
            prob_lstm, data["y_seq"],
            thresholds["lstm"], "Generalist LSTM",
            prob_today=prob_today_lstm
        )
    else:
        print("  Skip LSTM, too little data to create sequences.")
        metrics_lstm = {"roc_auc": 0.5, "precision": 0.5,
                        "recall": 0.5, "specificity": 0.5,
                        "pred_today": None, "prob_today": None}

    # 4. Prediction XGBoost
    prob_xgb      = xgb_model.predict_proba(data["X_flat"])[:, 1]
    prob_today_xgb = float(xgb_model.predict_proba(data["X_today_xgb"])[:, 1][0])
    metrics_xgb   = evaluate_model(
        prob_xgb, data["y_flat"],
        thresholds["xgb"], "Generalist XGBoost",
        prob_today=prob_today_xgb
    )

    # 5. Final signal
    print_signal(ticker, metrics_lstm, metrics_xgb)