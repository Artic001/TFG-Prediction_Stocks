"""
lstm_v3_macro.py
=============
Experiment 3 - LSTM OHLCV + Technical Indicators + Macroeconomic Variables
Adds macro variables via FRED API on top of the 20 features from Exp.2.

Automatic region detection from ticker:
    TSLA, AAPL… -> US (Fed, CPI USA, VIX, HY spread, unemployment, industrial production)
    ITX.MC, SAN.MC -> EU (ECB, Eurozone CPI, VIX, HY EUR, unemployment EZ)
    BMW.DE, SAP.DE -> EU (same as Eurozone)
    SHEL.L, BP.L -> UK (Bank of England, UK CPI, VIX, UK unemployment)
    7203.T, 9984.T -> JP (Bank of Japan, Japan CPI, VIX, Japan unemployment)

Execution:
    python exp3_macro.py --horizon 1  --ticker TSLA
    python exp3_macro.py --horizon 1  --ticker ITX.MC
    python exp3_macro.py --horizon 30 --ticker BMW.DE
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings, os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_MILDL"] = "3"

import yfinance as yf
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, balanced_accuracy_score,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

tf.get_logger().setLevel("ERROR")


# PARAMETERS
TICKER      = "TSLA"
START_DATE  = "2010-07-01"
END_DATE    = "2026-03-01"
WINDOW_SIZE = 20
BATCH_SIZE  = 64
LR          = 1e-3
SEED        = 1

import os
from dotenv import load_dotenv
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

SUFFIX_TO_REGION = {
    "":   "US", "US": "US",
    "MC": "EU", "MA": "EU",
    "DE": "EU", "F":  "EU",
    "PA": "EU", "AS": "EU",
    "MI": "EU",
    "L":  "UK",
    "T":  "JP",
    "HK": "GLOBAL", "SS": "GLOBAL",
    "TO": "GLOBAL", "AX": "GLOBAL",
    "SW": "GLOBAL",
}

FRED_BY_REGION = {
    "US": [
        ("DFF",           "fed_rate",      "Fed Funds Rate"),
        ("T10Y2Y",        "yield_curve",   "Yield Curve 10y-2y"),
        ("CPIAUCSL",      "cpi",           "US CPI"),
        ("T10YIE",        "inflation_exp", "10-year implied inflation"),
        ("VIXCLS",        "vix",           "VIX"),
        ("BAMLH0A0HYM2",  "hy_spread",     "High Yield Spread US"),
        ("UNRATE",        "unemployment",  "US unemployment"),
        ("INDPRO",        "indprod",       "US industrial production"),
    ],
    "EU": [
        ("ECBDFR",             "fed_rate",    "ECB interest rates"),
        ("IRLTLT01EZM156N",    "yield_curve", "Eurozone 10-year government bonds"),
        ("CP0000EZ17M086NEST", "cpi",         "Eurozone CPI"),
        ("VIXCLS",             "vix",         "VIX (global)"),
        ("BAMLHE00EHYIOAS",    "hy_spread",   "HY Spread EUR"),
        ("LRHUTTTTEZM156S",    "unemployment","Eurozone unemployment"),
        ("PRMNTO01EZQ661S",    "indprod",     "Eurozone industrial production"),
    ],
    "UK": [
        ("IUDSOIA",          "fed_rate",    "Bank of England interest rate"),
        ("IRLTLT01GBM156N",  "yield_curve", "UK 10-year government bonds"),
        ("GBRCPIALLMINMEI",  "cpi",         "UK CPI"),
        ("VIXCLS",           "vix",         "VIX (global)"),
        ("LRHUTTTTGBM156S",  "unemployment","UK unemployment"),
    ],
    "JP": [
        ("IRSTCB01JPM156N",  "fed_rate",    "Bank of Japan interest rate"),
        ("IRLTLT01JPM156N",  "yield_curve", "Japan 10-year government bonds"),
        ("JPNCPIALLMINMEI",  "cpi",         "Japan CPI"),
        ("VIXCLS",           "vix",         "VIX (global)"),
        ("LRHUTTTTJPM156S",  "unemployment","Japan unemployment"),
    ],
    "GLOBAL": [
        ("VIXCLS",          "vix",         "VIX (global)"),
        ("BAMLH0A0HYM2",    "hy_spread",   "HY Spread US"),
        ("T10Y2Y",          "yield_curve", "Yield Curve US"),
    ],
}

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


def get_region(ticker: str) -> str:
    parts  = ticker.upper().split(".")
    suffix = parts[-1] if len(parts) > 1 else ""
    return SUFFIX_TO_REGION.get(suffix, "US")


def get_fred_series(ticker: str):
    region = get_region(ticker)
    return FRED_BY_REGION.get(region, FRED_BY_REGION["GLOBAL"]), region


np.random.seed(SEED)
tf.random.set_seed(SEED)

HORIZON_CONFIG = {
    1: {
        "epochs": 150, "units": 64, "dropout": 0.3,
        "l2": 5e-4, "patience": 20,
    },
    30: {
        "epochs": 150, "units": 64, "dropout": 0.5,
        "l2": 3e-3, "patience": 20,
    },
}


# DATA
def load_stock():
    print(f"[DATA] Downloading {TICKER} ({START_DATE} -> {END_DATE})...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"       {len(df)} sessions  "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
    return df


# MACRO DATA (FRED API)
def fetch_fred_series(series_id: str, col_name: str) -> pd.Series:
    """Download a FRED series and return a pandas Series indexed by date."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
        f"&observation_start={START_DATE}"
        f"&observation_end={END_DATE}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    s = pd.Series(
        {pd.Timestamp(o["date"]): float(o["value"])
         for o in obs if o["value"] != "."},
        name=col_name,
    )
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def load_macro(stock_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Download the corresponding FRED series for the ticker’s region,
    transform them, and align them to the trading calendar using forward fill.
    """
    fred_series, region = get_fred_series(TICKER)
    print(f"[MACRO] Detected region: {region} | {len(fred_series)} series FRED...")

    frames = {}
    for series_id, col_name, desc in fred_series:
        try:
            s = fetch_fred_series(series_id, col_name)
            frames[col_name] = s
            print(f"         OK {series_id:<25} -> {col_name:<18} "
                  f"({len(s)} observations)")
        except Exception as e:
            print(f"         ERROR {series_id:<25}: {e}")

    s = {col: frames[col].dropna() for col in frames}

    transformed_list = []
    for col, serie in s.items():
        transform, new_name = MACRO_TRANSFORMS.get(col, ("identity", col))
        if transform == "diff":
            t = serie.diff().rename(new_name)
        elif transform == "pct_change_12":
            t = serie.pct_change(12).rename(new_name)
        elif transform == "pct_change":
            t = serie.pct_change().rename(new_name)
        elif transform == "log":
            t = np.log(serie.clip(lower=1e-9)).rename(new_name)
        else:
            t = serie.rename(new_name)
        transformed_list.append(t)

    if "vix" in s:
        transformed_list.append(s["vix"].pct_change().rename("vix_chg"))

    transformed = pd.concat(transformed_list, axis=1)
    transformed = transformed.replace([np.inf, -np.inf], np.nan)

    daily_idx = pd.date_range(
        start=transformed.index.min(),
        end=transformed.index.max(),
        freq="D"
    ).tz_localize(None)
    macro_daily = transformed.reindex(daily_idx).ffill()
    stock_idx   = pd.to_datetime(stock_index).tz_localize(None).normalize()
    macro       = macro_daily.reindex(stock_idx)

    result = macro.replace([np.inf, -np.inf], np.nan)
    print(f"         -> {len(transformed.columns)} generated macro features")
    return result


# FEATURES EXP.1 (OHLCV)
def build_base_features(df):
    C = df["Close"]; O = df["Open"]
    H = df["High"];  L = df["Low"]
    V = df["Volume"]
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
    return out


# FEATURES EXP.2 (TECHNICAL)
def build_technical_indicators(df):
    C = df["Close"]; O = df["Open"]
    H = df["High"];  L = df["Low"]
    V = df["Volume"]
    out = pd.DataFrame(index=df.index)

    ema21 = C.ewm(span=21, adjust=False).mean()
    ema50 = C.ewm(span=50, adjust=False).mean()
    out["ema_ratio_21_50"] = (ema21 / ema50) - 1

    tr    = pd.concat([H - L,
                       (H - C.shift(1)).abs(),
                       (L - C.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    up    = H - H.shift(1)
    down  = L.shift(1) - L
    dm_p  = up.where((up > down) & (up > 0), 0.0)
    dm_m  = down.where((down > up) & (down > 0), 0.0)
    di_p  = 100 * dm_p.rolling(14).mean() / (atr14 + 1e-9)
    di_m  = 100 * dm_m.rolling(14).mean() / (atr14 + 1e-9)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    out["adx"] = dx.rolling(14).mean() / 100

    delta = C.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = (100 - 100 / (1 + gain / (loss + 1e-9))) / 100
    out["roc_10"] = C.pct_change(10)

    sma20  = C.rolling(20).mean()
    std20  = C.rolling(20).std()
    bb_up  = sma20 + 2 * std20
    bb_low = sma20 - 2 * std20
    out["bb_width"] = (bb_up - bb_low) / (sma20 + 1e-9)
    out["atr_14"]   = atr14 / (C + 1e-9)

    mfv = ((C - L) - (H - C)) / (H - L + 1e-9) * V
    out["cmf"] = mfv.rolling(20).sum() / (V.rolling(20).sum() + 1e-9)

    obv      = (np.sign(C.diff()) * V).fillna(0).cumsum()
    obv_norm = obv / (obv.rolling(50).std() + 1e-9)
    out["obv_slope"] = obv_norm.diff(10) / 10

    max50 = H.rolling(50).max()
    min50 = L.rolling(50).min()
    out["price_position"] = (C - min50) / (max50 - min50 + 1e-9)
    out["dist_max_50"]    = (max50 - C) / (C + 1e-9)

    return out.replace([np.inf, -np.inf], np.nan)


# COMBINE ALL FEATURES
def build_all_features(df, macro: pd.DataFrame):
    """Combines: 10 OHLCV + 10 Technical + ~9 Macro = ~29 features."""
    base = build_base_features(df)
    tech = build_technical_indicators(df)

    base.index  = pd.to_datetime(base.index).tz_localize(None).normalize()
    tech.index  = pd.to_datetime(tech.index).tz_localize(None).normalize()
    macro.index = pd.to_datetime(macro.index).tz_localize(None).normalize()

    combined = base.join(tech, how="inner").join(macro, how="left")
    return combined.replace([np.inf, -np.inf], np.nan)


# TARGET
def build_target(df, horizon: int):
    C = df["Close"]
    if horizon == 1:
        future_ret = C.pct_change(1).shift(-1)
        target = (future_ret > 0).astype(int)
    else:
        future_mean = C.shift(-horizon).rolling(horizon).mean().shift(-(horizon - 1))
        past_mean   = C.rolling(horizon).mean()
        target = (future_mean > past_mean).astype(int)
    return target.rename("target")


# SPLIT + SCALING + SEQUENCES
def prepare_data(df, macro: pd.DataFrame, horizon: int):
    features = build_all_features(df, macro)
    target   = build_target(df, horizon)

    idx      = features.index.intersection(target.index)
    features = features.loc[idx].ffill().bfill().dropna()
    target   = target.loc[features.index]

    features = features.iloc[:-horizon]
    target   = target.iloc[:-horizon]

    X_raw = features.values
    y_raw = target.values
    dates = features.index

    n       = len(X_raw)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_tr_raw  = X_raw[:n_train]
    X_val_raw = X_raw[n_train : n_train + n_val]
    X_te_raw  = X_raw[n_train + n_val:]
    y_tr      = y_raw[:n_train]
    y_val     = y_raw[n_train : n_train + n_val]
    y_te      = y_raw[n_train + n_val:]
    dates_te  = dates[n_train + n_val:]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr_raw)
    X_val_s = scaler.transform(X_val_raw)
    X_te_s  = scaler.transform(X_te_raw)

    def make_seqs(X, y, w):
        return (np.array([X[i-w:i] for i in range(w, len(X))]),
                np.array([y[i]     for i in range(w, len(y))]))

    X_train, y_train = make_seqs(X_tr_s,  y_tr,  WINDOW_SIZE)
    X_val,   y_val   = make_seqs(X_val_s, y_val, WINDOW_SIZE)
    X_test,  y_test  = make_seqs(X_te_s,  y_te,  WINDOW_SIZE)
    dates_test       = dates_te[WINDOW_SIZE:]

    n_base  = len(build_base_features(df).columns)
    n_tech  = len(build_technical_indicators(df).columns)
    n_macro = X_raw.shape[1] - n_base - n_tech

    print(f"\n[PREP] Horizon          : {horizon} days")
    print(f"       Features OHLCV    : {n_base}")
    print(f"       Technical features: {n_tech}")
    print(f"       Macro features    : {n_macro}")
    print(f"       TOTAL features    : {X_raw.shape[1]}")
    print(f"       Train  : {X_train.shape}")
    print(f"       Val    : {X_val.shape}")
    print(f"       Test   : {X_test.shape}")
    print(f"       % upward test     : {y_test.mean():.2%}")
    print(f"       Test dates        : {dates_test[0].date()} -> "
          f"{dates_test[-1].date()}")

    # Last 20 days of the entire dataset, real prediction today
    X_tot_s  = np.concatenate([X_tr_s, X_val_s, X_te_s], axis=0)
    X_today   = X_tot_s[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, X_raw.shape[1])

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "dates_test": dates_test,
        "n_features": X_raw.shape[1],
        "X_today":  X_today,
    }


# MODEL
def build_model(n_features, horizon: int, stacked: bool = False):
    cfg = HORIZON_CONFIG[horizon]

    if stacked:
        layers = [
            LSTM(cfg["units"], input_shape=(WINDOW_SIZE, n_features),
                 kernel_regularizer=regularizers.l2(cfg["l2"] * 2),
                 recurrent_regularizer=regularizers.l2(cfg["l2"] * 2),
                 dropout=cfg["dropout"], recurrent_dropout=0.2,
                 return_sequences=True),
            Dropout(cfg["dropout"]),
            LSTM(cfg["units"] // 2,
                 kernel_regularizer=regularizers.l2(cfg["l2"] * 2),
                 recurrent_regularizer=regularizers.l2(cfg["l2"] * 2),
                 dropout=cfg["dropout"], recurrent_dropout=0.2),
            Dropout(cfg["dropout"]),
            Dense(16, activation="relu",
                  kernel_regularizer=regularizers.l2(cfg["l2"])),
            Dropout(cfg["dropout"] * 0.5),
            Dense(1, activation="sigmoid"),
        ]
        model = Sequential(layers, name="LSTM_Macro_Stacked")
    else:
        layers = [
            LSTM(cfg["units"], input_shape=(WINDOW_SIZE, n_features),
                 kernel_regularizer=regularizers.l2(cfg["l2"]),
                 recurrent_regularizer=regularizers.l2(cfg["l2"]),
                 dropout=cfg["dropout"], recurrent_dropout=0.2),
            Dropout(cfg["dropout"]),
            Dense(16, activation="relu",
                  kernel_regularizer=regularizers.l2(cfg["l2"])),
            Dropout(cfg["dropout"] * 0.5),
            Dense(1, activation="sigmoid"),
        ]
        model = Sequential(layers, name="LSTM_Macro_Standard")

    model.compile(
        optimizer=Adam(learning_rate=LR, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# TRAINING
def train_model(model, data, horizon: int):
    cfg = HORIZON_CONFIG[horizon]

    if horizon == 30:
        cbs = [
            EarlyStopping(monitor="val_auc", mode="max",
                          patience=15,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_auc", mode="max",
                              factor=0.5, patience=7,
                              min_lr=1e-6, verbose=0),
        ]
    else:
        cbs = [
            EarlyStopping(monitor="val_auc", mode="max",
                          patience=cfg["patience"],
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_auc", mode="max",
                              factor=0.5, patience=10,
                              min_lr=1e-6, verbose=0),
        ]

    print(f"\n[TRAIN] Horizon={horizon}d | units={cfg['units']} | "
          f"dropout={cfg['dropout']} | l2={cfg['l2']} | monitor=val_auc (max)")

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=cfg["epochs"],
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1,
    )
    return history


# OPTIMAL THRESHOLD
def find_best_threshold(model, data):
    y_prob = model.predict(data["X_val"], verbose=0).flatten()
    print(f"\n[THRESHOLD] Val range: [{y_prob.min():.3f}, "
          f"{y_prob.max():.3f}]  mean={y_prob.mean():.3f}")

    best_t, best_ba = 0.5, 0.0
    rows = []
    for t in np.arange(0.35, 0.66, 0.01):
        y_pred = (y_prob >= t).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        pred_ratio = y_pred.mean()
        if pred_ratio < 0.10 or pred_ratio > 0.90:
            continue
        ba = balanced_accuracy_score(data["y_val"], y_pred)
        f1 = f1_score(data["y_val"], y_pred, zero_division=0)
        rows.append((t, ba, f1))
        if ba > best_ba:
            best_ba, best_t = ba, t

    rows.sort(key=lambda x: -x[1])
    print(f"            Optimal: t={best_t:.2f}  BalAcc={best_ba:.4f}")
    print("            Top 5:")
    for t, ba, f1 in rows[:5]:
        print(f"              t={t:.2f}  BalAcc={ba:.4f}  F1={f1:.4f}")
    return best_t


# EVALUATION
def evaluate(model, data, threshold, horizon: int):
    y_prob = model.predict(data["X_test"], verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    y_test = data["y_test"]

    # Real prediction today - last 20 days of the entire dataset
    prob_today = float(model.predict(data["X_today"], verbose=0).flatten()[0])
    pred_today = "UP" if prob_today >= threshold else "DOWN"

    acc  = accuracy_score(y_test, y_pred)
    ba   = balanced_accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS - Exp.3 Macro | Horizon {horizon}d | {TICKER}")
    print(f"{'='*60}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {ba:.4f}")
    print(f"  F1-Score          : {f1:.4f}")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=["Downward", "Upward"]))
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:>4}  FP={cm[0,1]:>4}")
    print(f"    FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")
    print(f"  prob_today            : {prob_today:.4f}")
    print(f"  pred_today            : {pred_today}")
    print(f"  pct_upward_test       : {float(np.mean(y_test)):.2%}")
    print(f"{'='*60}")

    return y_prob, {
        "accuracy": acc, "balanced_accuracy": ba,
        "f1": f1, "roc_auc": auc,
        "precision": prec, "recall": rec,
        "specificity":      specificity,
        "prob_today":       round(prob_today, 4),
        "pred_today":       pred_today,
        "pct_upward_test":  round(float(np.mean(y_test)), 3),
    }


# INVESTOR SIGNAL
def investment_signal(metrics: dict, horizon: int):
    import datetime

    auc         = metrics["roc_auc"]
    precision   = metrics["precision"]
    recall      = metrics["recall"]
    acc         = metrics["accuracy"]
    specificity = metrics["specificity"]
    hor_txt     = "day" if horizon == 1 else f"{horizon} days"
    today       = datetime.date.today().strftime("%d %b %Y")

    pred_today = metrics.get("pred_today", None)

    if auc < 0.55:
        signal     = "INSUFFICIENT"
        signal_lbl = "INSUFFICIENT SIGNAL"
        action     = "The model does not have enough reliability to recommend any action."
        reason     = f"ROC-AUC is {auc:.3f}, too close to a random model (0.50)."
    elif pred_today == "UP" and precision > 0.55 and specificity > 0.50:
        signal     = "BUY"
        signal_lbl = "SIGNAL: BUY"
        action     = (f"The model recommends buying {TICKER} and holding\n"
                      f"  during the next {hor_txt}.")
        reason     = (f"In situations similar to the current one, the model has\n"
                      f"  correctly predicted the UPWARD direction {precision*100:.1f}%\n"
                      f"  of the times in the evaluated period.")
    elif pred_today == "DOWN" and specificity > 0.55:
        signal     = "DO NOT BUY"
        signal_lbl = "SIGNAL: DO NOT BUY / SELL"
        action     = (f"The model recommends not buying {TICKER} now.\n"
                      f"  If you have open positions, consider closing them.")
        reason     = (f"In situations similar to the current one, the model has\n"
                      f"  correctly predicted the DOWNWARD direction {specificity*100:.1f}%\n"
                      f"  of the times in the evaluated period.")
    else:
        signal     = "INSUFFICIENT"
        signal_lbl = "INSUFFICIENT SIGNAL"
        action     = "The model does not generate a clear enough signal to act."
        reason     = "Reliability in both directions is similar or low."

    conf_txt = ("Moderate-High" if auc >= 0.70
                else "Moderate" if auc >= 0.60
                else "Low")

    W = 60
    print(f"\n{'='*W}")
    print(f"  {TICKER} - {signal_lbl}")
    print(f"  {today}  |  Horizon: {hor_txt}")
    print(f"{'='*W}")
    print(f"  {action}")
    print(f"{'-'*W}")
    print(f"  Why?")
    print(f"  {reason}")
    print(f"{'-'*W}")
    print(f"  Historical model performance (test set):")
    print(f"    Precision (up)  : {precision*100:.1f}%")
    print(f"    Precision (down) : {specificity*100:.1f}%")
    print(f"    Global accuracy      : {acc*100:.1f}%")
    print(f"    ROC-AUC              : {auc:.3f}")
    print(f"    Reliability           : {conf_txt}")
    print(f"{'='*W}")
    print(f"  WARNING: This prediction is indicative and does not")
    print(f"        constitute any investment advice.")
    print(f"{'='*W}\n")

    return signal


# CHARTS
def plot_all(history, data, y_prob, threshold, horizon: int, metrics: dict):
    y_test  = data["y_test"]
    y_pred  = (y_prob >= threshold).astype(int)
    dates   = pd.to_datetime(data["dates_test"])
    correct = y_pred == y_test

    # Fig 1: training history
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(
        f"Exp.3 Macro | Horizon {horizon}d | Training history",
        fontweight="bold", fontsize=13)

    for ax, key, title, c1, c2 in [
        (axes[0], "loss", "Loss (BCE)",  "#2563EB", "#DC2626"),
        (axes[1], "auc",  "ROC-AUC",     "#059669", "#D97706"),
    ]:
        tr  = history.history[key]
        val = history.history[f"val_{key}"]
        ep  = range(1, len(tr) + 1)
        ax.plot(ep, tr,  color=c1, lw=1.8, label="Train")
        ax.plot(ep, val, color=c2, lw=1.8, ls="--", label="Val")
        best = (np.argmin(val) if "loss" in key else np.argmax(val))
        ax.axvline(best + 1, color="gray", ls=":", lw=1.2,
                   label=f"Best ep.{best+1}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Fig 2: predictions
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"Exp.3 Macro | Horizon {horizon}d | "
        f"AUC={metrics['roc_auc']:.4f}  Acc={metrics['accuracy']:.4f}",
        fontweight="bold", fontsize=13)

    axes[0].plot(dates, y_prob, color="#8B5CF6", lw=0.9,
                 alpha=0.9, label="P(upward)")
    axes[0].axhline(threshold, color="#F59E0B", ls="--", lw=1.3,
                    label=f"Threshold ({threshold:.2f})")
    axes[0].fill_between(dates, 0, y_test * 0.1,
                         alpha=0.25, color="#10B981", label="Actual target")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(dates[correct],  np.ones(correct.sum()),
                    color="#10B981", s=7, alpha=0.5, label="Correct")
    axes[1].scatter(dates[~correct], np.zeros((~correct).sum()),
                    color="#EF4444", s=7, alpha=0.5, label="Error")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Incorrect", "Correct"])
    axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

    # Fig 3: ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#8B5CF6", lw=2,
            label=f"Exp.3 Macro (AUC={metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.fill_between(fpr, tpr, alpha=0.07, color="#8B5CF6")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - Exp.3 Macro | {horizon}d | {TICKER}",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1,
                        choices=[1, 30],
                        help="Prediction horizon in days (default: 1)")
    parser.add_argument("--stacked", action="store_true", default=False,
                        help="Use Stacked LSTM (2 layers) instead of standard (1 layer)")
    parser.add_argument("--ticker", type=str, required=True,
                        help="Stock ticker (ex: TSLA, AAPL, ITX.MC, BMW.DE)")
    args    = parser.parse_args()
    horizon = args.horizon
    TICKER  = args.ticker.upper()

    _, region = get_fred_series(TICKER)
    print(f"\n{'='*60}")
    print(f"  Exp.3 MACRO - LSTM | {TICKER} | {horizon}d | Region: {region}")
    print(f"{'='*60}")

    df        = load_stock()
    macro     = load_macro(df.index)
    data      = prepare_data(df, macro, horizon)
    model     = build_model(data["n_features"], horizon, stacked=args.stacked)
    model.summary()
    history   = train_model(model, data, horizon)
    threshold = find_best_threshold(model, data)
    y_prob, metrics = evaluate(model, data, threshold, horizon)
    investment_signal(metrics, horizon)
    plot_all(history, data, y_prob, threshold, horizon, metrics)

    print(f"\n[DONE] Experiment 3 Macro | {horizon}d completed.")