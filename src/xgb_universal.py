"""
xgb_universal.py
================
Universal XGBoost model for directional stock prediction.

Implements the same 4 progressive experiments as the LSTM model:
  Exp.1 - OHLCV (10 features)
  Exp.2 - + Technical indicators (20 features)
  Exp.3 - + Regional macroeconomic variables (~28–29 features)
  Exp.4 - + Global market context (~35 features)

Differences compared to LSTM:
  - No temporal window: each row is treated as an independent observation
  - Hyperparameters optimized with Optuna (Bayesian search, 30 trials)
  - Optimal threshold selected using Balanced Accuracy on validation set (same as LSTM)
  - Same metrics, splits, and plots for direct comparability

Execution:
  python xgb_universal.py --ticker TSLA --exp 4 --horizon 1
  python xgb_universal.py --ticker TSLA --exp 1 --horizon 30
  python xgb_universal.py --ticker ITX.MC --exp 3 --horizon 1
  python xgb_universal.py --ticker BMW.DE --exp 4 --horizon 30

Dependencies:
  pip install xgboost optuna
"""

import argparse
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_MILDL"] = "3"

import yfinance as yf
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve,
)


# REGIONAL CONFIGURATION
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
        ("DFF",          "fed_rate",        "Federal Funds Rate"),
        ("T10Y2Y",       "yield_curve",     "10Y–2Y Yield Curve"),
        ("CPIAUCSL",     "cpi",             "US CPI"),
        ("T10YIE",       "inflation_exp",   "10Y Breakeven Inflation"),
        ("VIXCLS",       "vix",             "VIX"),
        ("BAMLH0A0HYM2", "hy_spread",       "US High Yield Spread"),
        ("UNRATE",       "unemployment",    "US Unemployment Rate"),
        ("INDPRO",       "indprod",         "US Industrial Production"),
    ],
    "EU": [
        ("ECBDFR",             "fed_rate",        "ECB Policy Rate"),
        ("IRLTLT01EZM156N",    "yield_curve",     "Euro Area 10Y Bond Yield"),
        ("CP0000EZ17M086NEST", "cpi",             "Euro Area CPI"),
        ("VIXCLS",             "vix",             "Global VIX"),
        ("BAMLHE00EHYIOAS",    "hy_spread",       "Euro High Yield Spread"),
        ("LRHUTTTTEZM156S",    "unemployment",    "Euro Area Unemployment Rate"),
        ("PRMNTO01EZQ661S",    "indprod",         "Euro Area Industrial Production"),
    ],
    "UK": [
        ("IUDSOIA",           "fed_rate",        "Bank of England Rate"),
        ("IRLTLT01GBM156N",   "yield_curve",     "UK 10Y Bond Yield"),
        ("GBRCPIALLMINMEI",   "cpi",             "UK CPI"),
        ("VIXCLS",            "vix",             "Global VIX"),
        ("LRHUTTTTGBM156S",   "unemployment",    "UK Unemployment Rate"),
    ],
    "JP": [
        ("IRSTCB01JPM156N",   "fed_rate",        "Bank of Japan Rate"),
        ("IRLTLT01JPM156N",   "yield_curve",     "Japan 10Y Bond Yield"),
        ("JPNCPIALLMINMEI",   "cpi",             "Japan CPI"),
        ("VIXCLS",            "vix",             "Global VIX"),
        ("LRHUTTTTJPM156S",   "unemployment",    "Japan Unemployment Rate"),
    ],
    "GLOBAL": [
        ("VIXCLS",         "vix",           "Global VIX"),
        ("BAMLH0A0HYM2",   "hy_spread",     "US High Yield Spread"),
        ("T10Y2Y",         "yield_curve",   "US Yield Curve (10Y–2Y)"),
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

SUFFIX_TO_MARKET = {
    "":   ("^GSPC",      "^STOXX50E",  "S&P 500"),
    "US": ("^GSPC",      "^STOXX50E",  "S&P 500"),
    "MC": ("^IBEX",      "^GSPC",      "IBEX 35"),
    "MA": ("^IBEX",      "^GSPC",      "IBEX 35"),
    "DE": ("^GDAXI",     "^GSPC",      "DAX"),
    "F":  ("^GDAXI",     "^GSPC",      "DAX"),
    "PA": ("^FCHI",      "^GSPC",      "CAC 40"),
    "AS": ("^AEX",       "^GSPC",      "AEX"),
    "MI": ("FTSEMIB.MI", "^GSPC",      "FTSE MIB"),
    "L":  ("^FTSE",      "^GSPC",      "FTSE 100"),
    "T":  ("^N225",      "^GSPC",      "Nikkei 225"),
    "HK": ("^HSI",       "^GSPC",      "Hang Seng"),
    "SS": ("000001.SS",  "^GSPC",      "SSE Composite"),
    "TO": ("^GSPTSE",    "^GSPC",      "TSX Composite"),
    "AX": ("^AXJO",      "^GSPC",      "ASX 200"),
    "SW": ("^SSMI",      "^GSPC",      "SMI"),
}

from dotenv import load_dotenv
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# CONSTANTS
SEED          = 1
OPTUNA_TRIALS = 30

EXP_NAMES = {
    1: "OHLCV",
    2: "OHLCV+Technical",
    3: "OHLCV+Technical+Macro",
    4: "OHLCV+Technical+Macro+Market",
}
EXP_COLORS = {1: "#6366F1", 2: "#10B981", 3: "#F59E0B", 4: "#F97316"}

np.random.seed(SEED)


# HELPERS REGIONALS
def get_suffix(ticker: str) -> str:
    parts = ticker.upper().split(".")
    return parts[-1] if len(parts) > 1 else ""

def get_region(ticker: str) -> str:
    return SUFFIX_TO_REGION.get(get_suffix(ticker), "US")

def get_fred_series(ticker: str):
    region = get_region(ticker)
    return FRED_BY_REGION.get(region, FRED_BY_REGION["GLOBAL"]), region

def get_market_config(ticker: str):
    return SUFFIX_TO_MARKET.get(get_suffix(ticker), SUFFIX_TO_MARKET[""])


# DATA
def load_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"[DATA] Downloading {ticker} ({start} -> {end})...")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    if len(df) == 0:
        raise ValueError(f"No data found for {ticker}.")
    print(f"       {len(df)} sessions "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
    return df


# MACRO (FRED)
def fetch_fred(series_id, col, start, end):
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_API_KEY}"
           f"&file_type=json&observation_start={start}&observation_end={end}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    s = pd.Series(
        {pd.Timestamp(o["date"]): float(o["value"])
         for o in r.json()["observations"] if o["value"] != "."},
        name=col,
    )
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def load_macro(stock_idx, ticker, start, end):
    fred_series, region = get_fred_series(ticker)
    print(f"[MACRO] Region: {region} | {len(fred_series)} series FRED...")
    raw = {}
    for sid, col, desc in fred_series:
        try:
            raw[col] = fetch_fred(sid, col, start, end)
            print(f"         OK {sid:<25} -> {col:<18} ({len(raw[col])} obs)")
        except Exception as e:
            print(f"         ERROR {sid:<25}: {e}")

    if not raw:
        return pd.DataFrame(index=stock_idx)

    transformed = []
    for col, s in raw.items():
        tr, new_name = MACRO_TRANSFORMS.get(col, ("identity", col))
        sc = s.dropna()
        if tr == "diff":            t = sc.diff().rename(new_name)
        elif tr == "pct_change_12": t = sc.pct_change(12).rename(new_name)
        elif tr == "pct_change":    t = sc.pct_change().rename(new_name)
        elif tr == "log":           t = np.log(sc.clip(lower=1e-9)).rename(new_name)
        else:                       t = sc.rename(new_name)
        transformed.append(t)
    if "vix" in raw:
        transformed.append(raw["vix"].dropna().pct_change().rename("vix_chg"))

    df_t = pd.concat(transformed, axis=1).replace([np.inf, -np.inf], np.nan)
    daily = pd.date_range(df_t.index.min(), df_t.index.max(),
                          freq="D").tz_localize(None)
    macro = df_t.reindex(daily).ffill().reindex(
        pd.to_datetime(stock_idx).tz_localize(None).normalize())
    print(f"         -> {len(df_t.columns)} features macro")
    return macro


# MARKET CONTEXT
def load_market_context(stock_idx, ticker, start, end):
    mkt_t, sec_t, mkt_name = get_market_config(ticker)
    print(f"[MARKET] Market context...")

    def dl(t, name):
        raw = yf.download(t, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        c = raw["Close"].dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        print(f"         OK {t:<15} ({len(c)} sessions) [{name}]")
        return c

    idx_close = dl(mkt_t, mkt_name)
    sec_close = dl(sec_t, "secondary index")

    out = pd.DataFrame(index=idx_close.index)
    out["index_ret_1d"] = np.log(idx_close / idx_close.shift(1))
    out["index_ret_5d"] = np.log(idx_close / idx_close.shift(5))
    out["index_vol"]    = out["index_ret_1d"].rolling(20).std()
    sec = sec_close.reindex(idx_close.index, method="ffill")
    out["sec_ret_1d"]   = np.log(sec / sec.shift(1))
    out["sec_vol"]      = out["sec_ret_1d"].rolling(20).std()
    out["index_corr"]   = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)

    sidx   = pd.to_datetime(stock_idx).tz_localize(None).normalize()
    market = out.reindex(sidx, method="ffill")
    print(f"         -> {len(market.columns)} market features")
    return market, idx_close


def add_correlation(df_stock, market, idx_close):
    sidx   = pd.to_datetime(df_stock.index).tz_localize(None).normalize()
    ia     = idx_close.reindex(sidx, method="ffill")
    sr     = np.log(df_stock["Close"] / df_stock["Close"].shift(1))
    ir     = np.log(ia / ia.shift(1))
    corr   = sr.rolling(20).corr(ir)
    corr.index = sidx
    market = market.copy()
    market["index_corr"] = corr.values
    return market


# FEATURES
def build_base(df):
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
    C, O, H, L, V = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    ema21 = C.ewm(span=21, adjust=False).mean()
    ema50 = C.ewm(span=50, adjust=False).mean()
    out["ema_ratio_21_50"] = (ema21 / ema50) - 1
    tr    = pd.concat([H-L, (H-C.shift(1)).abs(), (L-C.shift(1)).abs()],
                      axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    up = H - H.shift(1)
    down = L.shift(1) - L
    dm_p = up.where((up > down) & (up > 0), 0.)
    dm_m = down.where((down > up) & (down > 0), 0.)
    di_p = 100 * dm_p.rolling(14).mean() / (atr14 + 1e-9)
    di_m = 100 * dm_m.rolling(14).mean() / (atr14 + 1e-9)
    out["adx"]    = (100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)).rolling(14).mean() / 100
    delta = C.diff()
    out["rsi_14"] = (100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() /
                     (-delta.clip(upper=0).rolling(14).mean() + 1e-9))) / 100
    out["roc_10"] = C.pct_change(10)
    sma20 = C.rolling(20).mean()
    std20 = C.rolling(20).std()
    out["bb_width"] = 4 * std20 / (sma20 + 1e-9)
    out["atr_14"]   = atr14 / (C + 1e-9)
    mfv = ((C - L) - (H - C)) / (H - L + 1e-9) * V
    out["cmf"]       = mfv.rolling(20).sum() / (V.rolling(20).sum() + 1e-9)
    obv = (np.sign(C.diff()) * V).fillna(0).cumsum()
    out["obv_slope"] = (obv / (obv.rolling(50).std() + 1e-9)).diff(10) / 10
    max50 = H.rolling(50).max()
    min50 = L.rolling(50).min()
    out["price_position"] = (C - min50) / (max50 - min50 + 1e-9)
    out["dist_max_50"]    = (max50 - C) / (C + 1e-9)
    return out.replace([np.inf, -np.inf], np.nan)


def build_features(df, exp, macro=None, market=None):
    def ni(f):
        f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
        return f
    base = ni(build_base(df))
    if exp == 1:
        return base
    comb = base.join(ni(build_technical(df)), how="inner")
    if exp == 2:
        return comb.replace([np.inf, -np.inf], np.nan)
    if macro is not None and len(macro.columns) > 0:
        comb = comb.join(ni(macro.copy()), how="left")
    if exp == 3:
        return comb.replace([np.inf, -np.inf], np.nan)
    if market is not None and len(market.columns) > 0:
        comb = comb.join(ni(market.copy()), how="left")
    return comb.replace([np.inf, -np.inf], np.nan)


# TARGET
def build_target(df, horizon):
    C = df["Close"]
    if horizon == 1:
        return (C.pct_change(1).shift(-1) > 0).astype(int).rename("target")
    future = C.shift(-horizon).rolling(horizon).mean().shift(-(horizon - 1))
    return (future > C.rolling(horizon).mean()).astype(int).rename("target")


# SPLIT + SCALING + SEQUENCES
def prepare_data(df, exp, horizon, macro=None, market=None):
    feats  = build_features(df, exp, macro, market)
    target = build_target(df, horizon)

    idx    = feats.index.intersection(target.index)
    feats  = feats.loc[idx].ffill().bfill().dropna()
    target = target.loc[feats.index]
    feats  = feats.iloc[:-horizon]
    target = target.iloc[:-horizon]

    X, y, dates = feats.values, target.values, feats.index
    n = len(X)
    n_train, n_val = int(n * 0.70), int(n * 0.15)

    Xtr, Xv, Xte = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    ytr, yv, yte = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]
    dte          = dates[n_train + n_val:]

    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xv_s  = sc.transform(Xv)
    Xte_s = sc.transform(Xte)

    print(f"\n[PREP] Exp.{exp} | Horizon {horizon}d | {X.shape[1]} features")
    print(f"       Train:{Xtr_s.shape}  Val:{Xv_s.shape}  Test:{Xte_s.shape}")
    print(f"       % upward test: {yte.mean():.2%}")
    print(f"       Dates test: {dte[0].date()} -> {dte[-1].date()}")

    # Last row of the entire dataset, real prediction today
    X_tot_s = np.concatenate([Xtr_s, Xv_s, Xte_s], axis=0)
    X_today  = X_tot_s[-1:, :]

    return {
        "X_train":      Xtr_s, "y_train": ytr,
        "X_val":        Xv_s,  "y_val":   yv,
        "X_test":       Xte_s, "y_test":  yte,
        "dates_test":   dte,
        "n_features":   X.shape[1],
        "feature_names": list(feats.columns),
        "X_today":       X_today,
    }


# OPTUNA - HYPERPARAMETER OPTIMIZATION
def optimize_xgb(data, horizon, n_trials=OPTUNA_TRIALS):
    """
    Bayesian search of XGBoost hyperparameters.
    Optimizes Balanced Accuracy on the validation set.
    """
    print(f"\n[OPTUNA] Optimizing hyperparameters ({n_trials} trials) "
          f"[metric: Balanced Accuracy]...")

    scale_pos_weight = (data["y_train"] == 0).sum() / (data["y_train"] == 1).sum()

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 600),
            "max_depth"        : trial.suggest_int("max_depth", 3, 6),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight" : trial.suggest_int("min_child_weight", 3, 15),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma"            : trial.suggest_float("gamma", 0.0, 2.0),
            "scale_pos_weight" : scale_pos_weight,
            "objective"        : "binary:logistic",
            "eval_metric"      : "logloss",
            "random_state"     : SEED,
            "n_jobs"           : -1,
            "verbosity"        : 0,
        }
        modelo = xgb.XGBClassifier(**params)
        modelo.fit(
            data["X_train"], data["y_train"],
            eval_set=[(data["X_val"], data["y_val"])],
            verbose=False,
        )
        y_pred = modelo.predict(data["X_val"])
        return balanced_accuracy_score(data["y_val"], y_pred)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["scale_pos_weight"] = scale_pos_weight
    best["objective"]        = "binary:logistic"
    best["eval_metric"]      = "logloss"
    best["random_state"]     = SEED
    best["n_jobs"]           = -1
    best["verbosity"]        = 0

    print(f"         Best BalAcc val: {study.best_value:.4f}")
    print(f"         Best params:")
    for k, v in study.best_params.items():
        print(f"           {k:<22}: {v}")

    return best, study


# FINAL TRAINING
def train_modelo(data, best_params):
    print(f"\n[TRAIN] Final training with best hyperparameters...")
    modelo = xgb.XGBClassifier(**best_params)
    modelo.fit(
        data["X_train"], data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=False,
    )
    return modelo


# OPTIMAL THRESHOLD
def find_threshold(modelo, data):
    yp = modelo.predict_proba(data["X_val"])[:, 1]
    print(f"\n[THRESHOLD] Val rang: [{yp.min():.3f}, "
          f"{yp.max():.3f}]  mean={yp.mean():.3f}")
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.35, 0.66, 0.01):
        ypr = (yp >= t).astype(int)
        if len(np.unique(ypr)) < 2:
            continue
        pred_ratio = ypr.mean()
        if pred_ratio < 0.10 or pred_ratio > 0.90:
            continue
        ba = balanced_accuracy_score(data["y_val"], ypr)
        if ba > best_ba:
            best_ba, best_t = ba, t
    print(f"            Optimal: t={best_t:.2f}  BalAcc={best_ba:.4f}")
    return best_t


# EVALUATION
def evaluate(modelo, data, threshold, ticker, exp, horizon):
    yp    = modelo.predict_proba(data["X_test"])[:, 1]
    ypred = (yp >= threshold).astype(int)
    yt    = data["y_test"]

    m = {
        "accuracy"          : accuracy_score(yt, ypred),
        "balanced_accuracy" : balanced_accuracy_score(yt, ypred),
        "f1"                : f1_score(yt, ypred, zero_division=0),
        "roc_auc"           : roc_auc_score(yt, yp),
        "precision"         : precision_score(yt, ypred, zero_division=0),
        "recall"            : recall_score(yt, ypred, zero_division=0),
    }
    cm = confusion_matrix(yt, ypred)
    tn, fp = cm[0, 0], cm[0, 1]
    m["specificity"]     = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    m["pct_upward_test"] = round(float(np.mean(yt)), 3)

    # Actual prediction for today, last row of the entire dataset
    prob_today     = float(modelo.predict_proba(data["X_today"])[:, 1][0])
    m["prob_today"] = round(prob_today, 4)
    m["pred_today"] = "UP" if prob_today >= threshold else "DOWN"

    print(f"\n{'='*62}")
    print(f"  XGB RESULTS - Exp.{exp} {EXP_NAMES[exp]} | {horizon}d | {ticker}")
    print(f"{'='*62}")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.4f}")
        else:
            print(f"  {k:<22}: {v}")
    print()
    print(classification_report(yt, ypred, target_names=["Down", "Up"]))
    print(f"  TN={cm[0,0]:>4}  FP={cm[0,1]:>4}")
    print(f"  FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")
    print(f"  prob_today            : {prob_today:.4f}")
    print(f"  pred_today            : {m['pred_today']}")
    print(f"  pct_upward_test       : {float(np.mean(yt)):.2%}")
    print(f"{'='*62}")
    return yp, m


# INVESTOR SIGNAL
def investment_signal(metrics: dict, ticker: str, horizon: int):
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
        action     = "The model is not sufficiently reliable to recommend any stock."
        reason     = f"The ROC-AUC is {auc:.3f}, too close to a random model (0.50)."
    elif pred_today == "UP" and precision > 0.55 and specificity > 0.50:
        signal     = "BUY"
        signal_lbl = "SIGNAL: BUY"
        action     = (f"The model recommends buying {ticker} and holding\n"
                      f"  during the next {hor_txt}.")
        reason     = (f"In situations similar to the current one, the model has\n"
                      f"  correctly predicted the direction of UP in {precision*100:.1f}%\n"
                      f"  of the times in the evaluated period.")
    elif pred_today == "DOWN" and specificity > 0.55:
        signal     = "DO NOT BUY"
        signal_lbl = "SIGNAL: DO NOT BUY / SELL"
        action     = (f"The model recommends not buying {ticker} now.\n"
                      f"  If you have open positions, consider closing them.")
        reason     = (f"In scenarios similar to the current one, the model has\n"
                      f"  correctly predicted the direction of DOWN in {specificity*100:.1f}%\n"
                      f"  of the times in the evaluated period.")
    else:
        signal     = "INSUFFICIENT"
        signal_lbl = "INSUFFICIENT SIGNAL"
        action     = "The model does not generate a sufficiently clear signal to act on."
        reason     = "The reliability in both directions is similar or low."

    conf_txt = ("Moderate-High" if auc >= 0.70
                else "Moderate" if auc >= 0.60
                else "Low")

    W = 62
    print(f"\n{'='*W}")
    print(f"  {ticker} - {signal_lbl}")
    print(f"  {today}  |  Horizon: {hor_txt}")
    print(f"{'='*W}")
    print(f"  {action}")
    print(f"{'-'*W}")
    print(f"  Per que?")
    print(f"  {reason}")
    print(f"{'-'*W}")
    print(f"    Historical model performance (test set):")
    print(f"    Precision (up)  : {precision*100:.1f}%")
    print(f"    Precision (down) : {specificity*100:.1f}%")
    print(f"    Global accuracy      : {acc*100:.1f}%")
    print(f"    ROC-AUC              : {auc:.3f}")
    print(f"    Reliability           : {conf_txt}")
    print(f"{'='*W}")
    print(f"  WARNING: This prediction is for informational purposes only and does not")
    print(f"        constitute any investment advice.")
    print(f"{'='*W}\n")

    return signal


# CHARTS
def plot_results(modelo, data, study, yp, threshold,
                 ticker, exp, horizon, metrics):
    col   = EXP_COLORS[exp]
    yt    = data["y_test"]
    ypred = (yp >= threshold).astype(int)
    dates = pd.to_datetime(data["dates_test"])
    corr  = ypred == yt

    # Optuna: trial history
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"XGB Exp.{exp} | {ticker} | {horizon}d - Optuna",
                 fontweight="bold")
    vals = [t.value for t in study.trials]
    axes[0].plot(vals, color=col, lw=1.5, alpha=0.7)
    axes[0].axhline(max(vals), color="gray", ls="--", lw=1.2,
                    label=f"Best: {max(vals):.4f}")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("AUC val")
    axes[0].set_title("Optuna evolution", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Feature importances (top 15)
    fi = pd.Series(modelo.feature_importances_,
                   index=data["feature_names"]).sort_values(ascending=True)
    fi_top = fi.tail(15)
    fi_top.plot(kind="barh", ax=axes[1], color=col, alpha=0.8)
    axes[1].set_title("Feature Importance (top 15)", fontweight="bold")
    axes[1].set_xlabel("Importance")
    axes[1].grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()

    # Probabilities + Correct/Incorrect
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"XGB Exp.{exp} | {ticker} | {horizon}d | "
                 f"AUC={metrics['roc_auc']:.4f}  Acc={metrics['accuracy']:.4f}",
                 fontweight="bold")
    axes[0].plot(dates, yp, color=col, lw=0.9, alpha=0.9, label="P(upward)")
    axes[0].axhline(threshold, color="gray", ls="--", lw=1.3,
                    label=f"Threshold ({threshold:.2f})")
    axes[0].fill_between(dates, 0, yt * 0.08, alpha=0.3,
                         color="#10B981", label="Actual target")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Probability")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[1].scatter(dates[corr],  np.ones(corr.sum()),
                    color="#10B981", s=6, alpha=0.5, label="Correct")
    axes[1].scatter(dates[~corr], np.zeros((~corr).sum()),
                    color="#EF4444", s=6, alpha=0.5, label="Error")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Incorrect", "Correct"])
    axes[1].set_xlabel("Data")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(yt, yp)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f"XGB Exp.{exp} (AUC={metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.07, color=col)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC - XGB Exp.{exp} | {ticker} | {horizon}d",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Universal XGBoost for stock prediction - TFG"
    )
    parser.add_argument("--ticker",  type=str, default="TSLA",
                        help="Ticker (ex: TSLA, ITX.MC, BMW.DE, 7203.T)")
    parser.add_argument("--exp",     type=int, default=4, choices=[1, 2, 3, 4],
                        help="Experiment: 1=OHLCV 2=+Tec 3=+Macro 4=+Market")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 30],
                        help="Horizon: 1 o 30 days")
    parser.add_argument("--start",   type=str, default="2010-07-01",
                        help="Data inici (YYYY-MM-DD)")
    parser.add_argument("--end",     type=str, default="2026-03-01",
                        help="Data fi (YYYY-MM-DD)")
    parser.add_argument("--trials",  type=int, default=OPTUNA_TRIALS,
                        help=f"Trials Optuna (default: {OPTUNA_TRIALS})")
    args = parser.parse_args()

    TICKER  = args.ticker.upper()
    EXP     = args.exp
    HORIZON = args.horizon
    START   = args.start
    END     = args.end

    _, region          = get_fred_series(TICKER)
    mkt_t, _, mkt_name = get_market_config(TICKER)

    print(f"\n{'='*62}")
    print(f"  XGBoost UNIVERSAL - {TICKER} | Exp.{EXP} | {HORIZON}d")
    print(f"  Region: {region} | Mercat: {mkt_name}")
    print(f"{'='*62}\n")

    df     = load_stock(TICKER, START, END)
    macro  = None
    market = None

    if EXP >= 3:
        macro = load_macro(df.index, TICKER, START, END)

    if EXP >= 4:
        market, idx_close = load_market_context(df.index, TICKER, START, END)
        market = add_correlation(df, market, idx_close)

    data               = prepare_data(df, EXP, HORIZON, macro, market)
    best_params, study = optimize_xgb(data, HORIZON, n_trials=args.trials)
    modelo              = train_modelo(data, best_params)
    threshold          = find_threshold(modelo, data)
    yp, metrics        = evaluate(modelo, data, threshold, TICKER, EXP, HORIZON)
    investment_signal(metrics, TICKER, HORIZON)
    plot_results(modelo, data, study, yp, threshold,
                 TICKER, EXP, HORIZON, metrics)

    print(f"\n[DONE] XGB {TICKER} Exp.{EXP} {HORIZON}d | "
          f"AUC={metrics['roc_auc']:.4f} "
          f"Acc={metrics['accuracy']:.4f} "
          f"BalAcc={metrics['balanced_accuracy']:.4f}")