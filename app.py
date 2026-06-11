"""
app.py - Stock Market Prediction Web App
===================================
Streamlit app that runs 4 LSTM experiments + 4 XGBoost
for a 30-day horizon for any stock entered by the user.

Execution:
    streamlit run app.py

Dependencies:
    pip install streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
import warnings
import os
import sys
import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_MILDL"] = "3"

# Page configuration
st.set_page_config(
    page_title="StockPredict ML",
    page_icon="StockPredict ML",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #13131a;
    --border:    #1e1e2e;
    --accent:    #7c6af7;
    --green:     #22c55e;
    --red:       #ef4444;
    --yellow:    #eab308;
    --muted:     #6b7280;
    --text:      #e2e8f0;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
}

/* Input */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.1rem !important;
}

/* Main button */
[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* Result cards */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.result-card.buy   { border-left: 4px solid var(--green); }
.result-card.sell  { border-left: 4px solid var(--red); }
.result-card.none  { border-left: 4px solid var(--muted); }

.signal-buy  { color: var(--green);  font-weight: 700; font-size: 1.1rem; }
.signal-sell { color: var(--red);    font-weight: 700; font-size: 1.1rem; }
.signal-none { color: var(--muted);  font-weight: 700; font-size: 1.1rem; }

.metric-row { display: flex; gap: 2rem; margin-top: 0.5rem; flex-wrap: wrap; }
.metric-item { font-size: 0.82rem; color: var(--muted); }
.metric-item span { color: var(--text); font-weight: 600; }

.exp-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: var(--accent);
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.model-label {
    font-size: 0.75rem;
    background: var(--border);
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    margin-right: 0.4rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
}

.header-bar {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.2rem;
}

.disclaimer {
    font-size: 0.72rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    padding-top: 0.8rem;
    margin-top: 1.5rem;
}

/* Streamlit divider */
hr { border-color: var(--border) !important; }

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: var(--accent) !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Info / warning */
[data-testid="stInfo"] {
    background: var(--surface) !important;
    border-color: var(--accent) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# Heavy imports (loaded once)
@st.cache_resource
def load_heavy_imports():
    import yfinance as yf
    import xgboost as xgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import regularizers
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        roc_auc_score, precision_score, recall_score,
        confusion_matrix, balanced_accuracy_score,
    )
    from sklearn.isotonic import IsotonicRegression
    return True

load_heavy_imports()


# FRED regional configuration
SUFFIX_TO_REGION = {
    "": "US", "US": "US",
    "MC": "EU", "MA": "EU", "DE": "EU", "F": "EU",
    "PA": "EU", "AS": "EU", "MI": "EU",
    "L": "UK", "T": "JP", "HK": "GLOBAL",
    "SS": "GLOBAL", "TO": "GLOBAL", "AX": "GLOBAL", "SW": "GLOBAL",
}
FRED_SERIES = {
    "US": {"DFF": "fed_rate", "T10Y2Y": "yield_curve", "CPIAUCSL": "cpi",
           "T10YIE": "inflation_exp", "VIXCLS": "vix", "BAMLH0A0HYM2": "hy_spread",
           "UNRATE": "unemployment", "INDPRO": "indprod"},
    "EU": {"ECBDFR": "fed_rate", "IRLTLT01EZM156N": "yield_curve",
           "CP0000EZ17M086NEST": "cpi", "VIXCLS": "vix",
           "BAMLHE00EHYIOAS": "hy_spread", "LRHUTTTTEZM156S": "unemployment",
           "PRMNTO01EZQ661S": "indprod"},
    "UK": {"IUDSOIA": "fed_rate", "IRLTLT01GBM156N": "yield_curve",
           "GBRCPIALLMINMEI": "cpi", "VIXCLS": "vix", "LRHUTTTTGBM156S": "unemployment"},
    "JP": {"IRSTCB01JPM156N": "fed_rate", "IRLTLT01JPM156N": "yield_curve",
           "JPNCPIALLMINMEI": "cpi", "VIXCLS": "vix", "LRHUTTTTJPM156S": "unemployment"},
    "GLOBAL": {"VIXCLS": "vix", "BAMLH0A0HYM2": "hy_spread", "T10Y2Y": "yield_curve"},
}
SUFFIX_TO_MARKET = {
    "": ("^GSPC", "^STOXX50E", "S&P 500"),
    "US": ("^GSPC", "^STOXX50E", "S&P 500"),
    "MC": ("^IBEX", "^GSPC", "IBEX 35"),
    "MA": ("^IBEX", "^GSPC", "IBEX 35"),
    "DE": ("^GDAXI", "^GSPC", "DAX"),
    "F": ("^GDAXI", "^GSPC", "DAX"),
    "PA": ("^FCHI", "^GSPC", "CAC 40"),
    "AS": ("^AEX", "^GSPC", "AEX"),
    "MI": ("FTSEMIB.MI", "^GSPC", "FTSE MIB"),
    "L": ("^FTSE", "^GSPC", "FTSE 100"),
    "T": ("^N225", "^GSPC", "Nikkei 225"),
    "HK": ("^HSI", "^GSPC", "Hang Seng"),
    "SS": ("000001.SS", "^GSPC", "SSE"),
    "TO": ("^GSPTSE", "^GSPC", "TSX"),
    "AX": ("^AXJO", "^GSPC", "ASX 200"),
    "SW": ("^SSMI", "^GSPC", "SMI"),
}
FRED_API_KEY = "d1579a90b1b46f86b9b802630f4c5fda"
START_DATE = "2010-07-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")
WINDOW_SIZE = 20
BATCH_SIZE = 64
SEED = 42
HORIZON = 30


# Helpers regionals
def get_region(ticker):
    parts = ticker.upper().split(".")
    suffix = parts[-1] if len(parts) > 1 else ""
    return SUFFIX_TO_REGION.get(suffix, "US"), suffix

def get_market_config(ticker):
    parts = ticker.upper().split(".")
    suffix = parts[-1] if len(parts) > 1 else ""
    cfg = SUFFIX_TO_MARKET.get(suffix, ("^GSPC", "^STOXX50E", "S&P 500"))
    return cfg


# Data pipeline (shared across all experiments)
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock(ticker):
    import yfinance as yf
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_macro(ticker):
    import requests
    region, _ = get_region(ticker)
    series_map = FRED_SERIES.get(region, FRED_SERIES["GLOBAL"])
    frames = {}
    for sid, fname in series_map.items():
        try:
            url = (f"https://api.stlouisfed.org/fred/series/observations"
                   f"?series_id={sid}&api_key={FRED_API_KEY}&file_type=json"
                   f"&observation_start={START_DATE}&observation_end={END_DATE}")
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                obs = r.json().get("observations", [])
                s = pd.Series(
                    {o["date"]: float(o["value"])
                     for o in obs if o["value"] != "."},
                    name=fname, dtype=float)
                s.index = pd.to_datetime(s.index)
                frames[fname] = s
        except Exception:
            pass
    if not frames:
        return None
    macro_df = pd.DataFrame(frames)
    macro_df = macro_df.resample("B").last().ffill().bfill()
    return macro_df


MACRO_TRANSFORMS_GEN = {
    "fed_rate"     : ("diff",          "fed_rate_chg"),
    "yield_curve"  : ("identity",      "yield_curve"),
    "cpi"          : ("pct_change_12", "cpi_yoy"),
    "inflation_exp": ("identity",      "inflation_exp"),
    "vix"          : ("log",           "vix_log"),
    "hy_spread"    : ("diff",          "hy_spread_chg"),
    "unemployment" : ("diff",          "unemployment_chg"),
    "indprod"      : ("pct_change",    "indprod_chg"),
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_generalista(_stock_index):
    """
    Loads US macro data with the same transformations as train_generalista.py.
    Necessary for compatibility with the pre-trained scaler.

    Returns: fed_rate_chg, yield_curve, cpi_yoy, inflation_exp,
     vix_log, hy_spread_chg, unemployment_chg, indprod_chg, vix_chg

    Total: 9 columns.
    """
    import requests
    FRED_US_GEN = [
        ("DFF",          "fed_rate"),
        ("T10Y2Y",       "yield_curve"),
        ("CPIAUCSL",     "cpi"),
        ("T10YIE",       "inflation_exp"),
        ("VIXCLS",       "vix"),
        ("BAMLH0A0HYM2", "hy_spread"),
        ("UNRATE",       "unemployment"),
        ("INDPRO",       "indprod"),
    ]
    frames = {}
    for sid, col in FRED_US_GEN:
        try:
            url = (f"https://api.stlouisfed.org/fred/series/observations"
                   f"?series_id={sid}&api_key={FRED_API_KEY}&file_type=json"
                   f"&observation_start={START_DATE}&observation_end={END_DATE}")
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                obs = r.json().get("observations", [])
                s = pd.Series(
                    {pd.Timestamp(o["date"]): float(o["value"])
                     for o in obs if o["value"] != "."},
                    name=col, dtype=float)
                s.index = pd.to_datetime(s.index).tz_localize(None)
                frames[col] = s
        except Exception:
            pass

    if not frames:
        return None

    transformed = []
    for col, s in {c: frames[c].dropna() for c in frames}.items():
        tr, new_name = MACRO_TRANSFORMS_GEN.get(col, ("identity", col))
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
    sidx = pd.to_datetime(_stock_index).tz_localize(None).normalize()
    return macro_daily.reindex(sidx).replace([np.inf, -np.inf], np.nan)


@st.cache_data(ttl=3600, show_spinner=False)
def load_market(ticker):
    import yfinance as yf
    idx1, idx2, _ = get_market_config(ticker)
    frames = {}
    for sym in [idx1, idx2]:
        try:
            d = yf.download(sym, start=START_DATE, end=END_DATE,
                            auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            frames[sym] = d["Close"]
        except Exception:
            pass
    if not frames:
        return None
    idx_close = frames.get(idx1)
    mkt = pd.DataFrame()
    if idx1 in frames:
        r = np.log(frames[idx1] / frames[idx1].shift(1))
        mkt["index_ret_1d"] = r
        mkt["index_ret_5d"] = np.log(frames[idx1] / frames[idx1].shift(5))
        mkt["index_vol"] = r.rolling(20).std()
    if idx2 in frames:
        r2 = np.log(frames[idx2] / frames[idx2].shift(1))
        mkt["stoxx_ret_1d"] = r2
        mkt["stoxx_vol"] = r2.rolling(20).std()
    if idx_close is not None and idx2 in frames:
        mkt["index_corr"] = (
            mkt.get("index_ret_1d", pd.Series(dtype=float))
            .rolling(20).corr(mkt.get("stoxx_ret_1d", pd.Series(dtype=float)))
        )
    return mkt.replace([np.inf, -np.inf], np.nan), idx_close


def build_features_ohlcv(df):
    C, O, H, L, V = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    rng = (H - L).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["log_ret_1d"]   = np.log(C / C.shift(1))
    out["log_ret_5d"]   = np.log(C / C.shift(5))
    out["log_ret_10d"]  = np.log(C / C.shift(10))
    out["range_pct"]    = (H - L) / C
    out["body_pct"]     = (C - O).abs() / rng
    out["overnight_gap"] = (O / C.shift(1)) - 1
    out["vol_ratio"]    = V / V.rolling(20).mean()
    out["realized_vol"] = out["log_ret_1d"].rolling(10).std()
    out["close_sma20"]  = (C / C.rolling(20).mean()) - 1
    out["dir_5d"]       = (out["log_ret_1d"] > 0).rolling(5).mean()
    return out.replace([np.inf, -np.inf], np.nan)

def build_features_technical(df):
    C, H, L, V = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)

    # Trend
    ema21 = C.ewm(span=21, adjust=False).mean()
    ema50 = C.ewm(span=50, adjust=False).mean()
    out["ema_ratio_21_50"] = (ema21 / ema50) - 1

    tr    = pd.concat([H - L, (H - C.shift(1)).abs(), (L - C.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    up    = H - H.shift(1)
    down  = L.shift(1) - L
    dm_p  = up.where((up > down) & (up > 0), 0.0)
    dm_m  = down.where((down > up) & (down > 0), 0.0)
    di_p  = 100 * dm_p.rolling(14).mean() / (atr14 + 1e-9)
    di_m  = 100 * dm_m.rolling(14).mean() / (atr14 + 1e-9)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    out["adx"] = dx.rolling(14).mean() / 100

    # Momentum
    delta = C.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = (100 - 100 / (1 + gain / (loss + 1e-9))) / 100
    out["roc_10"] = C.pct_change(10)

    # Volatility
    sma20  = C.rolling(20).mean()
    std20  = C.rolling(20).std()
    bb_up  = sma20 + 2 * std20
    bb_low = sma20 - 2 * std20
    out["bb_width"] = (bb_up - bb_low) / (sma20 + 1e-9)
    out["atr_14"]   = atr14 / (C + 1e-9)

    # Volume
    mfv = ((C - L) - (H - C)) / (H - L + 1e-9) * V
    out["cmf"] = mfv.rolling(20).sum() / (V.rolling(20).sum() + 1e-9)

    obv      = (np.sign(C.diff()) * V).fillna(0).cumsum()
    obv_norm = obv / (obv.rolling(50).std() + 1e-9)
    out["obv_slope"] = obv_norm.diff(10) / 10

    # Support / Resistance
    max50 = H.rolling(50).max()
    min50 = L.rolling(50).min()
    out["price_position"] = (C - min50) / (max50 - min50 + 1e-9)
    out["dist_max_50"]    = (max50 - C) / (C + 1e-9)

    return out.replace([np.inf, -np.inf], np.nan)

def build_target(df):
    C = df["Close"]
    future_mean = C.shift(-HORIZON).rolling(HORIZON).mean().shift(-(HORIZON - 1))
    past_mean   = C.rolling(HORIZON).mean()
    return (future_mean > past_mean).astype(int).rename("target")

def prepare_sequences(X_raw, y_raw, dates):
    np.random.seed(SEED)
    n_train = int(len(X_raw) * 0.70)
    n_val   = int(len(X_raw) * 0.15)

    X_tr = X_raw[:n_train]
    X_vl = X_raw[n_train:n_train + n_val]
    X_te = X_raw[n_train + n_val:]
    y_tr = y_raw[:n_train]
    y_vl = y_raw[n_train:n_train + n_val]
    y_te = y_raw[n_train + n_val:]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_te = scaler.transform(X_te)

    def make_seqs(X, y):
        return (np.array([X[i-WINDOW_SIZE:i] for i in range(WINDOW_SIZE, len(X))]),
                np.array([y[i] for i in range(WINDOW_SIZE, len(y))]))

    Xtr, ytr = make_seqs(X_tr, y_tr)
    Xvl, yvl = make_seqs(X_vl, y_vl)
    Xte, yte = make_seqs(X_te, y_te)

    # Last 20 days of the entire dataset (includes data after test period)
    # These are the most recent days available - real prediction for today
    X_tot_scaled = np.concatenate([X_tr, X_vl, X_te], axis=0)
    X_today = X_tot_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, X_raw.shape[1])

    return {"X_train": Xtr, "y_train": ytr, "X_val": Xvl, "y_val": yvl,
            "X_test": Xte, "y_test": yte, "scaler": scaler,
            "X_today": X_today}

def prepare_flat(X_raw, y_raw):
    from sklearn.preprocessing import StandardScaler
    n_train = int(len(X_raw) * 0.70)
    n_val   = int(len(X_raw) * 0.15)

    X_tr = X_raw[:n_train]
    X_vl = X_raw[n_train:n_train + n_val]
    X_te = X_raw[n_train + n_val:]
    y_tr = y_raw[:n_train]
    y_vl = y_raw[n_train:n_train + n_val]
    y_te = y_raw[n_train + n_val:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_te = scaler.transform(X_te)

    # Last row of the entire dataset - real prediction for today (XGBoost)
    X_tot_scaled = np.concatenate([X_tr, X_vl, X_te], axis=0)
    X_today = X_tot_scaled[-1:, :]

    return {"X_train": X_tr, "y_train": y_tr, "X_val": X_vl, "y_val": y_vl,
            "X_test": X_te, "y_test": y_te, "X_today": X_today}


# LSTM training
def train_lstm(data, n_features, exp, stacked=False):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import regularizers

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    configs = {
        1: {"units": 48, "dropout": 0.4, "l2": 2e-3, "dense": 8,  "double": False},
        2: {"units": 48, "dropout": 0.5, "l2": 3e-3, "dense": 16, "double": True},
        3: {"units": 64, "dropout": 0.5, "l2": 3e-3, "dense": 16, "double": True},
        4: {"units": 64, "dropout": 0.5, "l2": 3e-3, "dense": 16, "double": True},
    }
    cfg = configs[exp]

    if stacked:
        # Stacked LSTM: 2 layers - captures more complex temporal patterns
        # Higher regularization to compensate for extra parameters
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
            Dense(cfg.get("dense", 16), activation="relu",
                  kernel_regularizer=regularizers.l2(cfg["l2"])),
            Dropout(cfg["dropout"] * 0.5),
            Dense(1, activation="sigmoid"),
        ]
        arch_name = f"LSTM_Stacked_Exp{exp}"
    else:
        # Standard LSTM: 1 layer (original architecture)
        layers = [
            LSTM(cfg["units"], input_shape=(WINDOW_SIZE, n_features),
                 kernel_regularizer=regularizers.l2(cfg["l2"]),
                 recurrent_regularizer=regularizers.l2(cfg["l2"]),
                 dropout=cfg["dropout"], recurrent_dropout=0.2),
            Dropout(cfg["dropout"]),
            Dense(cfg.get("dense", 16), activation="relu",
                  kernel_regularizer=regularizers.l2(cfg["l2"])),
        ]
        if cfg["double"]:
            layers.append(Dropout(cfg["dropout"]))
        layers.append(Dense(1, activation="sigmoid"))
        arch_name = f"LSTM_Standard_Exp{exp}"

    model = Sequential(layers, name=arch_name)
    model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    cbs = [
        EarlyStopping(monitor="val_auc", mode="max",
                      patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_auc", mode="max",
                          factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]
    model.fit(data["X_train"], data["y_train"],
              validation_data=(data["X_val"], data["y_val"]),
              epochs=150, batch_size=BATCH_SIZE, callbacks=cbs, verbose=0)
    return model


def eval_lstm(model, data):
    from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                                  accuracy_score, f1_score, precision_score,
                                  recall_score, confusion_matrix)
    y_prob_val = model.predict(data["X_val"], verbose=0).flatten()
    y_val = data["y_val"]

    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.35, 0.66, 0.01):
        pred = (y_prob_val >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        pred_ratio = pred.mean()
        if pred_ratio < 0.10 or pred_ratio > 0.90:
            continue
        ba = balanced_accuracy_score(y_val, pred)
        if ba > best_ba:
            best_ba, best_t = ba, t

    y_prob = model.predict(data["X_test"], verbose=0).flatten()
    y_pred = (y_prob >= best_t).astype(int)
    y_test = data["y_test"]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0

    # REAL PREDICTION TODAY - last 20 days of the entire dataset
    # NOT the last test window (which may be months old)
    # instead, use the 20 most recent available days.
    last_window = data["X_today"]  # shape (1, 20, n_features)
    prob_today = float(model.predict(last_window, verbose=0).flatten()[0])
    pred_today = "UP" if prob_today >= best_t else "DOWN"

    return {
        "roc_auc":           round(roc_auc_score(y_test, y_prob), 4),
        "accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "f1":                round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision":         round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_test, y_pred, zero_division=0), 4),
        "specificity":       specificity,
        "threshold":         round(best_t, 2),
        "prob_today":          round(prob_today, 4),
        "pred_today":          pred_today,
        "pct_upward_test":  round(float(np.mean(y_test)), 3),
    }


# XGBoost training
def train_xgb(data):
    import xgboost as xgb
    import optuna

    X_tr, y_tr = data["X_train"], data["y_train"]
    X_vl, y_vl = data["X_val"],   data["y_val"]
    pos = int((y_tr == 0).sum())
    neg = int((y_tr == 1).sum())
    spw = pos / neg if neg > 0 else 1.0

    def objective(trial):
        from sklearn.metrics import balanced_accuracy_score
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "scale_pos_weight": spw,
            "random_state": SEED, "eval_metric": "auc", "verbosity": 0,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        p = m.predict(X_vl)
        return balanced_accuracy_score(y_vl, p)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best = study.best_params
    best["scale_pos_weight"] = spw
    best["random_state"] = SEED
    best["eval_metric"] = "auc"
    best["verbosity"] = 0
    model = xgb.XGBClassifier(**best)
    model.fit(X_tr, y_tr)
    return model


def eval_xgb(model, data):
    from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                                  accuracy_score, f1_score, precision_score,
                                  recall_score, confusion_matrix)
    y_prob_val = model.predict_proba(data["X_val"])[:, 1]
    y_val = data["y_val"]

    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.35, 0.66, 0.01):
        pred = (y_prob_val >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        pred_ratio = pred.mean()
        if pred_ratio < 0.10 or pred_ratio > 0.90:
            continue
        ba = balanced_accuracy_score(y_val, pred)
        if ba > best_ba:
            best_ba, best_t = ba, t

    y_prob = model.predict_proba(data["X_test"])[:, 1]
    y_pred = (y_prob >= best_t).astype(int)
    y_test = data["y_test"]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0

    # REAL PREDICTION TODAY - last row of the entire dataset
    # NOT the last test row (which may be months old)
    # instead, use the most recent available day.
    last_row = data["X_today"]  # shape (1, n_features)
    prob_today   = float(model.predict_proba(last_row)[:, 1][0])
    pred_today   = "UP" if prob_today >= best_t else "DOWN"

    return {
        "roc_auc":           round(roc_auc_score(y_test, y_prob), 4),
        "accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "f1":                round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision":         round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_test, y_pred, zero_division=0), 4),
        "specificity":       specificity,
        "threshold":         round(best_t, 2),
        "prob_today":          round(prob_today, 4),
        "pred_today":          pred_today,
        "pct_upward_test":  round(float(np.mean(y_test)), 3),
    }


# Investment signal

# FAST MODE - Pre-trained generalist model

@st.cache_resource
def load_generalista_models():
    """
    Load pre-trained generalist models from disk.
    Runs once thanks to cache_resource.
    Returns None if files do not exist.
    """
    import json
    import joblib
    import os

    files = [
        "models/model_lstm_generalista.keras",
        "models/model_xgb_generalista.pkl",
        "models/scaler_generalista.pkl",
        "models/threshold_generalista.json",
    ]
    if not all(os.path.exists(f) for f in files):
        return None

    try:
        import tensorflow as tf
        lstm    = tf.keras.models.load_model("models/model_lstm_generalista.keras")
        xgb_m   = joblib.load("models/model_xgb_generalista.pkl")
        scaler  = joblib.load("models/scaler_generalista.pkl")
        with open("models/threshold_generalista.json") as f:
            thresholds = json.load(f)
        return {"lstm": lstm, "xgb": xgb_m,
                "scaler": scaler, "thresholds": thresholds}
    except Exception:
        return None


def build_features_exp4(df, macro_df, market_df, idx_close=None):
    """
    Builds the complete Exp.4 features for fast mode.
    IMPORTANT: it must generate exactly the same features as train_generalista.py
    to ensure compatibility with the pre-trained scaler.

    index_corr = 20-day rolling correlation between the STOCK and the INDEX,
    not between the two indices. idx_close is needed to compute it.
    """
    f1 = build_features_ohlcv(df)
    f2 = build_features_technical(df)
    features = pd.concat([f1, f2], axis=1)

    if macro_df is not None:
        macro_aligned = macro_df.reindex(features.index).ffill().bfill()
        features = pd.concat([features, macro_aligned], axis=1)

    if market_df is not None:
        mkt_aligned = market_df.reindex(features.index).ffill().bfill()

        # Recalculate index_corr as stock vs index correlation
        if idx_close is not None:
            stock_idx  = pd.to_datetime(df.index).tz_localize(None).normalize()
            ia         = idx_close.reindex(stock_idx, method="ffill")
            sr         = np.log(df["Close"] / df["Close"].shift(1))
            ir         = np.log(ia / ia.shift(1))
            corr       = sr.rolling(20).corr(ir)
            corr.index = stock_idx
            mkt_aligned = mkt_aligned.copy()
            mkt_aligned["index_corr"] = corr.values

        features = pd.concat([features, mkt_aligned], axis=1)

    return features.replace([np.inf, -np.inf], np.nan)


def run_mode_rapid(ticker, df, macro_df, market_df_raw, models):
    """
    Runs prediction with the pre-trained generalist models.
    No retraining. Loads models from disk and predicts in seconds.

    Returns test metrics and signal for LSTM and XGBoost.
    """
    WINDOW_SIZE = 20
    HORIZON     = 30

    # Extract market_df and idx_close from tuple returned by load_market
    market_df = market_df_raw[0] if market_df_raw else None
    idx_close = market_df_raw[1] if market_df_raw else None

    # Feature construction Exp.4
    # We pass idx_close to correctly compute index_corr (stock vs index)
    features = build_features_exp4(df, macro_df, market_df, idx_close)
    target   = build_target(df)

    idx      = features.index.intersection(target.index)
    features = features.loc[idx].ffill().bfill().dropna()
    target_s = target.loc[features.index]
    features = features.iloc[:-HORIZON]
    target_s = target_s.iloc[:-HORIZON]

    X_raw = features.values.astype(np.float32)
    y_raw = target_s.values.astype(np.float32)

    # Scale with pre-trained scaler (no fit)
    scaler = models["scaler"]
    n_features_model  = scaler.n_features_in_
    n_features_actual = X_raw.shape[1]

    if n_features_model != n_features_actual:
        st.warning(
            f"Feature incompatibility: model expects "
            f"{n_features_model} features but got "
            f"{n_features_actual}. Re-run train_generalista.py."
        )
        return None

    try:
        X_scaled = scaler.transform(X_raw)
    except Exception as e:
        st.warning(f"Scaling error: {e}")
        return None

    # Temporal split for honest evaluation
    n       = len(X_scaled)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    X_te    = X_scaled[n_train + n_val:]
    y_te    = y_raw[n_train + n_val:]

    if len(X_te) < WINDOW_SIZE + 10:
        return None

    # Sequences for LSTM
    X_seq = np.array([X_te[i-WINDOW_SIZE:i]
                      for i in range(WINDOW_SIZE, len(X_te))])
    y_seq = y_te[WINDOW_SIZE:]

    # Real prediction today - using the last 20 days of the entire dataset
    X_today_lstm = X_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, X_scaled.shape[1])
    X_today_xgb  = X_scaled[-1:, :]

    def calc_metrics(y_prob, y_true, threshold):
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, balanced_accuracy_score,
            f1_score, precision_score, recall_score, confusion_matrix,
        )
        y_pred = (y_prob >= threshold).astype(int)
        if len(np.unique(y_pred)) < 2:
            return None
        cm     = confusion_matrix(y_true, y_pred)
        tn, fp = cm[0, 0], cm[0, 1]
        spec   = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
        return {
            "roc_auc"          : round(roc_auc_score(y_true, y_prob), 4),
            "accuracy"         : round(accuracy_score(y_true, y_pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
            "f1"               : round(f1_score(y_true, y_pred, zero_division=0), 4),
            "precision"        : round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall"           : round(recall_score(y_true, y_pred, zero_division=0), 4),
            "specificity"      : spec,
            "threshold"        : threshold,
            "pct_upward_test" : round(float(np.mean(y_true)), 3),
        }

    # Prediction LSTM
    lstm_metrics  = None
    lstm_prob_today = None
    lstm_pred_today = None
    try:
        prob_lstm     = models["lstm"].predict(X_seq, verbose=0).flatten()
        lstm_metrics  = calc_metrics(prob_lstm, y_seq, models["thresholds"]["lstm"])
        # Real prediction today - using the last 20 days of the entire dataset
        t_lstm        = models["thresholds"]["lstm"]
        ultima_lstm   = X_today_lstm
        lstm_prob_today = float(models["lstm"].predict(ultima_lstm, verbose=0).flatten()[0])
        lstm_pred_today = "UP" if lstm_prob_today >= t_lstm else "DOWN"
        if lstm_metrics:
            lstm_metrics["prob_today"] = round(lstm_prob_today, 4)
            lstm_metrics["pred_today"] = lstm_pred_today
    except Exception:
        pass

    # Prediction XGBoost
    xgb_metrics   = None
    xgb_prob_today = None
    xgb_pred_today = None
    try:
        prob_xgb    = models["xgb"].predict_proba(X_te)[:, 1]
        xgb_metrics = calc_metrics(prob_xgb, y_te, models["thresholds"]["xgb"])
        # Real prediction today - last day of the entire dataset
        t_xgb         = models["thresholds"]["xgb"]
        ultima_xgb    = X_today_xgb
        xgb_prob_today = float(models["xgb"].predict_proba(ultima_xgb)[:, 1][0])
        xgb_pred_today = "UP" if xgb_prob_today >= t_xgb else "DOWN"
        if xgb_metrics:
            xgb_metrics["prob_today"] = round(xgb_prob_today, 4)
            xgb_metrics["pred_today"] = xgb_pred_today
    except Exception:
        pass

    if lstm_metrics is None and xgb_metrics is None:
        return None

    # Signal based on pred_today + test metrics
    lstm_signal = get_signal(lstm_metrics)[0] if lstm_metrics else "INSUFFICIENT"
    xgb_signal  = get_signal(xgb_metrics)[0]  if xgb_metrics else "INSUFFICIENT"

    return {
        "lstm": {
            "metrics"    : lstm_metrics or {},
            "signal"     : lstm_signal,
            "icon"       : "",
            "specificity": lstm_metrics["specificity"] if lstm_metrics else 0.0,
        },
        "xgb": {
            "metrics"    : xgb_metrics or {},
            "signal"     : xgb_signal,
            "icon"       : "",
            "specificity": xgb_metrics["specificity"] if xgb_metrics else 0.0,
        },
    }


def get_signal(metrics):
    """
    Determine the investment signal by combining:

    pred_today: prediction from the LAST WINDOW (the last 20 real days)
    → indicates whether the model believes TODAY is UP or DOWN
    test metrics: AUC, Precision, Specificity
    → indicate whether we can TRUST this prediction

    If the model is not reliable (AUC < 0.55) → INSUFFICIENT SIGNAL, regardless of what the last window says.
    """
    auc       = metrics["roc_auc"]
    prec      = metrics["precision"]
    rec       = metrics["recall"]
    spec      = metrics["specificity"]
    pred_today = metrics.get("pred_today", None)

    # First filter: is the model reliable enough?
    if auc < 0.55:
        return "INSUFFICIENT", "", spec

    # Second filter: what does the model predict TODAY (last window)?
    if pred_today == "UP":
        # Model predicts UP today - but is it reliable for upward moves?
        if prec > 0.55 and spec > 0.50:
            return "BUY", "", spec
        else:
            return "INSUFFICIENT", "", spec
    elif pred_today == "DOWN":
        # Model predicts DOWN today - but is it reliable for downward moves?
        if spec > 0.55:
            return "DO NOT BUY", "", spec
        else:
            return "INSUFFICIENT", "", spec
    else:
        # Fallback if pred_today is not available (compatibility)
        if rec > spec and prec > 0.55 and spec > 0.50:
            return "BUY", "", spec
        elif spec > rec and spec > 0.55:
            return "DO NOT BUY", "", spec
        else:
            return "INSUFFICIENT", "", spec


def conf_label(auc):
    if auc >= 0.70:
        return "Moderate-High"
    elif auc >= 0.60:
        return "Moderate"
    return "Low"


# Complete pipeline for the experiment
def run_experiment(ticker, exp, df, macro_df, market_df, stacked=False):
    """Run LSTM + XGBoost for an experiment. Return a dictionary with results."""

    # Feature engineering
    f1 = build_features_ohlcv(df)
    target = build_target(df)

    if exp == 1:
        features = f1
    elif exp == 2:
        f2 = build_features_technical(df)
        features = pd.concat([f1, f2], axis=1)
    elif exp == 3:
        f2 = build_features_technical(df)
        features = pd.concat([f1, f2], axis=1)
        if macro_df is not None:
            macro_aligned = macro_df.reindex(features.index).ffill().bfill()
            macro_diff = macro_aligned.diff()
            macro_diff.columns = [c + "_chg" for c in macro_diff.columns]
            features = pd.concat([features, macro_aligned, macro_diff], axis=1)
    else:  # exp == 4
        f2 = build_features_technical(df)
        features = pd.concat([f1, f2], axis=1)
        if macro_df is not None:
            macro_aligned = macro_df.reindex(features.index).ffill().bfill()
            macro_diff = macro_aligned.diff()
            macro_diff.columns = [c + "_chg" for c in macro_diff.columns]
            features = pd.concat([features, macro_aligned, macro_diff], axis=1)
        if market_df is not None:
            mkt_aligned = market_df.reindex(features.index).ffill().bfill()
            features = pd.concat([features, mkt_aligned], axis=1)

    # Align
    idx = features.index.intersection(target.index)
    features = features.loc[idx].dropna()
    target_s = target.loc[features.index]
    features = features.iloc[:-HORIZON]
    target_s = target_s.iloc[:-HORIZON]

    X_raw = features.values.astype(np.float32)
    y_raw = target_s.values.astype(np.float32)

    # LSTM
    data_seq = prepare_sequences(X_raw, y_raw, features.index)
    lstm_model = train_lstm(data_seq, X_raw.shape[1], exp, stacked=stacked)
    lstm_metrics = eval_lstm(lstm_model, data_seq)
    lstm_signal, lstm_icon, lstm_spec = get_signal(lstm_metrics)

    # XGBoost
    data_flat = prepare_flat(X_raw, y_raw)
    xgb_model = train_xgb(data_flat)
    xgb_metrics = eval_xgb(xgb_model, data_flat)
    xgb_signal, xgb_icon, xgb_spec = get_signal(xgb_metrics)

    return {
        "lstm": {"metrics": lstm_metrics, "signal": lstm_signal,
                 "icon": lstm_icon, "specificity": lstm_spec},
        "xgb":  {"metrics": xgb_metrics,  "signal": xgb_signal,
                 "icon": xgb_icon,  "specificity": xgb_spec},
    }


# Render experiment results.
def render_exp_result(exp_num, exp_name, result):
    for model_key, label in [("lstm", "LSTM"), ("xgb", "XGBoost")]:
        r = result[model_key]
        m = r["metrics"]
        signal = r["signal"]
        spec = r["specificity"]

        card_class = {"BUY": "buy", "DO NOT BUY": "sell"}.get(signal, "none")
        sig_class  = {"BUY": "signal-buy", "DO NOT BUY": "signal-sell"}.get(signal, "signal-none")

        # Prediction from the last window (today)
        prob_today = m.get("prob_today", None)
        pred_today = m.get("pred_today", None)
        pred_color = "#22c55e" if pred_today == "UP" else "#ef4444" if pred_today == "DOWN" else "#6b7280"

        if signal == "BUY":
            action_txt = "The model recommends <b>BUY</b> and to hold for 30 days."
            why_txt = f"Accuracy up: <b>{m['precision']*100:.1f}%</b> - Accuracy down: <b>{spec*100:.1f}%</b>"
        elif signal == "DO NOT BUY":
            action_txt = "The model recommends <b>DO NOT BUY</b> now."
            why_txt = f"Accuracy up: <b>{m['precision']*100:.1f}%</b> - Accuracy down: <b>{spec*100:.1f}%</b>"
        else:
            action_txt = "The model <b>does not generate a sufficiently clear signal</b> to act."
            why_txt = f"Accuracy up: <b>{m['precision']*100:.1f}%</b> - Accuracy down: <b>{spec*100:.1f}%</b>"

        # Separate construction of HTML to avoid issues with nested f-strings.
        pred_today_block = ""
        if prob_today is not None and pred_today is not None:
            pred_today_block = (
                f'<div style="margin-top:0.6rem; padding:0.5rem 0.8rem;'
                f'background:#0a0a0f; border-radius:6px;'
                f'border-left:3px solid {pred_color};">'
                f'<span style="font-size:0.78rem; color:#6b7280;">Prediction today (last window): </span>'
                f'<span style="color:{pred_color}; font-weight:700; margin-left:0.5rem;">{pred_today}</span>'
                f'<span style="color:#6b7280; font-size:0.78rem; margin-left:0.5rem;">'
                f'(prob: {prob_today:.3f} | threshold: {m["threshold"]:.2f})</span>'
                f'</div>'
            )

        # Warning if the market was strongly bullish during the test period.
        pct_sub = m.get('pct_upward_test', None)
        bias_warning = ''
        if pct_sub is not None and pct_sub > 0.60:
            bias_warning = (
                f'<div style="font-size:0.75rem; color:#eab308; margin-top:0.4rem;'
                f'padding:0.3rem 0.6rem; background:#1a1500; border-radius:4px;'
                f'border-left:2px solid #eab308;">'
                f'Warning: on {pct_sub*100:.0f}% of test days were upward. '
                f'Metrics may be inflated due to bullish trend.'
                f'</div>'
            )

        card_html = (
            f'<div class="result-card {card_class}">'
            f'<div class="exp-title">Exp.{exp_num} - {exp_name}'
            f'<span class="model-label">{label}</span></div>'
            f'<div class="header-bar"><span class="{sig_class}">{signal}</span></div>'
            f'<div style="font-size:0.88rem; margin:0.4rem 0 0.6rem 0; color:#cbd5e1;">{action_txt}</div>'
            f'{pred_today_block}'
            f'<div class="metric-row" style="margin-top:0.6rem;">'
            f'<div class="metric-item">ROC-AUC <span>{m["roc_auc"]:.3f}</span></div>'
            f'<div class="metric-item">Accuracy <span>{m["accuracy"]*100:.1f}%</span></div>'
            f'<div class="metric-item">Precision (up) <span>{m["precision"]*100:.1f}%</span></div>'
            f'<div class="metric-item">Precision (down) <span>{spec*100:.1f}%</span></div>'
            f'<div class="metric-item">Reliability <span>{conf_label(m["roc_auc"])}</span></div>'
            f'<div class="metric-item">Upward test <span>{pct_sub*100:.0f}%</span></div>'
            if pct_sub is not None else ''
            f'</div>'
            f'{bias_warning}'
            f'<div style="font-size:0.78rem; color:#6b7280; margin-top:0.5rem;">{why_txt}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


#
# LAYOUT PRINCIPAL
#
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <h1 style="font-size:2.2rem; margin:0; letter-spacing:-0.02em;">
        StockPredict <span style="color:#7c6af7;">ML</span>
    </h1>
    <p style="color:#6b7280; font-size:0.95rem; margin-top:0.4rem;">
        30-day directional prediction - LSTM + XGBoost
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Mode selector
generalista_models = load_generalista_models()
models_available  = generalista_models is not None

if models_available:
    mode_opcions = [
        "Fast mode (pre-trained model, ~10 seconds)",
        "Full mode (trains per stock, ~3 minutes)",
    ]
    mode_sel = st.radio(
        "Analysis mode:",
        mode_opcions,
        horizontal=True,
        help=(
            "Fast mode: uses the pre-trained model with 20 US stocks. "
            "Only for US stocks. Very fast.\n\n"
            "Full mode: trains a new model specific to the stock. "
            "Works for any stock worldwide."
        ),
    )
    mode_rapid = mode_sel == mode_opcions[0]
else:
    st.info(
        "Fast mode not available. Run train_generalista.py to enable it. "
        "Using full mode."
    )
    mode_rapid = False

# Stacked LSTM toggle - only shown in full mode
if not mode_rapid:
    use_stacked = st.toggle(
        "Stacked LSTM (2 layers)",
        value=False,
        help=(
            "Standard: 1 LSTM layer — original architecture (faster).\n"
            "Stacked: 2 LSTM layers — captures more complex patterns "
            "(more regularization, slower). Use to compare architectures."
        ),
    )
else:
    use_stacked = False

st.divider()

# Form
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    ticker_input = st.text_input(
        "Ticker",
        placeholder="Ex: AAPL, TSLA, KO, ITX.MC, BMW.DE ..."
        if not mode_rapid else "Ex: AAPL, TSLA, MSFT, KO (stocks US)",
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
with col3:
    run_btn = st.button("Analyze", use_container_width=True)

if ticker_input:
    region, suffix = get_region(ticker_input)
    _, _, mkt_name = get_market_config(ticker_input)
    st.markdown(
        f"<div style='font-size:0.8rem; color:#6b7280; margin-top:-0.5rem;'>"
        f"Detected region: <b style='color:#7c6af7'>{region}</b> - "
        f"Market index: <b style='color:#7c6af7'>{mkt_name}</b>"
        f"</div>",
        unsafe_allow_html=True
    )

st.divider()

# Execution
if run_btn and ticker_input:
    ticker = ticker_input.strip().upper()

    progress_area = st.empty()
    status_area   = st.empty()
    results_area  = st.container()

    EXP_NAMES = {
        1: "OHLCV Baseline",
        2: "OHLCV + Technical",
        3: "OHLCV + Technical + Macro",
        4: "OHLCV + Technical + Macro + Market",
    }

    # MODE RAPID
    if mode_rapid:
        with status_area:
            st.info(f"Downloading data for **{ticker}**...")
        try:
            df = load_stock(ticker)
            if len(df) < 300:
                st.error(f"Not enough data found for **{ticker}**.")
                st.stop()
        except Exception as e:
            st.error(f"Error downloading {ticker}: {e}")
            st.stop()

        with status_area:
            st.info("Downloading US macro and market data...")
        # Use load_macro_generalist to ensure transformations are identical
        # to train_generalist.py (fed_rate_chg, cpi_yoy, vix_log, etc.)

        macro_gen     = load_macro_generalista(df.index)
        market_result = load_market(ticker)

        with status_area:
            st.info("Applying pre-trained model...")
        result_rapid = run_mode_rapid(
            ticker, df, macro_gen, market_result, generalista_models
        )
        status_area.empty()

        if result_rapid is None:
            st.error(
                "The generalist model could not process this ticker. "
                "Try the full mode or check that the ticker is US."
            )
            st.stop()

        # Sentiment
        sentiment_result = None
        try:
            from sentiment_module import get_sentiment_signal, _load_finbert
            sent_status = st.empty()
            sent_status.info("Analyzing recent news with FinBERT...")
            _load_finbert()
            best_signal = (result_rapid["lstm"]["signal"]
                           if result_rapid["lstm"]["signal"] != "INSUFFICIENT"
                           else result_rapid["xgb"]["signal"])
            sentiment_result = get_sentiment_signal(ticker, best_signal)
            sent_status.empty()
        except Exception:
            pass

        with results_area:
            today      = pd.Timestamp.today().strftime("%d %b %Y")
            n_sessions = len(df)

            st.markdown(f"""
            <div style="background:#13131a; border:1px solid #1e1e2e;
                        border-radius:12px; padding:1.2rem 1.5rem;
                        margin-bottom:1.5rem;">
                <div style="font-family:'Space Mono',monospace; font-size:1.4rem;
                            font-weight:700; color:#e2e8f0;">
                    {ticker}
                    <span style="font-size:0.75rem; color:#7c6af7;
                                 margin-left:0.8rem; background:#1e1e2e;
                                 padding:0.2rem 0.6rem; border-radius:4px;">
                        PRE-TRAINED MODEL
                    </span>
                    <span style="font-size:0.8rem; color:#6b7280; margin-left:1rem;">
                        {today} - Horizon: 30 days - {n_sessions} sessions
                    </span>
                </div>
                <div style="font-size:0.82rem; color:#6b7280; margin-top:0.4rem;">
                    Prediction based on model trained with 20 US stocks.
                    Not investment advice.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Generalist Model Prediction (Exp.4)")
            render_exp_result(4, "OHLCV + Technical + Macro + Market", result_rapid)

            # Consensus between both models
            s_lstm = result_rapid["lstm"]["signal"]
            s_xgb  = result_rapid["xgb"]["signal"]
            if s_lstm == s_xgb and s_lstm != "INSUFFICIENT":
                consens_color = "#22c55e" if s_lstm == "BUY" else "#ef4444"
                consens_txt   = f"CONSENSUS: {s_lstm}"
            elif s_lstm == s_xgb == "INSUFFICIENT":
                consens_color = "#6b7280"
                consens_txt   = "CONSENSUS: INSUFFICIENT"
            else:
                consens_color = "#eab308"
                consens_txt   = f"DIVERGENT - LSTM: {s_lstm} / XGBoost: {s_xgb}"

            st.markdown(
                f"<div style='background:#13131a; border:1px solid #1e1e2e; "
                f"border-radius:8px; padding:1rem 1.4rem; margin:1rem 0;'>"
                f"<span style='color:{consens_color}; font-weight:700; "
                f"font-size:1.1rem;'>{consens_txt}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Sentiment
            if sentiment_result:
                st.markdown("---")
                st.markdown("### Recent News Sentiment")
                label    = sentiment_result["sentiment_label"]
                score    = sentiment_result["sentiment_score"]
                strength = sentiment_result["strength"]
                adj      = sentiment_result["adjusted_signal"]
                n_news   = sentiment_result["news_count"]
                pos_r    = sentiment_result["positive_ratio"]
                neg_r    = sentiment_result["negative_ratio"]
                neu_r    = sentiment_result["neutral_ratio"]
                news     = sentiment_result["news"]
                sent_color = {"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#6b7280"}.get(label, "#6b7280")
                card_cls   = ("buy" if label == "POSITIVE" else "sell" if label == "NEGATIVE" else "none")
                st.markdown(f"""
                <div class="result-card {card_cls}">
                    <div class="exp-title">News Sentiment (FinBERT)</div>
                    <div class="header-bar">
                        <span style="color:{sent_color}; font-weight:700;
                                     font-size:1.1rem;">
                            SENTIMENT {label} ({strength})
                        </span>
                    </div>
                    <div style="font-size:0.88rem; margin:0.4rem 0 0.6rem 0;
                                color:#cbd5e1;">
                        Adjusted signal: <b>{adj}</b>
                    </div>
                    <div class="metric-row">
                        <div class="metric-item">News <span>{n_news}</span></div>
                        <div class="metric-item">Score <span>{score:+.3f}</span></div>
                        <div class="metric-item">Positive <span>{pos_r*100:.0f}%</span></div>
                        <div class="metric-item">Negative <span>{neg_r*100:.0f}%</span></div>
                        <div class="metric-item">Neutral <span>{neu_r*100:.0f}%</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if news:
                    with st.expander(f"View {len(news)} news", expanded=False):
                        for n in news:
                            icon = "[+]" if n["sentiment"] > 0.05 else "[-]" if n["sentiment"] < -0.05 else "[ ]"
                            st.markdown(
                                f"{icon} **[{n['sentiment']:+.2f}]** {n['title']}",
                                unsafe_allow_html=True
                            )

            st.markdown("""
            <div class="disclaimer">
                <b>Legal notice:</b> Forecast is indicative, based on historical
                 performance. Not investment advice.
            </div>
            """, unsafe_allow_html=True)

    # FULL MODE (original)
    else:
        all_results = {}
        total_steps = 4
        progress_bar = progress_area.progress(0)

        # Carregar data comunes
        with status_area:
            st.info(f"Downloading data for **{ticker}**...")
        try:
            df = load_stock(ticker)
            if len(df) < 300:
                st.error(f"Not enough data found for **{ticker}**. Verify the ticker.")
                st.stop()
        except Exception as e:
            st.error(f"Error downloading {ticker}: {e}")
            st.stop()

        with status_area:
            st.info(f"Downloading macroeconomic data ({get_region(ticker)[0]})...")
        macro_df = load_macro(ticker)

        with status_area:
            st.info("Downloading context de market...")
        market_result = load_market(ticker)
        market_df = market_result[0] if market_result else None

        # Run the 4 experiments.
        for exp_num in range(1, 5):
            with status_area:
                st.info(f"Training Exp.{exp_num} - {EXP_NAMES[exp_num]} (LSTM + XGBoost)...")
            try:
                result = run_experiment(ticker, exp_num, df, macro_df, market_df, stacked=use_stacked)
                all_results[exp_num] = result
            except Exception as e:
                all_results[exp_num] = None
                st.warning(f"Exp.{exp_num} ha fallat: {e}")
            progress_bar.progress(exp_num / total_steps)

        progress_area.empty()
        status_area.empty()

        # News sentiment
        sentiment_result = None
        try:
            from sentiment_module import get_sentiment_signal, _load_finbert
            sent_status = st.empty()
            sent_status.info("Analyzing recent news with FinBERT...")
            _load_finbert()
            best_signal = "INSUFFICIENT"
            for exp_num in [2, 1, 3, 4]:
                if exp_num in all_results and all_results[exp_num]:
                    for mk in ["lstm", "xgb"]:
                        sig = all_results[exp_num][mk]["signal"]
                        if sig != "INSUFFICIENT":
                            best_signal = sig
                            break
                    if best_signal != "INSUFFICIENT":
                        break
            sentiment_result = get_sentiment_signal(ticker, best_signal)
            sent_status.empty()
        except Exception:
            pass

        # Show results
        with results_area:
            today      = datetime.date.today().strftime("%d %b %Y")
            n_sessions = len(df)
            date_range = f"{df.index[0].date()} -> {df.index[-1].date()}"

            st.markdown(f"""
            <div style="background:#13131a; border:1px solid #1e1e2e; border-radius:12px;
                        padding:1.2rem 1.5rem; margin-bottom:1.5rem;">
                <div style="font-family:'Space Mono',monospace; font-size:1.4rem;
                            font-weight:700; color:#e2e8f0;">
                    {ticker}
                    <span style="font-size:0.8rem; color:#6b7280; margin-left:1rem;">
                        {today} - Horizon: 30 days - {n_sessions} sessions ({date_range})
                    </span>
                </div>
                <div style="font-size:0.82rem; color:#6b7280; margin-top:0.4rem;">
                    Predictions based on historical performance. Not investment advice.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Signal summary table.
            summary_rows = []
            for exp_num, result in all_results.items():
                if result is None:
                    continue
                for model_key, label in [("lstm", "LSTM"), ("xgb", "XGBoost")]:
                    r    = result[model_key]
                    m    = r["metrics"]
                    spec = r["specificity"]
                    summary_rows.append({
                        "Experiment"    : f"Exp.{exp_num} {EXP_NAMES[exp_num]}",
                        "Modelo"         : label,
                        "Signal"        : r["signal"],
                        "ROC-AUC"       : f"{m['roc_auc']:.3f}",
                        "Prec. Up" : f"{m['precision']*100:.1f}%",
                        "Prec. Lowdes": f"{spec*100:.1f}%",
                        "Reliability"    : conf_label(m["roc_auc"]),
                    })

            if summary_rows:
                st.markdown("### Signal Summary")
                df_summary = pd.DataFrame(summary_rows)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### Detail by Experiment")

            for exp_num, result in all_results.items():
                if result is None:
                    continue
                with st.expander(f"Exp.{exp_num} - {EXP_NAMES[exp_num]}",
                                 expanded=(exp_num <= 2)):
                    render_exp_result(exp_num, EXP_NAMES[exp_num], result)

            # Sentiment section
            if sentiment_result:
                label    = sentiment_result["sentiment_label"]
                strength = sentiment_result["strength"]
                score    = sentiment_result["sentiment_score"]
                adj      = sentiment_result["adjusted_signal"]
                n_news   = sentiment_result["news_count"]
                pos_r    = sentiment_result["positive_ratio"]
                neg_r    = sentiment_result["negative_ratio"]
                neu_r    = sentiment_result["neutral_ratio"]
                news     = sentiment_result["news"]

                st.markdown("---")
                st.markdown("### Recent News Sentiment")

                sent_color = {"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#6b7280"}.get(label, "#6b7280")
                card_class = ("buy" if label == "POSITIVE" else "sell" if label == "NEGATIVE" else "none")

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="exp-title">Exp.5 - News Sentiment (FinBERT)</div>
                    <div class="header-bar">
                        <span style="color:{sent_color}; font-weight:700; font-size:1.1rem;">
                            SENTIMENT {label} ({strength})
                        </span>
                    </div>
                    <div style="font-size:0.88rem; margin:0.4rem 0 0.6rem 0; color:#cbd5e1;">
                        Adjusted signal: <b>{adj}</b>
                    </div>
                    <div class="metric-row">
                        <div class="metric-item">News <span>{n_news}</span></div>
                        <div class="metric-item">Average score <span>{score:+.3f}</span></div>
                        <div class="metric-item">Positive <span>{pos_r*100:.0f}%</span></div>
                        <div class="metric-item">Negative <span>{neg_r*100:.0f}%</span></div>
                        <div class="metric-item">Neutral <span>{neu_r*100:.0f}%</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if news:
                    with st.expander(f"View {len(news)} news", expanded=False):
                        for n in news:
                            icon = ("[+]" if n["sentiment"] > 0
                                    else "[-]" if n["sentiment"] < 0 else "[ ]")
                            st.markdown(
                                f"{icon} **[{n['sentiment']:+.2f}]** {n['title']}  "
                                f"<span style='color:#6b7280; font-size:0.78rem;'>"
                                f"rel={n['relevance']:.2f} - t={n['time_weight']:.2f}"
                                f"</span>",
                                unsafe_allow_html=True
                            )

            st.markdown("""
            <div class="disclaimer">
                <b>Legal notice:</b> The predictions shown are based on the model’s 
                performance during the historical test period. Past performance does 
                not guarantee future results. This tool is an academic project and does 
                not constitute any investment advice.
                
                The user is solely responsible for any decisions made.
            </div>
            """, unsafe_allow_html=True)

elif run_btn and not ticker_input:
    st.warning("Enter a ticker to continue.")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color:#6b7280;">
        <div style="font-size:3rem; margin-bottom:1rem;">?</div>
        <div style="font-family:'Space Mono',monospace; font-size:1.1rem; color:#e2e8f0;">
            Enter a ticker and click <b>Analyze</b>
        </div>
        <div style="margin-top:0.8rem; font-size:0.88rem;">
            Compatible with any global stock: US, EU, UK, JP, and more
        </div>
        <div style="margin-top:2rem; display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;">
            <span style="background:#13131a; border:1px solid #1e1e2e; border-radius:8px;
                         padding:0.4rem 1rem; font-family:'Space Mono',monospace; font-size:0.82rem;">
                AAPL  -  TSLA  -  KO  -  MSFT
            </span>
            <span style="background:#13131a; border:1px solid #1e1e2e; border-radius:8px;
                         padding:0.4rem 1rem; font-family:'Space Mono',monospace; font-size:0.82rem;">
                ITX.MC  -  BMW.DE  -  SHEL.L
            </span>
            <span style="background:#13131a; border:1px solid #1e1e2e; border-radius:8px;
                         padding:0.4rem 1rem; font-family:'Space Mono',monospace; font-size:0.82rem;">
                7203.T  -  0700.HK  -  BHP.AX
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)