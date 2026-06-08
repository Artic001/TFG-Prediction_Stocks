"""
lstm_v1_baseline.py
======================
Experiment 1 - LSTM OHLCV baseline
Supports configurable prediction horizons (1d and 30d)

Execution:
    python lstm_v1_baseline.py --horizon 1
    python lstm_v1_baseline.py --horizon 30
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, balanced_accuracy_score,
)
from sklearn.isotonic import IsotonicRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

tf.get_logger().setLevel("ERROR")


# PARAMETERS BASE
TICKER      = ""
START_DATE  = "2010-07-01"
END_DATE    = "2026-03-01"
WINDOW_SIZE = 20
BATCH_SIZE  = 64
LR          = 1e-3
SEED        = 1

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Specific configuration per horizon
HORIZON_CONFIG = {
    1: {
        "epochs"   : 150,
        "units"    : 32,
        "dropout"  : 0.4,
        "l2"       : 1e-3,
        "patience" : 20,
    },
    30: {
        "epochs"   : 150,
        "units"    : 48,
        "dropout"  : 0.4,
        "l2"       : 2e-3,
        "patience" : 20,
    },
}


# DATA
def load_data():
    print(f"[DATA] Downloading {TICKER} ({START_DATE}  {END_DATE})...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    print(f"       {len(df)} sessions  "
          f"({df.index[0].date()}  {df.index[-1].date()})")
    return df


# FEATURES
def build_features(df):
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
    return out.replace([np.inf, -np.inf], np.nan)


def build_target(df, horizon: int):
    """
    Target binari:
      horizon=1 : 1 si close_{t+1} > close_t
      horizon=30: 1 si mean(close[t+1:t+30]) > mean(close[t-29:t])
                  It smooths out noise from specific days
                  and captures genuine medium-term trends.
    """
    C = df["Close"]
    if horizon == 1:
        future_ret = C.pct_change(1).shift(-1)
        target = (future_ret > 0).astype(int)
    else:
        # Average of the next 30 days vs. average of the previous 30 days.
        future_mean = C.shift(-horizon).rolling(horizon).mean().shift(-(horizon - 1))
        past_mean   = C.rolling(horizon).mean()
        target = (future_mean > past_mean).astype(int)
    return target.rename("target")


# SPLIT + SCALING + SEQUENCES
def prepare_data(df, horizon: int):
    features = build_features(df)
    target   = build_target(df, horizon)

    idx      = features.index.intersection(target.index)
    features = features.loc[idx].dropna()
    target   = target.loc[features.index]

    # Remove the last 'horizon' rows (they do not have a valid target).
    features = features.iloc[:-horizon]
    target   = target.iloc[:-horizon]

    X_raw = features.values
    y_raw = target.values
    dates = features.index

    n_train = int(len(X_raw) * 0.70)
    n_val   = int(len(X_raw) * 0.15)

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

    print(f"\n[PREP] Horizon       : {horizon} days")
    print(f"       Features       : {X_raw.shape[1]}")
    print(f"       Train          : {X_train.shape}")
    print(f"       Val            : {X_val.shape}")
    print(f"       Test           : {X_test.shape}")
    print(f"       % upward test     : {y_test.mean():.2%}")
    print(f"       Test dates        : {dates_test[0].date()}  "
          f"{dates_test[-1].date()}")

    # Last 20 days of the entire dataset - real prediction today
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
            Dense(8, activation="relu",
                  kernel_regularizer=regularizers.l2(cfg["l2"])),
            Dropout(cfg["dropout"] * 0.5),
            Dense(1, activation="sigmoid"),
        ]
        model = Sequential(layers, name="LSTM_OHLCV_Stacked")
    else:
        layers = [
            LSTM(cfg["units"], input_shape=(WINDOW_SIZE, n_features),
                 kernel_regularizer=regularizers.l2(cfg["l2"]),
                 recurrent_regularizer=regularizers.l2(cfg["l2"]),
                 dropout=cfg["dropout"], recurrent_dropout=0.2),
            Dropout(cfg["dropout"]),
            Dense(8, activation="relu",
                  kernel_regularizer=regularizers.l2(cfg["l2"])),
            Dense(1, activation="sigmoid"),
        ]
        model = Sequential(layers, name="LSTM_OHLCV_Standard")

    model.compile(
        optimizer = Adam(learning_rate=LR, clipnorm=1.0),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy", tf.keras.metrics.AUC(name="auc")],
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
        fixed_epochs = 150
        monitor_str  = "val_auc (max)"
    else:
        cbs = [
            EarlyStopping(monitor="val_auc", mode="max",
                          patience=cfg["patience"],
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_auc", mode="max",
                              factor=0.5, patience=10,
                              min_lr=1e-6, verbose=1),
        ]
        fixed_epochs = cfg["epochs"]
        monitor_str  = "val_auc (max)"

    print(f"\n[TRAIN] Horizono={horizon}d | units={cfg['units']} | "
          f"dropout={cfg['dropout']} | l2={cfg['l2']} | "
          f"monitor={monitor_str}")

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data = (data["X_val"], data["y_val"]),
        epochs      = fixed_epochs,
        batch_size  = BATCH_SIZE,
        callbacks   = cbs,
        verbose     = 1,
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
    print(f"            ptim: t={best_t:.2f}  BalAcc={best_ba:.4f}")
    print("            Top 5:")
    for t, ba, f1 in rows[:5]:
        print(f"              t={t:.2f}  BalAcc={ba:.4f}  F1={f1:.4f}")
    return best_t


# EVALUATION
def evaluate(model, data, threshold, horizon: int):
    y_prob = model.predict(data["X_test"], verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    y_test = data["y_test"]

    # Real prediction for today minus the last 20 days from the entire dataset.
    prob_today = float(model.predict(data["X_today"], verbose=0).flatten()[0])
    pred_today = "UP" if prob_today >= threshold else "DOWN"

    metrics = {
        "experiment"       : "exp1_ohlcv",
        "horizon_days"     : horizon,
        "accuracy"         : round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "f1"               : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc"          : round(roc_auc_score(y_test, y_prob), 4),
        "precision"        : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"           : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "threshold"        : round(threshold, 2),
        "n_test"           : int(len(y_test)),
    }
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    metrics["specificity"]    = specificity
    metrics["prob_today"]     = round(prob_today, 4)
    metrics["pred_today"]     = pred_today
    metrics["pct_upward_test"] = round(float(np.mean(y_test)), 3)

    label = f"Horizon {horizon}d"
    print(f"\n{'='*54}")
    print(f"  RESULTS - Exp.1 OHLCV | {label} | {TICKER}")
    print(f"{'='*54}")
    for k, v in metrics.items():
        if k not in ("experiment", "n_test"):
            print(f"  {k:<22}: {v}")
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=["Downward", "Upward"]))
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:>4}  FP={cm[0,1]:>4}")
    print(f"    FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")
    print(f"  prob_today            : {prob_today:.4f}")
    print(f"  pred_today            : {pred_today}")
    print(f"  pct_upward_test       : {float(np.mean(y_test)):.2%}")



    return y_prob, metrics


# SIGNAL FOR THE INVESTOR
def investment_signal(metrics: dict, horizon: int):
    """
    Generate a clear signal for the investor based on the model’s real performance on the test set.

    Logic:
        AUC < 0.55 -> INSUFFICIENT SIGNAL (model too close to random)
        Model predicts UP + upward precision > 55% -> BUY
        Model predicts DOWN + downward precision > 55% -> DO NOT BUY / SELL
        Otherwise -> INSUFFICIENT SIGNAL
    """
    import datetime

    auc       = metrics["roc_auc"]
    precision = metrics["precision"]
    recall    = metrics["recall"]
    acc       = metrics["accuracy"]
    ba        = metrics["balanced_accuracy"]
    hor_txt   = "day" if horizon == 1 else f"{horizon} days"
    today     = datetime.date.today().strftime("%d %b %Y")

    prec_up     = precision
    specificity = metrics["specificity"]   # TN / (TN + FP), calculated from the confusion matrix

    pred_today = metrics.get("pred_today", None)

    # Determine the signal
    if auc < 0.55:
        signal     = "INSUFFICIENT"
        signal_col = "INSUFFICIENT SIGNAL"
        action     = "The model does not have enough reliability to recommend any action."
        reason     = f"ROC-AUC is {auc:.3f}, too close to a random model (0.50)."
    elif pred_today == "UP" and prec_up > 0.55 and specificity > 0.50:
        signal     = "BUY"
        signal_col = "SIGNAL: BUY"
        action     = (f"The model recommends buying {TICKER} and holding\n"
                      f"  during the next {hor_txt}.")
        reason     = (f"In situations similar to the current one, the model has\n"
                      f"  correctly predicted the UPWARD direction {prec_up*100:.1f}%\n"
                      f"  of the times in the evaluated period.")
    elif pred_today == "DOWN" and specificity > 0.55:
        signal     = "DO NOT BUY"
        signal_col = "SIGNAL: DO NOT BUY / SELL"
        action     = (f"The model recommends not buying {TICKER} now.\n"
                      f"  If you have open positions, consider closing them.")
        reason     = (f"In situations similar to the current one, the model has\n"
                      f"  correctly predicted the DOWNWARD direction {specificity*100:.1f}%\n"
                      f"  of the times in the evaluated period.")
    else:
        signal     = "INSUFFICIENT"
        signal_col = "INSUFFICIENT SIGNAL"
        action     = "The model does not generate a clear enough signal to act."
        reason     = "Reliability in both directions is similar or low."

    # Confidence level based on AUC
    if auc >= 0.70:
        conf_txt = "Moderate-High"
    elif auc >= 0.60:
        conf_txt = "Moderate"
    else:
        conf_txt = "Low"

    W = 54
    print(f"\n{''*W}")
    print(f"  {TICKER} - {signal_col}")
    print(f"  {today}  |  Horizon: {hor_txt}")
    print(f"{''*W}")
    print(f"  {action}")
    print(f"{''*W}")
    print(f"  Why?")
    print(f"  {reason}")
    print(f"{''*W}")
    print(f"  Historical model performance (test set):")
    print(f"    Precision (up)  : {prec_up*100:.1f}%")
    print(f"    Precision (down) : {specificity*100:.1f}%")
    print(f"    Global accuracy      : {acc*100:.1f}%")
    print(f"    ROC-AUC              : {auc:.3f}")
    print(f"    Reliability           : {conf_txt}")
    print(f"{''*W}")
    print(f"    This prediction is indicative and does not")
    print(f"     constitute any investment advice.")
    print(f"{''*W}\n")

    return signal


# CHARTS
def plot_all(history, data, y_prob, threshold, horizon: int, metrics: dict):
    y_test  = data["y_test"]
    y_pred  = (y_prob >= threshold).astype(int)
    dates   = pd.to_datetime(data["dates_test"])
    correct = y_pred == y_test
    prefix  = f"exp1_{horizon}d"

    #  Fig 1: training history
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(
        f"Exp.1 OHLCV | Horizon {horizon}d | Training history",
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
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    #  Fig 2: predictions vs. actual
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"Exp.1 OHLCV | Horizon {horizon}d | "
        f"AUC={metrics['roc_auc']:.4f}  Acc={metrics['accuracy']:.4f}",
        fontweight="bold", fontsize=13)

    axes[0].plot(dates, y_prob, color="#6366F1", lw=0.9, alpha=0.9,
                 label="P(upward)")
    axes[0].axhline(threshold, color="#F59E0B", ls="--", lw=1.3,
                    label=f"Threshold ({threshold:.2f})")
    axes[0].fill_between(dates, 0, y_test * 0.1,
                         alpha=0.25, color="#10B981", label="Actual target")
    axes[0].set_ylabel("Probability"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)

    axes[1].scatter(dates[correct],  np.ones(correct.sum()),
                    color="#10B981", s=7, alpha=0.5, label="Correct")
    axes[1].scatter(dates[~correct], np.zeros((~correct).sum()),
                    color="#EF4444", s=7, alpha=0.5, label="Error")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Incorrect", "Correct"])
    axes[1].set_xlabel("Date"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

    #  Fig 3: curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#6366F1", lw=2,
            label=f"LSTM OHLCV {horizon}d  (AUC={metrics['roc_auc']:.4f})")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.fill_between(fpr, tpr, alpha=0.07, color="#6366F1")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - Exp.1 OHLCV | {horizon}d | {TICKER}",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1,
                        choices=[1, 5, 10, 30],
                        help="Prediction horizon in days (default: 1).")
    parser.add_argument("--stacked", action="store_true", default=False,
                        help="Use Stacked LSTM (2 layers) instead of standard (1 layer)")
    parser.add_argument("--ticker", type=str, required=True,
                        help="Stock ticker (ex: TSLA, AAPL, ITX.MC)")
    args    = parser.parse_args()
    horizon = args.horizon
    TICKER  = args.ticker.upper()

    print(f"\n{'='*54}")
    print(f"  Exp.1 BASELINE - LSTM OHLCV | {TICKER} | {horizon}d")
    print(f"{'='*54}")

    df        = load_data()
    data      = prepare_data(df, horizon)
    model     = build_model(data["n_features"], horizon, stacked=args.stacked)
    model.summary()
    history   = train_model(model, data, horizon)
    threshold = find_best_threshold(model, data)
    y_prob, metrics = evaluate(model, data, threshold, horizon)
    investment_signal(metrics, horizon)
    plot_all(history, data, y_prob, threshold, horizon, metrics)

    print(f"\n[DONE] Experiment 1 OHLCV | {horizon}d completed.")