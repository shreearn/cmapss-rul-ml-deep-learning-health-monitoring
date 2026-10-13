# src/01_prepare_cmapss.py
# Usage (from project root):  python src/01_prepare_cmapss.py
# Optional args:             python src/01_prepare_cmapss.py --fd FD001 --window 30 --cap 125

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

# -----------------------------
# Helpers
# -----------------------------
def build_cols():
    """Return canonical CMAPSS column names (26 columns)."""
    return (["unit", "cycle", "setting_1", "setting_2", "setting_3"] +
            [f"sensor_{i}" for i in range(1, 22)])

def read_split(raw_dir: str, fd: str, split: str, cols: list) -> pd.DataFrame:
    """Read train/test split with whitespace separator and no header."""
    fn = f"{split}_{fd}.txt"
    path = os.path.join(raw_dir, fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    if df.shape[1] != len(cols):
        raise ValueError(
            f"{fn}: expected {len(cols)} columns, got {df.shape[1]} (check separators/encoding)"
        )
    return df

def load_truth_rul(raw_dir: str, fd: str) -> pd.Series:
    """Load RUL truth file for test engines (one value per engine, same order as in test file)."""
    path = os.path.join(raw_dir, f"RUL_{fd}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    truth = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"]).squeeze("columns")
    return truth

def add_train_rul(train: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Add RUL columns to TRAIN (exact RUL and optionally capped)."""
    train = train.copy()
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = (max_cycle - train["cycle"]).astype(int)
    train["RUL_capped"] = train["RUL"] if cap is None else train["RUL"].clip(upper=cap)
    return train

def add_test_rul(test: pd.DataFrame, truth_rul: pd.Series, cap: int | None = None) -> pd.DataFrame:
    """
    Add RUL to TEST using truth for the last row of each engine.
    For each unit u: failure_cycle = last_observed_cycle(u) + truth_rul(u).
    Then RUL(row) = failure_cycle - cycle(row).
    """
    test = test.copy()
    # order of truth matches the order units appear in test
    units_order = test["unit"].drop_duplicates().tolist()
    if len(units_order) != len(truth_rul):
        raise ValueError(
            "Mismatch: number of unique test engines vs rows in RUL truth file."
        )
    unit2truth = {u: int(truth_rul.iloc[i]) for i, u in enumerate(units_order)}

    # last observed cycle per unit in *test*
    last_cycle_map = test.groupby("unit")["cycle"].max().to_dict()

    # compute failure_cycle per row, then RUL
    failure_cycle = test["unit"].map(lambda u: last_cycle_map[u] + unit2truth[u])
    test["RUL"] = (failure_cycle - test["cycle"]).astype(int)
    test["RUL_capped"] = test["RUL"] if cap is None else test["RUL"].clip(upper=cap)
    return test

def fit_transform_scaler(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list, out_dir: str):
    """Fit scaler on TRAIN features only; transform both splits; save scaler."""
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[feature_cols] = scaler.transform(train[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])

    dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    return train_scaled, test_scaled

def make_windows(df: pd.DataFrame, feature_cols: list, label_col: str, window: int):
    """
    Create sliding windows per unit:
      X shape -> [num_windows, window, num_features]
      y shape -> [num_windows]
    Label is the RUL at the last timestep of the window.
    """
    X, y = [], []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        vals = g[feature_cols].values
        labels = g[label_col].values
        if len(g) < window:
            continue
        for t in range(window, len(g) + 1):
            X.append(vals[t - window:t])
            y.append(labels[t - 1])
    if not X:
        return np.empty((0, window, len(feature_cols))), np.empty((0,))
    return np.stack(X), np.array(y, dtype=np.float32)

# -----------------------------
# Main
# -----------------------------
def main(fd: str, base: str = ".", window: int = 30, cap: int | None = 125, save_windows: bool = True):
    cols = build_cols()

    raw_dir = os.path.join(base, "data", "raw", fd)
    out_dir = os.path.join(base, "data", "processed", fd)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Using dataset: {fd}")
    print(f"[INFO] RAW dir: {raw_dir}")
    print(f"[INFO] OUT dir: {out_dir}")

    # 1) Load splits
    train = read_split(raw_dir, fd, "train", cols)
    test = read_split(raw_dir, fd, "test", cols)

    print(f"[INFO] Shapes (rows, cols): train={train.shape}, test={test.shape}")
    assert train.shape[1] == 26 and test.shape[1] == 26, "Expected 26 columns."

    # 2) Add RUL labels
    truth_rul = load_truth_rul(raw_dir, fd)
    train = add_train_rul(train, cap=cap)
    test = add_test_rul(test, truth_rul, cap=cap)

    # Quick sanity checks
    assert train["RUL"].min() >= 0 and test["RUL"].min() >= 0, "Negative RUL found."
    print(f"[INFO] RUL stats (train): min={train['RUL'].min()}, max={train['RUL'].max()}")
    print(f"[INFO] RUL stats (test) : min={test['RUL'].min()}, max={test['RUL'].max()}")

    # 3) Feature scaling (fit on TRAIN only)
    feature_cols = ["setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    label_col = "RUL_capped" if cap is not None else "RUL"

    train_s, test_s = fit_transform_scaler(train, test, feature_cols, out_dir)

    # 4) Save row-wise (tabular) versions
    train_s.to_parquet(os.path.join(out_dir, "train.parquet"))
    test_s.to_parquet(os.path.join(out_dir, "test.parquet"))
    print(f"[SAVE] {os.path.join(out_dir, 'train.parquet')}")
    print(f"[SAVE] {os.path.join(out_dir, 'test.parquet')}")

    # 5) Optional: build & save sequence windows for deep models
    if save_windows and window is not None and window > 1:
        X_train, y_train = make_windows(train_s, feature_cols, label_col, window)
        X_test, y_test = make_windows(test_s, feature_cols, label_col, window)

        np.save(os.path.join(out_dir, f"X_train_w{window}.npy"), X_train)
        np.save(os.path.join(out_dir, f"y_train_w{window}.npy"), y_train)
        np.save(os.path.join(out_dir, f"X_test_w{window}.npy"), X_test)
        np.save(os.path.join(out_dir, f"y_test_w{window}.npy"), y_test)

        print(f"[SAVE] windows: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"[SAVE] windows: X_test  {X_test.shape}, y_test  {y_test.shape}")

    # 6) Save a tiny metadata file for reproducibility
    meta = {
        "fd": fd,
        "window": window,
        "cap": cap,
        "feature_cols": feature_cols,
        "label_col": label_col,
        "rows": {"train": int(len(train)), "test": int(len(test))},
    }
    with open(os.path.join(out_dir, "prep_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVE] {os.path.join(out_dir, 'prep_meta.json')}")

    print("[DONE] Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, default="FD001", help="FD001 | FD002 | FD003 | FD004")
    parser.add_argument("--base", type=str, default=".", help="Project root (use '.' if running from root)")
    parser.add_argument("--window", type=int, default=30, help="Sequence window length for deep models")
    parser.add_argument("--cap", type=int, default=125, help="Cap RUL at this value; set -1 to disable")
    parser.add_argument("--no-windows", action="store_true", help="Do not create sequence windows")
    args = parser.parse_args()

    cap_val = None if (args.cap is not None and args.cap < 0) else args.cap
    save_windows_flag = not args.no_windows
    main(fd=args.fd, base=args.base, window=args.window, cap=cap_val, save_windows=save_windows_flag)
