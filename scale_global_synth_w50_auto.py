# src/scale_global_synth_w50_auto.py
import os, json, warnings
import numpy as np
from joblib import load as joblib_load
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE = "."
SYN_DIR = os.path.join(BASE, "data", "synth", "GLOBAL")
REAL_DIRS = {
    "FD001": os.path.join(BASE, "data", "processed", "FD001"),
    "FD002": os.path.join(BASE, "data", "processed", "FD002"),
    "FD003": os.path.join(BASE, "data", "processed", "FD003"),
    "FD004": os.path.join(BASE, "data", "processed", "FD004"),
}

def feat_stats(arr):  # arr: [*,24]
    m = arr.mean(axis=0)
    s = arr.std(axis=0) + 1e-9
    return float(np.mean(np.abs(m))), float(np.mean(s))

# ---- load synth (unscaled) ----
X = np.load(os.path.join(SYN_DIR, "X_train_w50.npy"))       # [N,50,28] (first 24 features; last 4 = one-hot)
y = np.load(os.path.join(SYN_DIR, "y_train_w50.npy"))       # [N]
fd = np.load(os.path.join(SYN_DIR, "fd_train_idx.npy"))     # [N] ints like {0,1,2,3}
assert X.shape[-1] == 28, "Expected 24 features + 4 one-hot (total 28)"
N, W, C = X.shape
print(f"[READ] X {X.shape} y {y.shape} fd {fd.shape}")

# ---- load scalers ----
scalers = {name: joblib_load(os.path.join(path, "scaler.joblib")) for name, path in REAL_DIRS.items()}

# ---- infer mapping: domain-code -> FD name ----
domain_codes = sorted(np.unique(fd).tolist())
mapping = {}

print("[MAP] inferring fd-code -> FD name by trying each scaler per domain code ...")
for code in domain_codes:
    idx = np.where(fd == code)[0]
    # take a small sample for speed
    sample_idx = idx if len(idx) <= 500 else np.random.choice(idx, size=500, replace=False)
    sample = X[sample_idx, :, :24].reshape(-1, 24)  # [n*50,24]
    best = None
    for fd_name, scaler in scalers.items():
        xs = scaler.transform(sample)
        meanAbs, stdAvg = feat_stats(xs)
        score = abs(meanAbs - 0.0) + abs(stdAvg - 1.0)  # how close to 0/1
        if (best is None) or (score < best[0]):
            best = (score, fd_name, meanAbs, stdAvg)
    score, fd_name, mA, sA = best
    mapping[code] = fd_name
    print(f"  code {code} -> {fd_name}  (score={score:.3f}, meanAbs={mA:.3f}, stdAvg={sA:.3f}, rows={len(idx)})")

# persist mapping
map_path = os.path.join(SYN_DIR, "fd_code_to_name.json")
with open(map_path, "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)
print(f"[WRITE] mapping saved -> {map_path}")

# ---- rescale per domain using inferred mapping ----
Xs = X.copy()
for code in domain_codes:
    fd_name = mapping[code]
    scaler = scalers[fd_name]
    idx = np.where(fd == code)[0]
    block = X[idx, :, :24].reshape(-1, 24)
    block_s = scaler.transform(block)
    Xs[idx, :, :24] = block_s.reshape(len(idx), W, 24)

# ---- per-domain post-scale sanity ----
print("[CHECK] per-domain post-scale stats (want meanAbs≈0, stdAvg≈1)")
for code in domain_codes:
    idx = np.where(fd == code)[0]
    feats = Xs[idx, :, :24].reshape(-1, 24)
    meanAbs, stdAvg = feat_stats(feats)
    print(f"  code {code} ({mapping[code]}): meanAbs={meanAbs:.3f}  stdAvg={stdAvg:.3f}  (rows={len(idx)})")

# ---- save scaled X ----
outX = os.path.join(SYN_DIR, "X_train_w50_scaled.npy")
np.save(outX, Xs)
print(f"[WRITE] {outX}  shape={Xs.shape} (first 24 features scaled per inferred FD; one-hot unchanged)")
