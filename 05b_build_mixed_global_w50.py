# src/05b_build_mixed_global_w50.py
import os, numpy as np, json

BASE = "."
REAL = os.path.join(BASE, "data", "processed", "GLOBAL")
SYN  = os.path.join(BASE, "data", "synth", "GLOBAL")
OUT  = os.path.join(BASE, "data", "mix", "GLOBAL_w50")
os.makedirs(OUT, exist_ok=True)

# --- read real (already contains 24 features + 4 one-hot) ---
Xr = np.load(os.path.join(REAL, "X_train_w50.npy"))     # [Nr, 50, 28]
yr = np.load(os.path.join(REAL, "y_train_w50.npy"))     # [Nr]
# optional: per-row domain indices for real (if present)
fd_real_path = os.path.join(REAL, "fd_train_idx.npy")
fd_r = np.load(fd_real_path) if os.path.exists(fd_real_path) else None

# --- read synthesized (already scaled & has one-hot; from your auto-scaler script) ---
Xs = np.load(os.path.join(SYN,  "X_train_w50_scaled.npy"))  # [Ns, 50, 28]
ys = np.load(os.path.join(SYN,  "y_train_w50.npy"))         # [Ns]
fds = np.load(os.path.join(SYN,  "fd_train_idx.npy"))       # [Ns]

assert Xr.shape[1:] == Xs.shape[1:] == (50, 28), "Expect 50x28 windows (24 feat + 4 one-hot)."

# --- concat ---
X = np.concatenate([Xr, Xs], axis=0)
y = np.concatenate([yr, ys], axis=0)

# if we have domain labels for real, keep them; else fill with -1 for real part
if fd_r is None:
    fd_r = -1 * np.ones((len(yr),), dtype=np.int64)
fd = np.concatenate([fd_r, fds], axis=0)

# --- save ---
np.save(os.path.join(OUT, "X_train_w50.npy"), X)
np.save(os.path.join(OUT, "y_train_w50.npy"), y)
np.save(os.path.join(OUT, "fd_train_idx.npy"), fd)

print(f"[MIX] GLOBAL-50 mixed built -> {OUT}")
print(f"      X {X.shape} y {y.shape} fd {fd.shape} (C={X.shape[-1]})")
