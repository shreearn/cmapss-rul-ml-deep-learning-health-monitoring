# src/prep_global_synth_w50.py
# Build a GLOBAL (window=50) synthetic training set with 4-way one-hot FD channels.
# Input:  data/synth/FD00{1..4}/*X_train_w50.npy and *y_train_w50.npy  (shapes [N,50,24] and [N])
# Output: data/synth/GLOBAL/X_train_w50.npy  (shape [N_total,50,28]),
#         data/synth/GLOBAL/y_train_w50.npy  (shape [N_total]),
#         data/synth/GLOBAL/fd_train_idx.npy (shape [N_total], values 0..3)

import os, glob, numpy as np

BASE = "."
WIN = 50
FDS = ["FD001","FD002","FD003","FD004"]

IN_DIR  = os.path.join(BASE, "data", "synth")
OUT_DIR = os.path.join(IN_DIR, "GLOBAL")
os.makedirs(OUT_DIR, exist_ok=True)

def one_hot_block(dom_idx: int, W: int) -> np.ndarray:
    a = np.zeros((W, 4), dtype=np.float32)
    a[:, dom_idx] = 1.0
    return a  # [W,4]

X_all, y_all, fd_idx = [], [], []

for i, fd in enumerate(FDS):
    d = os.path.join(IN_DIR, fd)

    fx = glob.glob(os.path.join(d, "*X_train_w50.npy"))
    fy = glob.glob(os.path.join(d, "*y_train_w50.npy"))
    if not fx or not fy:
        raise FileNotFoundError(f"Missing files in {d}. Need *X_train_w50.npy and *y_train_w50.npy")

    X = np.load(fx[0])      # [N, 50, C]
    y = np.load(fy[0])      # [N]

    if X.ndim != 3 or X.shape[1] != WIN:
        raise ValueError(f"{fd}: expected [N,{WIN},C], got {X.shape}")

    if X.shape[2] == 24:
        # add 4-d one-hot across time
        oh = one_hot_block(i, WIN)                       # [W,4]
        oh = np.repeat(oh[None, ...], X.shape[0], axis=0)  # [N,W,4]
        Xg = np.concatenate([X.astype(np.float32), oh], axis=2)  # [N,W,28]
    elif X.shape[2] == 28:
        # already has one-hot channels—keep as-is
        Xg = X.astype(np.float32)
    else:
        raise ValueError(f"{fd}: last dim must be 24 or 28, got {X.shape[2]}")

    X_all.append(Xg)
    y_all.append(y.astype(np.float32))
    fd_idx.extend([i]*X.shape[0])

X_all = np.concatenate(X_all, axis=0)  # [N_total,50,28]
y_all = np.concatenate(y_all, axis=0)  # [N_total]
fd_idx = np.asarray(fd_idx, dtype=np.int64)

np.save(os.path.join(OUT_DIR, f"X_train_w{WIN}.npy"), X_all)
np.save(os.path.join(OUT_DIR, f"y_train_w{WIN}.npy"), y_all)
np.save(os.path.join(OUT_DIR, "fd_train_idx.npy"), fd_idx)

print("[OK] GLOBAL synth built:",
      "X", X_all.shape, "y", y_all.shape, "fd_idx", fd_idx.shape, "| C=", X_all.shape[-1])
