# src/05_prepare_global.py
import os, argparse
import numpy as np

FDS = ["FD001","FD002","FD003","FD004"]
BASE = "."
OUT_DIR = os.path.join(BASE, "data", "processed", "GLOBAL")
os.makedirs(OUT_DIR, exist_ok=True)

def one_hot_domain(fd_idx: int, w: int) -> np.ndarray:
    """[W,4] one-hot for domain fd_idx (0..3)."""
    dom = np.zeros((w, 4), dtype=np.float32)
    dom[:, fd_idx] = 1.0
    return dom

def load_fd(fd: str, w: int, fd_idx: int):
    """Load FD {fd} X/y with window w and append domain one-hot -> [N,W,28]."""
    pfx = os.path.join(BASE, "data", "processed", fd)
    xtr = np.load(os.path.join(pfx, f"X_train_w{w}.npy"))   # [N,W,24]
    ytr = np.load(os.path.join(pfx, f"y_train_w{w}.npy"))   # [N]
    xte = np.load(os.path.join(pfx, f"X_test_w{w}.npy"))    # [M,W,24]
    yte = np.load(os.path.join(pfx, f"y_test_w{w}.npy"))    # [M]

    # append domain channels
    dom_tr = one_hot_domain(fd_idx, w)                      # [W,4]
    dom_te = dom_tr                                         # same shape
    xtr = np.concatenate([xtr, np.broadcast_to(dom_tr, (xtr.shape[0], w, 4))], axis=2)  # -> [N,W,28]
    xte = np.concatenate([xte, np.broadcast_to(dom_te, (xte.shape[0], w, 4))], axis=2)  # -> [M,W,28]

    return xtr, ytr, xte, yte

def main(w: int):
    xs_tr, ys_tr, xs_te, ys_te = [], [], [], []
    fd_tr_idx, fd_te_idx = [], []

    for i, fd in enumerate(FDS):
        # sanity: ensure per-FD window files exist
        pfx = os.path.join(BASE, "data", "processed", fd)
        need = [f"X_train_w{w}.npy", f"y_train_w{w}.npy", f"X_test_w{w}.npy", f"y_test_w{w}.npy"]
        for f in need:
            path = os.path.join(pfx, f)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}. Prepare this FD with window={w} first.")

        xtr, ytr, xte, yte = load_fd(fd, w, i)
        xs_tr.append(xtr); ys_tr.append(ytr)
        xs_te.append(xte); ys_te.append(yte)
        fd_tr_idx.append(np.full(len(ytr), i, dtype=np.int64))
        fd_te_idx.append(np.full(len(yte), i, dtype=np.int64))

        print(f"[{fd}] train {xtr.shape}, test {xte.shape}")

    X_train = np.concatenate(xs_tr, axis=0)   # [N_all, W, 28]
    y_train = np.concatenate(ys_tr, axis=0)   # [N_all]
    X_test  = np.concatenate(xs_te, axis=0)   # [M_all, W, 28]
    y_test  = np.concatenate(ys_te, axis=0)   # [M_all]
    fd_train = np.concatenate(fd_tr_idx, axis=0)  # [N_all]
    fd_test  = np.concatenate(fd_te_idx, axis=0)  # [M_all]

    np.save(os.path.join(OUT_DIR, f"X_train_w{w}.npy"), X_train)
    np.save(os.path.join(OUT_DIR, f"y_train_w{w}.npy"), y_train)
    np.save(os.path.join(OUT_DIR, f"X_test_w{w}.npy"),  X_test)
    np.save(os.path.join(OUT_DIR, f"y_test_w{w}.npy"),  y_test)
    np.save(os.path.join(OUT_DIR, f"fd_train_idx.npy"), fd_train)
    np.save(os.path.join(OUT_DIR, f"fd_test_idx.npy"),  fd_test)

    print(f"[SAVE] GLOBAL: {X_train.shape} {y_train.shape} {X_test.shape} {y_test.shape}  (features with domain)= {X_train.shape[2]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30, choices=[30,50], help="rolling window length")
    args = ap.parse_args()
    main(args.window)
