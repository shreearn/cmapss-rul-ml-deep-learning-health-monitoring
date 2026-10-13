# src/06_train_global.py
# Train the GLOBAL CNN-LSTM-CBAM model for a chosen window (30 or 50).
# Examples:
#   python src/06_train_global.py --window 30 --cnn 96 --lstm 96
#   python src/06_train_global.py --window 50 --cnn 96 --lstm 96

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from math import sqrt
from sklearn.metrics import mean_squared_error

# ------------------------- utils -------------------------
def nasa_score(y_true, y_pred):
    s = 0.0
    for yt, yp in zip(y_true, y_pred):
        d = float(yp) - float(yt)
        s += (np.exp(-d/13) - 1) if d < 0 else (np.exp(d/10) - 1)
    return float(s)

def to_tensor(x: np.ndarray) -> torch.Tensor:
    # x: [B, W, C] -> [B, C, W]
    return torch.tensor(x, dtype=torch.float32).permute(0, 2, 1)

# ------------------------- model -------------------------
class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, in_channels, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):  # [B,C,T]
        avg_pool = torch.mean(x, dim=2)
        max_pool, _ = torch.max(x, dim=2)
        a = self.mlp(avg_pool) + self.mlp(max_pool)
        return x * self.sig(a).unsqueeze(-1).expand_as(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=k, padding=pad, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):  # [B,C,T]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        return x * self.sig(self.conv(a)).expand_as(x)

class CBAM1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention1D(ch)
        self.sa = SpatialAttention1D()

    def forward(self, x):
        return self.sa(self.ca(x))

class CNNLSTM_CBAM(nn.Module):
    def __init__(self, in_ch, cnn_ch=96, lstm_h=96, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, cnn_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_ch, cnn_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cbam = CBAM1D(cnn_ch)
        self.lstm = nn.LSTM(input_size=cnn_ch, hidden_size=lstm_h, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_h, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):  # x: [B,C,T]
        z = self.cbam(self.conv(x))
        z = z.permute(0, 2, 1)  # [B,T,C]
        _, (h, _) = self.lstm(z)
        return self.head(h[-1])

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30, choices=[30, 50])
    ap.add_argument("--cnn", type=int, default=96)
    ap.add_argument("--lstm", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val_batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    BASE = "."
    GLOBAL_DIR = os.path.join(BASE, "data", "processed", "GLOBAL")
    MODELS = os.path.join(BASE, "data", "models")
    REPORTS = os.path.join(BASE, "data", "reports")
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)

    W = args.window
    # required files
    xtr_p = os.path.join(GLOBAL_DIR, f"X_train_w{W}.npy")
    ytr_p = os.path.join(GLOBAL_DIR, f"y_train_w{W}.npy")
    xte_p = os.path.join(GLOBAL_DIR, f"X_test_w{W}.npy")
    yte_p = os.path.join(GLOBAL_DIR, f"y_test_w{W}.npy")
    fdi_p = os.path.join(GLOBAL_DIR, f"fd_test_idx.npy")

    for p in [xtr_p, ytr_p, xte_p, yte_p, fdi_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Prepare GLOBAL {W} first, e.g.: "
                f'python src/05_prepare_global.py --window {W}'
            )

    # load arrays (expect shapes [N, W, 28])
    X_train = np.load(xtr_p)
    y_train = np.load(ytr_p)
    X_test = np.load(xte_p)
    y_test = np.load(yte_p)
    fd_test = np.load(fdi_p)  # 0..3 per sample

    if X_train.ndim != 3 or X_train.shape[1] != W:
        raise RuntimeError(f"Bad train shape {X_train.shape}, expected [N,{W},28]")
    INCH = X_train.shape[2]  # should be 28 (24 + 4 domain one-hot)

    # tensors & loaders
    Xtr_t = to_tensor(X_train)
    Xte_t = to_tensor(X_test)
    ytr_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    yte_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    test_ds = torch.utils.data.TensorDataset(Xte_t, yte_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.val_batch, shuffle=False)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM_CBAM(in_ch=INCH, cnn_ch=args.cnn, lstm_h=args.lstm, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def evaluate(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pb = model(xb)
                ys.append(yb.cpu().numpy())
                ps.append(pb.cpu().numpy())
        y_true = np.concatenate(ys).ravel()
        y_pred = np.clip(np.concatenate(ps).ravel(), 0, None)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        nasa = nasa_score(y_true, y_pred)
        return rmse, nasa, y_true, y_pred

    print("[INFO] Training GLOBAL CNN-LSTM-CBAM…")
    best_rmse = 1e9
    best_state = None
    wait = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        rmse, nasa, *_ = evaluate(test_loader)
        print(f"Epoch {ep:03d} | loss={total/len(train_ds):.4f} | RMSE={rmse:.3f} | NASA={nasa:.1f}")

        if rmse < best_rmse - 0.01:
            best_rmse = rmse
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print("[INFO] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval
    rmse, nasa, y_true, y_pred = evaluate(test_loader)

    # per-domain breakdown
    perdom = {}
    for d in range(4):
        m = (fd_test == d)
        if m.sum() > 0:
            r = sqrt(mean_squared_error(y_true[m], np.clip(y_pred[m], 0, None)))
            s = nasa_score(y_true[m], np.clip(y_pred[m], 0, None))
            perdom[str(d)] = {"rmse": float(r), "nasa": float(s), "count": int(m.sum())}

    # save
    tag = f"GLOBAL_w{W}_cnn_lstm_cbam"
    out_pt = os.path.join(MODELS, f"{tag}.pt")
    with open(os.path.join(REPORTS, f"{tag}_metrics.json"), "w") as f:
        json.dump({"rmse": float(rmse), "nasa": float(nasa), "per_domain": perdom}, f, indent=2)
    torch.save(model.state_dict(), out_pt)

    print(f"[METRIC] GLOBAL RMSE={rmse:.3f} NASA={nasa:.1f}")
    print("[PER-DOMAIN]", perdom)
    print(f"[DONE] saved weights -> {out_pt}")

if __name__ == "__main__":
    main()
