# src/03_train_cnn_lstm_cbam.py
# Train CNN-LSTM-CBAM for a given FD and window W, and save with a clear tag.

# --- config (define before any use) ---
FD = "FD003" 
W = 50
# FD = "FD002"   # <-- set which FD you're training
# W  = 50        # <-- window length you're using
SAVE_TAG = f"{FD}_w{W}_cnn_lstm_cbam"

import os, json, time
import numpy as np
import torch
import torch.nn as nn
from math import sqrt
from sklearn.metrics import mean_squared_error

BASE = "."
PROCESSED = os.path.join(BASE, "data", "processed", FD)
MODELS    = os.path.join(BASE, "data", "models")
REPORTS   = os.path.join(BASE, "data", "reports")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)


def nasa_score(y_true, y_pred):
    s = 0.0
    for yt, yp in zip(y_true, y_pred):
        d = yp - yt
        s += (np.exp(-d/13) - 1) if d < 0 else (np.exp(d/10) - 1)
    return float(s)

# ---------- Data ----------
X_train = np.load(os.path.join(PROCESSED, f"X_train_w{W}.npy"))
y_train = np.load(os.path.join(PROCESSED, f"y_train_w{W}.npy"))
X_test  = np.load(os.path.join(PROCESSED, f"X_test_w{W}.npy"))
y_test  = np.load(os.path.join(PROCESSED, f"y_test_w{W}.npy"))

# Torch tensors for Conv1d: [B, C, T]
X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0,2,1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).permute(0,2,1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1,1)

train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
test_ds  = torch.utils.data.TensorDataset(X_test_t,  y_test_t)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=512, shuffle=False)

in_ch = X_train_t.shape[1]  # 24 features

# ---------- CBAM for 1D ----------
class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, in_channels, bias=False)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # x: [B, C, T]; pool over T
        avg_pool = torch.mean(x, dim=2)                  # [B, C]
        max_pool, _ = torch.max(x, dim=2)                # [B, C]
        attn = self.mlp(avg_pool) + self.mlp(max_pool)   # [B, C]
        attn = self.sig(attn).unsqueeze(-1)              # [B, C, 1]
        return x * attn.expand_as(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # x: [B, C, T]; pool over C to get 2-channel map
        avg_pool = torch.mean(x, dim=1, keepdim=True)             # [B, 1, T]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)           # [B, 1, T]
        a = torch.cat([avg_pool, max_pool], dim=1)                # [B, 2, T]
        attn = self.sig(self.conv(a))                             # [B, 1, T]
        return x * attn.expand_as(x)

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.sa = SpatialAttention1D(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))

# ---------- Model ----------
class CNNLSTM_CBAM(nn.Module):
    def __init__(self, in_ch, cnn_ch=64, lstm_h=64, dropout=0.1):
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
            nn.Linear(64, 1)
        )
    def forward(self, x):                 # x: [B, C, T]
        z = self.conv(x)                  # [B, cnn_ch, T]
        z = self.cbam(z)                  # attention
        z = z.permute(0,2,1)              # [B, T, cnn_ch] for LSTM
        _, (h, _) = self.lstm(z)
        out = self.head(h[-1])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM_CBAM(in_ch=in_ch, cnn_ch=96, lstm_h=96, dropout=0.1).to(device) # FD2 & FD4
opt = torch.optim.Adam(model.parameters(), lr=5e-4)  # lower LR  # FD2
# model = CNNLSTM_CBAM(in_ch=in_ch, cnn_ch=64, lstm_h=64, dropout=0.1).to(device)   # FD1
# opt = torch.optim.Adam(model.parameters(), lr=1e-3)    # FD1
loss_fn = nn.MSELoss()

best_rmse, best = 1e9, None
patience, wait = 15, 0  # FD2 & FD 4
# patience, wait = 10, 0  
# EPOCHS = 100  
EPOCHS = 150    # FD2 & FD4

def evaluate():
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pb = model(xb)
            ys.append(yb.cpu().numpy())
            ps.append(pb.cpu().numpy())
    y_true = np.concatenate(ys).ravel()
    y_pred = np.clip(np.concatenate(ps).ravel(), 0, None)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    score = nasa_score(y_true, y_pred)
    return rmse, score, y_true, y_pred

print("[INFO] Training CNN-LSTM-CBAM…")
for ep in range(1, EPOCHS+1):
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
    rmse, score, _, _ = evaluate()
    print(f"Epoch {ep:03d} | loss={total/len(train_ds):.4f} | RMSE={rmse:.3f} | NASA={score:.1f}")
    if rmse < best_rmse - 0.01:
        best_rmse, wait = rmse, 0
        best = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("[INFO] Early stopping.")
            break

# Save best FD1
'''
if best is not None:
    model.load_state_dict(best)
torch.save(model.state_dict(), os.path.join(MODELS, f"{FD}_cnn_lstm_cbam.pt"))

# Final eval + save artifacts
rmse, score, y_true, y_pred = evaluate()
np.save(os.path.join(REPORTS, f"{FD}_cnn_lstm_cbam_preds.npy"), np.vstack([y_true, y_pred]))
with open(os.path.join(REPORTS, f"{FD}_cnn_lstm_cbam_metrics.json"), "w") as f:
    json.dump({"rmse": rmse, "nasa": score}, f, indent=2)
'''

# save best fd2 and 3
# Save best checkpoint and final artifacts (tagged)
if best is not None:
    model.load_state_dict(best)
torch.save(model.state_dict(), os.path.join(MODELS, f"{SAVE_TAG}.pt"))

rmse, score, y_true, y_pred = evaluate()
np.save(os.path.join(REPORTS, f"{SAVE_TAG}_preds.npy"), np.vstack([y_true, y_pred]))
with open(os.path.join(REPORTS, f"{SAVE_TAG}_metrics.json"), "w") as f:
    json.dump({"rmse": rmse, "nasa": score}, f, indent=2)

# quick latency check (single forward)
model.eval()
xb = X_test_t[:1].to(device)
t0 = time.time()
with torch.no_grad():
    _ = model(xb)
lat_ms = (time.time() - t0) * 1000

print(f"[METRIC] CNN-LSTM-CBAM RMSE={rmse:.3f} NASA={score:.1f}  (latency≈{lat_ms:.1f} ms)")
print("[DONE] saved weights & reports.")

