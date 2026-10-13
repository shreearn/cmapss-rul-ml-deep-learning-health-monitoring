# src/06b_train_global_mixed.py
import os, json
import numpy as np
import torch, torch.nn as nn
from math import sqrt
from sklearn.metrics import mean_squared_error

BASE = "."
DATA = os.path.join(BASE, "data", "mix", "GLOBAL_w50")
MODELS = os.path.join(BASE, "data", "models")
REPORTS = os.path.join(BASE, "data", "reports")
os.makedirs(MODELS, exist_ok=True); os.makedirs(REPORTS, exist_ok=True)

W, C = 50, 28   # 24 features + 4 one-hot
DOMS = 4

def nasa_score(y_true, y_pred):
    s = 0.0
    for yt, yp in zip(y_true, y_pred):
        d = yp - yt
        s += (np.exp(-d/13) - 1) if d < 0 else (np.exp(d/10) - 1)
    return float(s)

# --- load mixed train (we'll hold out 15% as a test split just for sanity) ---
X = np.load(os.path.join(DATA, "X_train_w50.npy"))   # [N,50,28]
y = np.load(os.path.join(DATA, "y_train_w50.npy"))   # [N]
fd = np.load(os.path.join(DATA, "fd_train_idx.npy")) # [N] (-1 if real had no per-row fd)

# simple split
N = len(y)
perm = np.random.RandomState(42).permutation(N)
cut = int(0.85 * N)
tr_idx, te_idx = perm[:cut], perm[cut:]
Xtr, ytr, fdtr = X[tr_idx], y[tr_idx], fd[tr_idx]
Xte, yte, fdte = X[te_idx], y[te_idx], fd[te_idx]

# to tensors [B,C,T]
def to_tensor(x): return torch.tensor(x, dtype=torch.float32).permute(0,2,1)
Xtr_t, Xte_t = to_tensor(Xtr), to_tensor(Xte)
ytr_t, yte_t = torch.tensor(ytr, dtype=torch.float32).view(-1,1), torch.tensor(yte, dtype=torch.float32).view(-1,1)

train_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
test_ds  = torch.utils.data.TensorDataset(Xte_t, yte_t)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)

# --- model (same as your CBAM net for global) ---
class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden, bias=False),
                                 nn.ReLU(), nn.Linear(hidden, in_channels, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg_pool = torch.mean(x, dim=2)
        mx, _ = torch.max(x, dim=2)
        a = self.mlp(avg_pool) + self.mlp(mx)
        return x * self.sig(a).unsqueeze(-1).expand_as(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        pad=(k-1)//2
        self.conv=nn.Conv1d(2,1,kernel_size=k,padding=pad,bias=False)
        self.sig=nn.Sigmoid()
    def forward(self,x):
        avg=torch.mean(x,dim=1,keepdim=True)
        mx,_=torch.max(x,dim=1,keepdim=True)
        a=torch.cat([avg,mx],dim=1)
        return x * self.sig(self.conv(a)).expand_as(x)

class CBAM1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca=ChannelAttention1D(ch); self.sa=SpatialAttention1D()
    def forward(self,x): return self.sa(self.ca(x))

class CNNLSTM_CBAM(nn.Module):
    def __init__(self, in_ch, cnn_ch=96, lstm_h=96, dropout=0.1):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(in_ch,cnn_ch,3,padding=1), nn.ReLU(),
            nn.Conv1d(cnn_ch,cnn_ch,3,padding=1), nn.ReLU(),
        )
        self.cbam=CBAM1D(cnn_ch)
        self.lstm=nn.LSTM(input_size=cnn_ch, hidden_size=lstm_h, batch_first=True)
        self.head=nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_h,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x):
        z=self.cbam(self.conv(x))
        z=z.permute(0,2,1)
        _,(h,_) = self.lstm(z)
        return self.head(h[-1])

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNNLSTM_CBAM(in_ch=C, cnn_ch=96, lstm_h=96, dropout=0.1).to(device)
opt=torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn=nn.MSELoss()

def evaluate():
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb,yb in test_loader:
            xb,yb=xb.to(device), yb.to(device)
            pb=model(xb)
            ys.append(yb.cpu().numpy()); ps.append(pb.cpu().numpy())
    y_true=np.concatenate(ys).ravel()
    y_pred=np.clip(np.concatenate(ps).ravel(), 0, None)
    rmse=sqrt(mean_squared_error(y_true,y_pred))
    nasa=nasa_score(y_true,y_pred)
    return rmse, nasa, y_true, y_pred

best_rmse=1e9; best=None; patience=12; wait=0
print("[INFO] Training GLOBAL50_MIXED CNN-LSTM-CBAM…")
for ep in range(1,200):
    model.train(); total=0.0
    for xb,yb in train_loader:
        xb,yb=xb.to(device), yb.to(device)
        opt.zero_grad(); pred=model(xb); loss=loss_fn(pred,yb)
        loss.backward(); opt.step()
        total+=loss.item()*xb.size(0)
    rmse, nasa, *_ = evaluate()
    print(f"Epoch {ep:03d} | loss={total/len(train_ds):.4f} | RMSE={rmse:.3f} | NASA={nasa:.1f}")
    if rmse < best_rmse - 0.01:
        best_rmse=rmse; wait=0
        best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        wait+=1
        if wait>=patience:
            print("[INFO] Early stopping."); break

if best is not None:
    model.load_state_dict(best)

rmse, nasa, y_true, y_pred = evaluate()
tag = "GLOBAL_w50_mixed_cnn_lstm_cbam"
wpath = os.path.join(MODELS, f"{tag}.pt")
torch.save(model.state_dict(), wpath)
with open(os.path.join(REPORTS, f"{tag}_metrics.json"), "w") as f:
    json.dump({"rmse":rmse,"nasa":nasa}, f, indent=2)

print(f"[METRIC] GLOBAL50_MIXED RMSE={rmse:.3f} NASA={nasa:.1f}")
print(f"[DONE] saved -> {wpath}")
