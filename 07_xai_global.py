# Usage:
#   python src/07_xai_global.py --fd_hint FD003 --window 30 --use GLOBAL
#   python src/07_xai_global.py --fd_hint FD003 --window 50 --use GLOBAL50
#   python src/07_xai_global.py --fd_hint FD003 --window 50 --use GLOBAL50_MIXED

import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
from joblib import load as joblib_load
import matplotlib.pyplot as plt

# ==== shared model pieces (same as training/server) ====
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
        avg_pool = torch.mean(x, dim=2)
        max_pool, _ = torch.max(x, dim=2)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = self.sig(attn).unsqueeze(-1)
        return x * attn.expand_as(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        p = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=p, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_pool, max_pool], dim=1)
        attn = self.sig(self.conv(a))
        return x * attn.expand_as(x)

class CBAM1D(nn.Module):
    def __init__(self, ch, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention1D(ch, reduction)
        self.sa = SpatialAttention1D(kernel_size)
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
            nn.Linear(64, 1)
        )
    def forward(self, x):  # [B, C, T]
        z = self.conv(x)
        z = self.cbam(z)
        z = z.permute(0, 2, 1)    # [B,T,C]
        _, (h, _) = self.lstm(z)
        return self.head(h[-1])

FEATURES = ["setting_1","setting_2","setting_3"] + [f"sensor_{i}" for i in range(1,22)]
FDS_ORDER = ["FD001","FD002","FD003","FD004"]

def one_hot_block(fd_hint: str, W: int) -> np.ndarray:
    dom = np.zeros((W, len(FDS_ORDER)), dtype=np.float32)
    idx = FDS_ORDER.index(fd_hint)
    dom[:, idx] = 1.0
    return dom

def integrated_gradients(model, x, steps=64):
    model.eval()
    baseline = torch.zeros_like(x)
    ig = torch.zeros_like(x)
    for i in range(1, steps+1):
        alpha = float(i) / steps
        xi = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        yi = model(xi)
        grads = torch.autograd.grad(outputs=yi, inputs=xi,
                                    grad_outputs=torch.ones_like(yi),
                                    retain_graph=False)[0]
        ig += grads
    ig = (x - baseline) * ig / steps
    return ig.detach()

def main(fd_hint: str, window: int, sample_idx: int, use_key: str):
    BASE = "."
    REG_PATH = os.path.join(BASE, "serving", "model_registry.json")
    reg = json.load(open(REG_PATH, "r", encoding="utf-8"))

    # pick global entry
    if use_key not in reg:
        raise RuntimeError(f"Model key '{use_key}' not found in serving/model_registry.json")
    g = reg[use_key]
    W = int(g["window"])
    assert W == window, f"{use_key} expects window={W}"

    weights_path = g["weights"]
    scaler_path = reg[fd_hint]["scaler"]

    # input window from hinted FD
    X_path = os.path.join(BASE, "data", "processed", fd_hint, f"X_test_w{W}.npy")
    X_all = np.load(X_path)         # [N,W,24]
    X = X_all[sample_idx]           # [W,24]

    # scale + append one-hot
    scaler = joblib_load(scaler_path)
    Xs = scaler.transform(X)        # [W,24]
    Xg = np.concatenate([Xs, one_hot_block(fd_hint, W)], axis=1)  # [W,28]

    # model
    in_ch = Xg.shape[1]  # 28
    model = CNNLSTM_CBAM(in_ch=in_ch, cnn_ch=96, lstm_h=96)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Torch input [1,C,T]
    xt = torch.tensor(Xg, dtype=torch.float32).unsqueeze(0).permute(0,2,1).clone().requires_grad_(True)

    # forward + saliency
    y = model(xt)
    y.backward(torch.ones_like(y))
    sal = xt.grad.detach().abs().squeeze(0).cpu().numpy()   # [28,W]
    sal = sal.T                                            # [W,28]

    # IG
    ig_t = integrated_gradients(model, xt, steps=64)        # [1,28,W]
    ig = ig_t.squeeze(0).permute(1,0).cpu().numpy()         # [W,28]

    # split (24 features vs 4 domain channels)
    sal_feat, sal_dom = sal[:, :24], sal[:, 24:]            # [W,24], [W,4]
    ig_feat,  ig_dom  = ig[:,  :24], ig[:,  24:]

    # save
    outdir = os.path.join(BASE, "data", "reports", "xai", f"{use_key}_{fd_hint}")
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "saliency_feat.npy"), sal_feat)
    np.save(os.path.join(outdir, "saliency_dom.npy"),  sal_dom)
    np.save(os.path.join(outdir, "ig_feat.npy"),       ig_feat)
    np.save(os.path.join(outdir, "ig_dom.npy"),        ig_dom)

    # quick plots
    plt.figure(figsize=(10,4)); plt.imshow(sal_feat, aspect="auto"); plt.title(f"{use_key} saliency |dY/dX| (features)")
    plt.xlabel("Feature (24)"); plt.ylabel(f"Time ({W})"); plt.colorbar(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "saliency_feat.png"), dpi=150); plt.close()

    plt.figure(figsize=(10,4)); plt.imshow(ig_feat, aspect="auto"); plt.title(f"{use_key} Integrated Gradients (features)")
    plt.xlabel("Feature (24)"); plt.ylabel(f"Time ({W})"); plt.colorbar(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ig_feat.png"), dpi=150); plt.close()

    # Domain channels bar (last timestep)
    last_dom = sal_dom[-1]  # [4]
    plt.figure(figsize=(5,3)); plt.bar(range(4), last_dom); plt.title("Domain saliency at last step")
    plt.xticks(range(4), ["FD001","FD002","FD003","FD004"]); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "saliency_domain_last.png"), dpi=150); plt.close()

    print("[DONE] Saved to", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd_hint", type=str, default="FD003", choices=["FD001","FD002","FD003","FD004"])
    ap.add_argument("--window",  type=int, default=30)
    ap.add_argument("--idx",     type=int, default=0, help="sample index from the test set of fd_hint")
    ap.add_argument("--use",     type=str, default="GLOBAL",
                    choices=["GLOBAL","GLOBAL50","GLOBAL50_MIXED"],
                    help="Pick the global model")
    args = ap.parse_args()
    main(args.fd_hint, args.window, args.idx, args.use)
