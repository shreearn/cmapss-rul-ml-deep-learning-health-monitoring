# src/04_xai_seq_explain.py
# Usage: python src/04_xai_seq_explain.py
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
# FD = "FD001"
# FD = "FD002"
FD = "FD003"
BASE = "."
PROCESSED = os.path.join(BASE, "data", "processed", FD)
MODELS    = os.path.join(BASE, "data", "models")
OUTDIR    = os.path.join(BASE, "data", "reports", "xai", FD)
os.makedirs(OUTDIR, exist_ok=True)

W = 50  
# W  = 50 # FD2
# WEIGHTS_PATH = os.path.join(MODELS, f"{FD}_cnn_lstm_cbam.pt") # FD 1
WEIGHTS_PATH = os.path.join(MODELS, f"{FD}_w{W}_cnn_lstm_cbam.pt") 
X_TEST_PATH  = os.path.join(PROCESSED, f"X_test_w{W}.npy")

# ----------------- MODEL (same as training) -----------------
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
    def forward(self, x):  # x: [B, C, T]
        avg_pool = torch.mean(x, dim=2)            # [B, C]
        max_pool, _ = torch.max(x, dim=2)          # [B, C]
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = self.sig(attn).unsqueeze(-1)        # [B, C, 1]
        return x * attn.expand_as(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):  # x: [B, C, T]
        avg_pool = torch.mean(x, dim=1, keepdim=True)   # [B, 1, T]
        max_pool, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, T]
        a = torch.cat([avg_pool, max_pool], dim=1)      # [B, 2, T]
        attn = self.sig(self.conv(a))                   # [B, 1, T]
        return x * attn.expand_as(x)

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.sa = SpatialAttention1D(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))

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
    def forward(self, x):  # x: [B, C, T]
        z = self.conv(x)            # [B, cnn_ch, T]
        z = self.cbam(z)            # attention
        z = z.permute(0,2,1)        # [B, T, cnn_ch]
        _, (h, _) = self.lstm(z)    # last hidden
        return self.head(h[-1])     # [B, 1]

# ----------------- LOAD DATA/WEIGHTS -----------------
X_test = np.load(X_TEST_PATH)                     # [N, T, C] = [N, 30, 24]
x = torch.tensor(X_test[0:1], dtype=torch.float32)        # pick first sample
x_t = x.permute(0, 2, 1).clone().requires_grad_(True)     # [1, C, T] for Conv1d

in_ch = x_t.shape[1]
model = CNNLSTM_CBAM(in_ch=in_ch) # fd1
model = CNNLSTM_CBAM(in_ch=in_ch, cnn_ch=96, lstm_h=96, dropout=0.1) # fd2
# model = CNNLSTM_CBAM(in_ch=in_ch, cnn_ch=64, lstm_h=64, dropout=0.1)
# model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

# ----------------- SALIENCY -----------------
# |∂y/∂x| at the chosen input
out = model(x_t)
out.backward(torch.ones_like(out))
saliency = x_t.grad.detach().abs().squeeze(0).numpy()  # [C, T]
saliency = saliency.T  # -> [T, C] for plotting as (time, features)

# ----------------- INTEGRATED GRADIENTS -----------------
def integrated_gradients(model, x, baseline=None, steps=64):
    """
    x: [1, C, T] (no grad required on entry)
    returns: [1, C, T] tensor of IG attributions
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)

    ig = torch.zeros_like(x)
    for i in range(1, steps + 1):
        alpha = float(i) / steps
        xi = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)  # LEAF
        yi = model(xi)  # [1, 1]
        grads = torch.autograd.grad(
            outputs=yi, inputs=xi, grad_outputs=torch.ones_like(yi), retain_graph=False
        )[0]  # [1, C, T]
        ig += grads

    ig = (x - baseline) * ig / steps
    return ig.detach()

# compute IG and reshape for plotting
ig_t = integrated_gradients(model, x_t, steps=64)           # [1, C, T]
ig = ig_t.squeeze(0).permute(1, 0).cpu().numpy()            # -> [T, C] (time, features)

# ----------------- SAVE ARTIFACTS -----------------
# Save raw arrays (for further analysis)
np.save(os.path.join(OUTDIR, "saliency.npy"), saliency)
np.save(os.path.join(OUTDIR, "integrated_gradients.npy"), ig)

# Heatmaps
plt.figure(figsize=(10, 4))
plt.imshow(saliency, aspect="auto")
plt.xlabel("Feature (24)")
plt.ylabel("Time (30)")
plt.title("Saliency |dY/dX|")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "saliency_heatmap.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 4))
plt.imshow(ig, aspect="auto")
plt.xlabel("Feature (24)")
plt.ylabel("Time (30)")
plt.title("Integrated Gradients")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ig_heatmap.png"), dpi=150)
plt.close()

print("[DONE] Saved:",
      os.path.join(OUTDIR, "saliency_heatmap.png"),
      "and",
      os.path.join(OUTDIR, "ig_heatmap.png"))
