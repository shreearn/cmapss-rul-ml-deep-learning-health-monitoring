# src/models_seq.py
import torch
import torch.nn as nn

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
        avg_pool = torch.mean(x, dim=2)              # [B, C]
        max_pool, _ = torch.max(x, dim=2)            # [B, C]
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = self.sig(attn).unsqueeze(-1)          # [B, C, 1]
        return x * attn.expand_as(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):  # x: [B, C, T]
        avg_pool = torch.mean(x, dim=1, keepdim=True)       # [B, 1, T]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)     # [B, 1, T]
        a = torch.cat([avg_pool, max_pool], dim=1)          # [B, 2, T]
        attn = self.sig(self.conv(a))                       # [B, 1, T]
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
        z = self.conv(x)
        z = self.cbam(z)
        z = z.permute(0,2,1)                 # [B, T, C]
        _, (h, _) = self.lstm(z)
        return self.head(h[-1])
