# src/demo_endpoints.py
# Minimal "live demo" routes: start/stop loop, live predict, telemetry, alerts, metrics, recs, and XAI for GLOBAL model.

from typing import Dict, Any, List, Tuple
import os, time, math
import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException, FastAPI
from joblib import load as joblib_load

# ------------------------------
# Tiny CBAM net to mirror training (only for /explain_global_live)
# ------------------------------
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
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = self.sig(attn).unsqueeze(-1)
        return x * attn.expand_as(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):  # [B,C,T]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_pool, max_pool], dim=1)
        attn = self.sig(self.conv(a))
        return x * attn.expand_as(x)

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention1D(channels, reduction)
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
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_h, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):  # x [B,C,T]
        z = self.conv(x)
        z = self.cbam(z)
        z = z.permute(0, 2, 1)  # [B,T,C]
        _, (h, _) = self.lstm(z)
        return self.head(h[-1])

# ------------------------------
# Demo state
# ------------------------------
FDS_ORDER = ["FD001", "FD002", "FD003", "FD004"]
DEMO: Dict[str, Any] = {
    "running": False,
    "fd_hint": "FD003",
    "W": 50,
    "use_key": "GLOBAL50_MIXED",
    "idx": 0,
    "last_window": None,       # np.ndarray [W,24]
    "last_pred": None,         # float
    "last_ts": None,           # float
    "alerts": [],              # list of dict
    "rul_points": [],          # spark cache for metrics
}

def _one_hot_block(fd_hint: str, W: int) -> np.ndarray:
    dom = np.zeros((W, len(FDS_ORDER)), dtype=np.float32)
    dom[:, FDS_ORDER.index(fd_hint)] = 1.0
    return dom

# Load replay windows for the chosen FD (expects processed test windows)
BASE = "."
def _load_window(fd: str, W: int) -> np.ndarray:
    # path convention: ./data/processed/FD003/X_test_w50.npy (adjust if your path differs)
    path = os.path.join(BASE, "data", "processed", fd, f"X_test_w{W}.npy")
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"Missing test data: {path}")
    X = np.load(path)
    return X  # [N, W, 24]

# Simple proxy extractors (temperature / vibration / pressure) for charts & rules
def _proxy_triplet(win: np.ndarray) -> Tuple[float, float, float]:
    # win shape [W,24] (assumed scaled). Return ~0..1 proxies.
    t_idx = [2, 7, 12]   # temp-ish
    v_idx = [3, 4, 8]    # vib-ish
    p_idx = [9, 10, 11]  # press-ish
    temp = float(np.clip(np.mean(win[:, t_idx]), 0, 1))
    vib  = float(np.clip(np.mean(win[:, v_idx]), 0, 1))
    press= float(np.clip(np.mean(win[:, p_idx]), 0, 1))
    return temp, vib, press

# ------------------------------
# Attach routes
# ------------------------------
def attach_demo_routes(
    app: FastAPI,
    REGISTRY: Dict[str, Dict],
    FEATURE_COLS: List[str],
    predict_global_internal,  # callable(fd_hint, window_data, use_key) -> float
):
    # Start the demo
    @app.post("/demo/start")
    def demo_start(fd_hint: str = "FD003", window: int = 50, use: str = "GLOBAL50_MIXED"):
        DEMO["fd_hint"] = fd_hint
        DEMO["W"] = int(window)
        DEMO["use_key"] = use
        DEMO["idx"] = 0
        X = _load_window(fd_hint, DEMO["W"])
        DEMO["data"] = X
        DEMO["running"] = True
        DEMO["alerts"].clear()
        DEMO["rul_points"].clear()
        DEMO["last_window"] = X[0]
        DEMO["last_pred"] = None
        DEMO["last_ts"] = time.time()
        return {"ok": True, "fd_hint": fd_hint, "window": int(window), "use": use}

    @app.post("/demo/stop")
    def demo_stop():
        DEMO["running"] = False
        return {"ok": True}

    # Live prediction (drives RUL tile + spark + telemetry proxies)
    @app.get("/predict_global_live")
    def predict_global_live():
        if not DEMO.get("running"):
            raise HTTPException(status_code=400, detail="demo not started. POST /demo/start first")
        X = DEMO.get("data")
        if X is None or len(X) == 0:
            raise HTTPException(status_code=500, detail="no test windows loaded")
        i = DEMO["idx"] % len(X)
        win = X[i]  # [W,24]
        pred = predict_global_internal(DEMO["fd_hint"], win.tolist(), DEMO["use_key"])
        DEMO["last_window"] = win
        DEMO["last_pred"] = pred
        DEMO["last_ts"] = time.time()
        DEMO["idx"] += 1

        # keep spark history
        DEMO["rul_points"].append(float(pred))
        if len(DEMO["rul_points"]) > 60:
            DEMO["rul_points"] = DEMO["rul_points"][-60:]

        # generate simple alerts
        _append_rule_alerts(pred, win)

        temp, vib, press = _proxy_triplet(win)
        return {
            "model": DEMO["use_key"],
            "fd_hint": DEMO["fd_hint"],
            "window": DEMO["W"],
            "idx": i,
            "predicted_rul": float(pred),
            "proxies": [temp, vib, press],
            "ts": DEMO["last_ts"],
        }

    @app.get("/telemetry_live")
    def telemetry_live():
        if DEMO.get("last_window") is None:
            raise HTTPException(status_code=400, detail="no window yet; call /predict_global_live once")
        win = DEMO["last_window"]
        t, v, p = _proxy_triplet(win)
        return {
            "fd_hint": DEMO["fd_hint"],
            "window": DEMO["W"],
            "idx": (DEMO["idx"] - 1),
            "proxies": [t, v, p],
            "ts": DEMO["last_ts"],
        }

    @app.get("/alerts")
    def get_alerts():
        # return newest first
        return DEMO["alerts"][:30]

    @app.get("/metrics")
    def get_metrics():
        return {
            "model": DEMO.get("use_key"),
            "rul_points": len(DEMO.get("rul_points", [])),
            "recent_alerts": len(DEMO.get("alerts", [])),
        }

    @app.get("/recommendations")
    def get_recommendations():
        pred = DEMO.get("last_pred")
        if pred is None:
            return []
        t, v, p = _proxy_triplet(DEMO["last_window"])
        recs = []
        if pred < 40 or v > 0.55:
            due = max(1, int(pred - 10)) if pred is not None else 20
            recs.append({
                "title": "Inspect Fan module",
                "due": f"due in {due} cycles",
                "reason": "Low RUL and rising vibration proxy.",
            })
        if t > 0.6:
            recs.append({
                "title": "Thermal check",
                "due": "schedule next stop",
                "reason": "Temperature proxy trending high.",
            })
        return recs

    # ---------- XAI ----------
    @app.get("/explain_global_live")
    def explain_global_live():
        """
        Returns aggregated feature importance (avg |dY/dX| across time) for the current live window,
        using the GLOBAL_* model selected in /demo/start.
        """
        if DEMO.get("last_window") is None:
            raise HTTPException(status_code=400, detail="no window yet; call /predict_global_live once")
        fd = DEMO["fd_hint"]
        W = DEMO["W"]
        use = DEMO["use_key"]

        g = REGISTRY[use]
        weights_path = g["weights"]
        scaler_path = REGISTRY[fd]["scaler"]

        # build GLOBAL input: 24 features + 4 one-hot domains = 28 channels
        X = DEMO["last_window"].astype(np.float32)       # [W,24] (unscaled in processed file)
        scaler = joblib_load(scaler_path)
        Xs = scaler.transform(X)                          # [W,24]
        Xg = np.concatenate([Xs, _one_hot_block(fd, W)], axis=1)  # [W,28]

        in_ch = Xg.shape[1]
        model = CNNLSTM_CBAM(in_ch=in_ch, cnn_ch=96, lstm_h=96, dropout=0.1)
        if not os.path.exists(weights_path):
            raise HTTPException(status_code=500, detail=f"Missing weights: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

        xt = torch.tensor(Xg, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).clone().requires_grad_(True)  # [1,28,W]
        y = model(xt)  # [1,1]
        y.backward(torch.ones_like(y))
        sal = xt.grad.detach().abs().squeeze(0).cpu().numpy()   # [28,W]
        # split back: first 24 real features only for importance
        sal_feat = sal[:24, :]                                  # [24,W]
        imp = sal_feat.mean(axis=1)                             # [24]
        # normalize 0..1 for chart
        imp_min, imp_max = float(imp.min()), float(imp.max())
        norm = (imp - imp_min) / (imp_max - imp_min + 1e-8)

        # simple linear proxy weights for /whatif fallback
        temp_idx = [2, 7, 12]
        vib_idx  = [3, 4, 8]
        press_idx= [9, 10, 11]
        w_temp = -float(imp[temp_idx].mean()) * 5.0
        w_vib  = -float(imp[vib_idx].mean())  * 8.0
        w_press= -float(imp[press_idx].mean())* 2.0

        return {
            "model": use,
            "fd_hint": fd,
            "window": W,
            "predicted_rul": float(max(0.0, y.detach().cpu().numpy().ravel()[0])),
            "feature_names": FEATURE_COLS,
            "features": [{"name": n, "score": float(s)} for n, s in zip(FEATURE_COLS, norm.tolist())],
            "window_saliency": sal_feat.T.tolist(),  # [W,24]
            "proxy_weights": {"temp": w_temp, "vib": w_vib, "press": w_press},
            "ts": DEMO["last_ts"],
        }

    @app.post("/whatif")
    def what_if(delta_temp: float = 0.0, delta_vib: float = 0.0, delta_press: float = 0.0):
        """
        Server-side what-if: linearized around current point using proxy weights from /explain_global_live.
        """
        xai = explain_global_live()
        w = xai["proxy_weights"]
        curr = float(DEMO.get("last_pred") or 0.0)
        delta = (w["temp"] * float(delta_temp)) + (w["vib"] * float(delta_vib)) + (w["press"] * float(delta_press))
        return {"adjusted_rul": float(max(0.0, curr + delta))}

    # ---------------- helpers ----------------
    def _append_rule_alerts(pred: float, win: np.ndarray):
        """
        Compose a few rules so the Alerts panel shows more than RUL.
        """
        ts = DEMO["last_ts"] or time.time()
        # RUL threshold
        if pred < 40:
            DEMO["alerts"].insert(0, {
                "id": f"a-{int(ts)}-rul",
                "severity": "warning",
                "title": "RUL getting low",
                "message": f"Estimated RUL {pred:.1f} cycles (<40).",
                "created_at": ts,
                "acknowledged": False,
            })
        # Vibration rising (proxy trend)
        v_first = float(np.mean(win[: len(win)//2, [3,4,8]]))
        v_last  = float(np.mean(win[len(win)//2 :, [3,4,8]]))
        if v_last - v_first > 0.15:
            DEMO["alerts"].insert(0, {
                "id": f"a-{int(ts)}-vib",
                "severity": "warning",
                "title": "Vibration spike",
                "message": f"Rising vibration trend (+{(v_last - v_first):.2f}).",
                "created_at": ts,
                "acknowledged": False,
            })
        # Temperature rising
        t_first = float(np.mean(win[: len(win)//2, [2,7,12]]))
        t_last  = float(np.mean(win[len(win)//2 :, [2,7,12]]))
        if t_last - t_first > 0.12:
            DEMO["alerts"].insert(0, {
                "id": f"a-{int(ts)}-temp",
                "severity": "warning",
                "title": "Temperature rising",
                "message": f"Unusual temp increase (+{(t_last - t_first):.2f}).",
                "created_at": ts,
                "acknowledged": False,
            })
        # keep last 30 alerts
        DEMO["alerts"] = DEMO["alerts"][:30]
