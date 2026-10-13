# src/server.py
# One-file backend: model serving (GLOBAL / GLOBAL50 / GLOBAL50_MIXED),
# demo simulator (replay CMAPSS windows "live"), alerts (RUL + proxy anomalies),
# telemetry, XAI (integrated gradients), what-if, simple planner stubs,
# and static hosting of the dashboard in /web.

import os, json, time, math, threading, random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load as joblib_load
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse

# ---------------- Model (same as training) ----------------
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
            nn.Conv1d(in_ch, cnn_ch, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(cnn_ch, cnn_ch, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.cbam = CBAM1D(cnn_ch)
        self.lstm = nn.LSTM(input_size=cnn_ch, hidden_size=lstm_h, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_h, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):  # x: [B,C,T]
        z = self.conv(x); z = self.cbam(z); z = z.permute(0,2,1)
        _, (h, _) = self.lstm(z)
        return self.head(h[-1])  # [B,1]

# ---------------- Shared constants / registry ----------------
FEATURE_COLS = ["setting_1","setting_2","setting_3"] + [f"sensor_{i}" for i in range(1,22)]
FDS_ORDER = ["FD001","FD002","FD003","FD004"]

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG_PATH = os.path.join(BASE, "serving", "model_registry.json")
if not os.path.exists(REG_PATH):
    raise RuntimeError(f"Missing {REG_PATH}")
REGISTRY: Dict[str, Dict] = json.load(open(REG_PATH, "r", encoding="utf-8"))

def one_hot_block(fd_hint: str, W: int) -> np.ndarray:
    dom = np.zeros((W, len(FDS_ORDER)), dtype=np.float32)
    if fd_hint not in FDS_ORDER:
        raise HTTPException(status_code=400, detail=f"fd_hint must be one of {FDS_ORDER}")
    dom[:, FDS_ORDER.index(fd_hint)] = 1.0
    return dom

# ---------------- FastAPI app & static hosting ----------------
app = FastAPI(title="CMAPSS RUL API • Full Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Serve /web (dashboard)
WEB_DIR = os.path.join(BASE, "web")
os.makedirs(WEB_DIR, exist_ok=True)
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

@app.get("/")
def _root():
    return RedirectResponse(url="/web/dashboard.html")

# ---------------- Simple per-FD loaders for single-FD endpoints ----------------
_MODEL_CACHE: Dict[str, Tuple[nn.Module, object, int]] = {}

def _infer_fd_width(window: int, fd: str) -> Tuple[int,int]:
    return (96,96) if (window >= 50 or fd in ("FD002","FD004")) else (64,64)

def _load_fd(fd: str):
    if fd in _MODEL_CACHE:
        return _MODEL_CACHE[fd]
    if fd not in REGISTRY:
        raise HTTPException(400, f"Unknown fd '{fd}'")
    entry = REGISTRY[fd]
    window = int(entry["window"])
    scaler = joblib_load(entry["scaler"])
    cnn_ch, lstm_h = _infer_fd_width(window, fd)
    model = CNNLSTM_CBAM(in_ch=len(FEATURE_COLS), cnn_ch=cnn_ch, lstm_h=lstm_h)
    if not os.path.exists(entry["weights"]):
        raise HTTPException(500, f"Missing weights: {entry['weights']}")
    model.load_state_dict(torch.load(entry["weights"], map_location="cpu"))
    model.eval()
    _MODEL_CACHE[fd] = (model, scaler, window)
    return _MODEL_CACHE[fd]

# ---------------- Global (domain-aware) loader ----------------
def _predict_global(fd_hint: str, window_data: np.ndarray, use_key: str) -> float:
    if use_key not in REGISTRY:
        raise HTTPException(400, f"Unknown global key '{use_key}'")
    g = REGISTRY[use_key]
    W = int(g["window"])
    if window_data.shape != (W, len(FEATURE_COLS)):
        raise HTTPException(400, f"window_data must be [{W}, {len(FEATURE_COLS)}]")
    scaler_path = REGISTRY[fd_hint]["scaler"]
    scaler = joblib_load(scaler_path)
    Xs = scaler.transform(window_data)                             # [W,24]
    Xg = np.concatenate([Xs, one_hot_block(fd_hint, W)], axis=1)   # [W,28]
    xt = torch.tensor(Xg, dtype=torch.float32).unsqueeze(0).permute(0,2,1)  # [1,28,W]
    model = CNNLSTM_CBAM(in_ch=28, cnn_ch=96, lstm_h=96)
    if not os.path.exists(g["weights"]):
        raise HTTPException(500, f"Missing weights: {g['weights']}")
    model.load_state_dict(torch.load(g["weights"], map_location="cpu"))
    model.eval()
    with torch.no_grad():
        return float(max(0.0, model(xt).cpu().numpy().ravel()[0]))

# ---------------- Public schemas ----------------
class PredictRequest(BaseModel):
    fd: str
    window_data: List[List[float]]

class PredictResponse(BaseModel):
    fd: str
    window: int
    predicted_rul: float

class PredictGlobalRequest(BaseModel):
    fd_hint: str
    window_data: List[List[float]]
    use: Optional[str] = "GLOBAL50_MIXED"

class PredictGlobalResponse(BaseModel):
    model: str
    fd_hint: str
    window: int
    predicted_rul: float

# ---------------- Classic endpoints (keep old ones working) ----------------
@app.get("/health")
def health():
    return {"ok": True, "fds": list(REGISTRY.keys())}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model, scaler, window = _load_fd(req.fd)
    X = np.array(req.window_data, dtype=np.float32)
    if X.shape != (window, len(FEATURE_COLS)):
        raise HTTPException(400, f"window_data must have shape [{window}, {len(FEATURE_COLS)}] for FD {req.fd}")
    xt = torch.tensor(scaler.transform(X), dtype=torch.float32).unsqueeze(0).permute(0,2,1)
    with torch.no_grad():
        p = float(max(0.0, model(xt).cpu().numpy().ravel()[0]))
    return PredictResponse(fd=req.fd, window=window, predicted_rul=p)

@app.post("/predict_global", response_model=PredictGlobalResponse)
def predict_global(req: PredictGlobalRequest):
    use_key = req.use or "GLOBAL50_MIXED"
    g = REGISTRY[use_key]
    W = int(g["window"])
    X = np.array(req.window_data, dtype=np.float32)
    p = _predict_global(req.fd_hint, X, use_key)
    return PredictGlobalResponse(model=use_key, fd_hint=req.fd_hint, window=W, predicted_rul=p)

# ---------------- DEMO SIMULATOR (real-time replay) ----------------
# State shared by live endpoints
_demo = {
    "running": False,
    "fd_hint": "FD003",
    "use": "GLOBAL50_MIXED",
    "W": 50,
    "idx": 0,
    "X": None,          # full test windows [N,W,24]
    "last_window": None,# [W,24]
    "last_pred": None,  # float
    "pred_trace": [],   # last ~50 values
    "telemetry": {      # proxy normalized traces for 3 sensors
        "temp": [], "vib": [], "press": []
    },
    "alerts": [],       # recent alerts (dicts)
    "last_ts": None
}

def _load_test_windows(fd_hint: str, W: int) -> np.ndarray:
    path = os.path.join(BASE, "data", "processed", fd_hint, f"X_test_w{W}.npy")
    if not os.path.exists(path):
        raise RuntimeError(f"Missing {path}")
    return np.load(path)  # [N,W,24]

def _norm01(a):
    if len(a)==0: return a
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi-lo < 1e-9: return [0.5]*len(a)
    return [(x-lo)/(hi-lo) for x in a]

def _maybe_alert(kind: str, msg: str):
    _demo["alerts"].insert(0, {
        "id": f"a-{int(time.time())}-{kind}",
        "severity": "critical" if kind=="crit" else "warning",
        "title": "RUL critically low" if kind=="crit" else "RUL getting low",
        "message": msg,
        "created_at": time.time(),
        "acknowledged": False
    })
    _demo["alerts"] = _demo["alerts"][:20]

def _proxy_alert(title: str, msg: str, severity="warning"):
    _demo["alerts"].insert(0, {
        "id": f"a-{int(time.time()*1000)}-{severity}",
        "severity": severity,
        "title": title,
        "message": msg,
        "created_at": time.time(),
        "acknowledged": False
    })
    _demo["alerts"] = _demo["alerts"][:20]

def _step_once():
    fd_hint, W, use = _demo["fd_hint"], _demo["W"], _demo["use"]
    X = _demo["X"]; i = _demo["idx"]
    if i >= len(X)-1: _demo["idx"] = 0; i = 0
    win = X[i]
    _demo["last_window"] = win.tolist()
    pred = _predict_global(fd_hint, win, use)
    _demo["last_pred"] = pred
    _demo["pred_trace"].append(pred); _demo["pred_trace"] = _demo["pred_trace"][-50:]
    _demo["last_ts"] = time.time()
    # proxies from 3 arbitrary sensors (normalized for charting)
    s_temp = win[:, 14]  # arbitrary but stable indices within 24
    s_vib  = win[:,  9]
    s_pres = win[:,  3]
    _demo["telemetry"]["temp"].append(float(np.mean(s_temp)))
    _demo["telemetry"]["vib"].append( float(np.mean(s_vib)))
    _demo["telemetry"]["press"].append(float(np.mean(s_pres)))
    for k in ("temp","vib","press"): _demo["telemetry"][k] = _demo["telemetry"][k][-80:]

    # basic rules
    if pred < 20: _maybe_alert("crit", f"Estimated RUL {pred:.1f} cycles (<20).")
    elif pred < 40: _maybe_alert("warn", f"Estimated RUL {pred:.1f} cycles (<40).")

    # simple anomaly heuristics on proxy slopes
    for name, trace in _demo["telemetry"].items():
        if len(trace) >= 8:
            slope = (trace[-1] - trace[-8]) / max(1e-6, abs(trace[-8])+1e-6)
            if name=="vib" and slope > 0.15:
                _proxy_alert("Vibration spike", f"Rising vibration trend (+{slope:.2f}).", "warning")
            if name=="temp" and slope > 0.12:
                _proxy_alert("Temperature rising", f"Unusual temp increase (+{slope:.2f}).", "warning")
            if name=="press" and slope < -0.12:
                _proxy_alert("Pressure drop", f"Pressure decreasing ({slope:.2f}).", "warning")

    _demo["idx"] += 1

def _loop():
    while _demo["running"]:
        try: _step_once()
        except Exception as e: print("[SIM ERR]", e)
        time.sleep(2.0)

@app.post("/demo/start")
def demo_start(fd_hint: str = "FD003", window: int = 50, use: str = "GLOBAL50_MIXED"):
    _demo["fd_hint"] = fd_hint; _demo["W"] = int(window); _demo["use"] = use
    _demo["X"] = _load_test_windows(fd_hint, int(window))
    _demo["idx"] = random.randint(5, 30)
    _demo["pred_trace"].clear(); _demo["telemetry"] = {"temp":[], "vib":[], "press":[]}
    _demo["alerts"].clear()
    if not _demo.get("running"):
        _demo["running"] = True
        threading.Thread(target=_loop, daemon=True).start()
    return {"ok": True, "fd_hint": fd_hint, "window": window, "use": use}

@app.post("/demo/stop")
def demo_stop():
    _demo["running"] = False
    return {"ok": True}

# ---------------- Live endpoints used by the dashboard ----------------
@app.get("/predict_global_live")
def predict_global_live():
    if _demo["last_window"] is None:
        raise HTTPException(409, "Demo not started. POST /demo/start first.")
    return {
        "model": _demo["use"],
        "fd_hint": _demo["fd_hint"],
        "window": _demo["W"],
        "idx": _demo["idx"],
        "predicted_rul": _demo["last_pred"],
        "trace": _demo["pred_trace"],
        "ts": _demo["last_ts"],
    }

@app.get("/telemetry_live")
def telemetry_live():
    if _demo["last_window"] is None:
        raise HTTPException(409, "Demo not started.")
    # return normalized series for plotting
    temp = _norm01(_demo["telemetry"]["temp"])
    vib  = _norm01(_demo["telemetry"]["vib"])
    press= _norm01(_demo["telemetry"]["press"])
    return {"temp": temp, "vib": vib, "press": press}

@app.get("/alerts")
def alerts():
    return _demo["alerts"]

@app.post("/alerts/ack")
def ack(alert_id: str):
    for a in _demo["alerts"]:
        if a["id"] == alert_id:
            a["acknowledged"] = True
            return {"ok": True}
    raise HTTPException(404, "alert id not found")

# ---------------- XAI: Integrated Gradients over last window ----------------
def _minmax_2d(a: np.ndarray):
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9: return np.zeros_like(a)
    return (a - lo) / (hi - lo)

def _integrated_gradients(model, xt, steps=64):
    baseline = torch.zeros_like(xt)
    ig = torch.zeros_like(xt)
    for i in range(1, steps+1):
        alpha = float(i)/steps
        xi = (baseline + alpha*(xt-baseline)).detach().requires_grad_(True)
        yi = model(xi)
        grads = torch.autograd.grad(outputs=yi, inputs=xi, grad_outputs=torch.ones_like(yi), retain_graph=False)[0]
        ig += grads
    ig = (xt - baseline) * ig / steps
    return ig.detach()

@app.get("/explain_global_live")
def explain_global_live():
    if _demo["last_window"] is None:
        raise HTTPException(409, "Demo not started.")
    # rebuild the global model to compute IG on the current window
    W = _demo["W"]; fd = _demo["fd_hint"]; use = _demo["use"]
    g = REGISTRY[use]; scaler = joblib_load(REGISTRY[fd]["scaler"])
    X = np.array(_demo["last_window"], dtype=np.float32)
    Xs = scaler.transform(X); Xg = np.concatenate([Xs, one_hot_block(fd, W)], axis=1)
    xt = torch.tensor(Xg, dtype=torch.float32).unsqueeze(0).permute(0,2,1).clone().requires_grad_(True)
    model = CNNLSTM_CBAM(in_ch=28, cnn_ch=96, lstm_h=96)
    model.load_state_dict(torch.load(g["weights"], map_location="cpu"))
    model.eval()
    y = model(xt)
    ig = _integrated_gradients(model, xt, steps=64)[0].cpu().numpy()  # [28,W]
    ig = np.abs(ig).T  # [W,28]
    feat = ig[:, :24]
    dom  = ig[:, 24:]
    # average across time for simple bar chart
    feat_avg = feat.mean(axis=0).tolist()
    return {
        "pred": float(max(0.0, y.detach().cpu().numpy().ravel()[0])),
        "features": FEATURE_COLS,
        "feat_avg": feat_avg,
        "heatmap": _minmax_2d(feat).tolist(),   # 0..1 heatmap if needed
        "domain": dom.mean(axis=0).tolist()
    }

# ---------------- What-if (very simple) ----------------
class WhatIfReq(BaseModel):
    fd_hint: str
    use: Optional[str] = "GLOBAL50_MIXED"
    # deltas applied to last window means of 3 proxies (temp, vib, press)
    delta_temp: float = 0.0
    delta_vib: float = 0.0
    delta_press: float = 0.0

@app.post("/whatif")
def whatif(req: WhatIfReq):
    if _demo["last_window"] is None:
        raise HTTPException(409, "Demo not started.")
    W = _demo["W"]; use = req.use or "GLOBAL50_MIXED"
    X = np.array(_demo["last_window"], dtype=np.float32).copy()  # [W,24]
    # modify same indices we used for proxies (14,9,3)
    X[:,14] += req.delta_temp
    X[:, 9] += req.delta_vib
    X[:, 3] += req.delta_press
    new_pred = _predict_global(req.fd_hint, X, use)
    return {"pred_after": new_pred, "fd_hint": req.fd_hint, "use": use}

# ---------------- Tiny planner stubs (for completeness) ----------------
@app.get("/recommendations")
def recommendations():
    p = _demo["last_pred"] or 50.0
    items = []
    if p < 40: items.append({"title":"Inspect Fan module", "due_cycles": max(5, int(p-10)), "evidence":"RUL < 40; rising vibration."})
    if p < 20: items.append({"title":"Plan AOG spare", "due_cycles": max(2, int(p-5)), "evidence":"RUL < 20 (critical)."})
    return {"items": items}

@app.post("/workorders")
def workorders(title: str, due_cycles: int):
    return {"ok": True, "id": f"WO-{int(time.time())}", "title": title, "due_cycles": due_cycles}

# ---------------- Simple metrics ----------------
@app.get("/metrics")
def metrics():
    return {
        "model": _demo["use"],
        "last_latency_ms": 5,  # local (mock)
        "rul_trace_len": len(_demo["pred_trace"]),
        "alerts": len(_demo["alerts"]),
    }
