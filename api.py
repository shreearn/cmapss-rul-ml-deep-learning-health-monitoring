# src/api.py
from __future__ import annotations

import json, os
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load as joblib_load

# ---- constants your demo expects ----
FEATURE_COLS = ["setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
FDS_ORDER = ["FD001", "FD002", "FD003", "FD004"]

# ---- paths ----
ROOT = Path(__file__).resolve().parent.parent  # project root
REG_PATH = ROOT / "serving" / "model_registry.json"
if not REG_PATH.exists():
    raise RuntimeError(f"model_registry.json not found at {REG_PATH}")
REGISTRY = json.loads(REG_PATH.read_text(encoding="utf-8"))

# ---- helper ----
def one_hot_block(fd_hint: str, W: int) -> np.ndarray:
    dom = np.zeros((W, len(FDS_ORDER)), dtype=np.float32)
    dom[:, FDS_ORDER.index(fd_hint)] = 1.0
    return dom

# import your demo’s utilities
from .demo_endpoints import attach_demo_routes, CNNLSTM_CBAM  # type: ignore

# internal predictor used by endpoints
def _predict_global_internal(fd_hint: str, window_data, use_key: str) -> float:
    g = REGISTRY[use_key]  # e.g. "GLOBAL50_MIXED"
    W = int(g["window"])

    # resolve paths (allow relative paths in registry)
    weights_path = Path(g["weights"])
    if not weights_path.is_absolute():
        weights_path = ROOT / weights_path

    scaler_path = Path(REGISTRY[fd_hint]["scaler"])
    if not scaler_path.is_absolute():
        scaler_path = ROOT / scaler_path

    if not scaler_path.exists():
        raise RuntimeError(f"Missing scaler for {fd_hint}: {scaler_path}")
    scaler = joblib_load(scaler_path)

    X = np.asarray(window_data, dtype=np.float32)
    if X.shape != (W, len(FEATURE_COLS)):
        raise RuntimeError(f"window_data must be [{W}, {len(FEATURE_COLS)}], got {X.shape}")

    Xs = scaler.transform(X)                                # [W,24]
    Xg = np.concatenate([Xs, one_hot_block(fd_hint, W)], 1) # [W,28]
    xt = torch.tensor(Xg, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)  # [1,28,W]

    model = CNNLSTM_CBAM(in_ch=Xg.shape[1], cnn_ch=96, lstm_h=96, dropout=0.1)
    if not weights_path.exists():
        raise RuntimeError(f"Missing weights: {weights_path}")
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred = float(model(xt).cpu().numpy().ravel()[0])
    return max(0.0, pred)

# ---- app ----
app = FastAPI(title="CMAPSS RUL API", version="0.1.0")

ALLOWED_ORIGINS = [
    "http://localhost:8080", "http://127.0.0.1:8080",  # your Vite dev server
    "http://localhost:5173", "http://127.0.0.1:5173",  # alternative Vite port
]
# ---- FastAPI app ----
app = FastAPI(title="CMAPSS RUL API")

# explicitly list the frontend origins you use
ALLOWED_ORIGINS = [
    "http://localhost:8080","http://localhost:8081",
    "http://127.0.0.1:8080","http://127.0.0.1:8081",
    "http://192.168.0.106:8080","http://192.168.0.106:8081",
]

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/health")
def health():
    return {"ok": True, "fds": list(REGISTRY.keys())}

# simple RUL endpoint the frontend can call
from pydantic import BaseModel
class RULPayload(BaseModel):
    fd: str                   # e.g. "FD001"
    window: List[List[float]] # shape [W, 24]
    use_key: str = "GLOBAL50_MIXED"  # must exist in model_registry.json

@app.post("/predict/rul")
def predict_rul(p: RULPayload):
    try:
        y = _predict_global_internal(p.fd, p.window, p.use_key)
        return {"rul": y}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# attach your demo routes (alerts, explain, etc.) if you need them
attach_demo_routes(app, REGISTRY, FEATURE_COLS, _predict_global_internal)
