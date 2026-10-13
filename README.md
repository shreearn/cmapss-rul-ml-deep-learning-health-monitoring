# CMAPSS Remaining Useful Life (RUL) Prediction

Deep learning pipeline for predicting the Remaining Useful Life of turbofan jet 
engines from multivariate sensor telemetry, using NASA's C-MAPSS degradation 
simulation dataset (FD001–FD004).

## Problem

In predictive maintenance, knowing *how many operating cycles are left* before 
a machine needs servicing is more valuable than just detecting that something 
is wrong. This project frames RUL estimation as a sequence regression problem: 
given a rolling window of an engine's sensor readings, predict the number of 
cycles remaining before failure.

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) — four 
sub-datasets (FD001–FD004) covering different combinations of operating 
conditions and fault modes, each with multiple engines' full run-to-failure 
sensor histories (21 sensors + 3 operational settings per cycle).

## Approach

**1. Tabular baselines.** LightGBM and CatBoost models on hand-engineered, 
windowed sensor features — used as a reference point and explained with SHAP 
to understand which sensors and operating conditions drive degradation.

**2. Deep sequence model.** A custom CNN-LSTM architecture with a CBAM 
(Convolutional Block Attention Module) block, built in PyTorch:

- 1D convolutions extract local temporal patterns across the sensor window

- CBAM applies channel attention (which sensors matter most) and spatial 
  attention (which time steps matter most) before the signal reaches the LSTM

- An LSTM layer models longer-range temporal dependencies across the window

- A regression head outputs the predicted RUL

**3. Global model + synthetic augmentation.** Rather than training a separate 
model per sub-dataset, FD001–FD004 are unified into one global training set 
(`05_prepare_global.py`, `05b_build_mixed_global_w50.py`), with synthetic 
window augmentation to balance degradation patterns across operating regimes. 
This took the global model's RMSE from 15.1 → 5.2 cycles (~66% reduction) 
over the non-augmented baseline.

## Explainability

- **Tabular models:** SHAP values to attribute predictions to individual 
  sensor features.

- **Deep model:** gradient-based attribution (saliency maps via PyTorch 
  autograd, and Integrated Gradients) to identify which time steps and 
  sensors in a given input window most influenced the RUL prediction — 
  since SHAP doesn't scale cleanly to sequence models with attention.

## Repo structure

| File | Purpose |

|---|---|

| `src/01_prepare_cmapss.py` | Load and window the raw C-MAPSS data |

| `src/02_train_tabular.py` | LightGBM / CatBoost baselines + SHAP |

| `src/03_train_cnn_lstm_cbam.py` | Train the CNN-LSTM-CBAM model per sub-dataset |

| `src/models_seq.py` | Model definitions (CBAM1D, ChannelAttention1D, SpatialAttention1D, CNNLSTM_CBAM) |

| `src/04_xai_seq_explain.py` | Gradient-based explainability for the sequence model |

| `src/05_prepare_global.py`, `05b_build_mixed_global_w50.py` | Build the unified, augmented global dataset |

| `src/06_train_global.py`, `06b_train_global_mixed.py` | Train the global model |

| `src/07_xai_global.py` | Explainability for the global model |

| `src/api.py`, `src/server.py`, `src/demo_endpoints.py` | Local serving experiments used during development |

| `docs/` | Sample API payloads and metrics from development |

## Tech stack

Python, PyTorch, LightGBM, CatBoost, SHAP, pandas, NumPy



## Also part of this project

The trained global model is served live through a FastAPI backend 
(deployed on Render) exposing endpoints for real-time telemetry, RUL 
prediction, alerting, and explainability. A React dashboard (built with 
Lovable, backed by Supabase) consumes these endpoints to give a 
fleet-level view: per-engine health and RUL, historical trends, a 
maintenance planner, model registry status, and prediction explainability 
— all maintained as separate private repos.
