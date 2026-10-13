import os, json
import numpy as np
import pandas as pd
from math import sqrt, exp
from sklearn.metrics import mean_squared_error
from joblib import dump
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
import shap

# ---------------- utils ----------------
def nasa_score(y_true, y_pred):
    s = 0.0
    for yt, yp in zip(y_true, y_pred):
        d = yp - yt
        s += (np.exp(-d/13) - 1) if d < 0 else (np.exp(d/10) - 1)
    return float(s)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ---------------- config ----------------
FD = os.environ.get("FD", "FD001")     # change via env if needed
BASE = "."
PROCESSED = os.path.join(BASE, "data", "processed", FD)
MODELDIR  = os.path.join(BASE, "data", "models")
REPORTS   = os.path.join(BASE, "data", "reports")
SHAPDIR   = os.path.join(REPORTS, "shap", FD)
ensure_dirs(MODELDIR, REPORTS, SHAPDIR)

print(f"[INFO] Using {FD}; loading parquet…")
train = pd.read_parquet(os.path.join(PROCESSED, "train.parquet"))
test  = pd.read_parquet(os.path.join(PROCESSED, "test.parquet"))

feature_cols = ["setting_1","setting_2","setting_3"] + [f"sensor_{i}" for i in range(1,22)]
label_col = "RUL_capped" if "RUL_capped" in train.columns else "RUL"

X_train, y_train = train[feature_cols].values, train[label_col].values
X_test,  y_test  = test[feature_cols].values,  test[label_col].values

# ---------------- LightGBM ----------------
print("[INFO] Training LightGBM…")
lgb = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.02,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb.fit(X_train, y_train)

y_pred_lgb = lgb.predict(X_test)
rmse_lgb = sqrt(mean_squared_error(y_test, y_pred_lgb))
nasa_lgb = nasa_score(y_test, y_pred_lgb)
print(f"[LGBM] RMSE={rmse_lgb:.3f} NASA={nasa_lgb:.1f}")

dump(lgb, os.path.join(MODELDIR, f"{FD}_lgbm.joblib"))
with open(os.path.join(REPORTS, f"{FD}_lgbm_metrics.json"), "w") as f:
    json.dump({"rmse": rmse_lgb, "nasa": nasa_lgb}, f, indent=2)

# SHAP for LightGBM
print("[INFO] SHAP for LightGBM…")
explainer = shap.TreeExplainer(lgb)
# use a small background for speed
bg_idx = np.random.default_rng(42).choice(len(X_train), size=min(2000, len(X_train)), replace=False)
shap_values = explainer.shap_values(X_test[:3000])  # subset for quick plot

# summary bar plot
plt.figure()
shap.summary_plot(shap_values, X_test[:3000], feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig(os.path.join(SHAPDIR, "lgbm_summary.png"), dpi=150)
plt.close()

# waterfall for one engine row
plt.figure()
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value, shap_values[0],
    feature_names=feature_cols, max_display=15, show=False
)
plt.tight_layout()
plt.savefig(os.path.join(SHAPDIR, "lgbm_waterfall_sample.png"), dpi=150)
plt.close()

# ---------------- CatBoost ----------------
print("[INFO] Training CatBoost…")
cat = CatBoostRegressor(
    depth=8, learning_rate=0.03, iterations=1500,
    loss_function="RMSE", random_seed=42, verbose=False
)
cat.fit(Pool(X_train, y_train))

y_pred_cat = cat.predict(X_test)
rmse_cat = sqrt(mean_squared_error(y_test, y_pred_cat))
nasa_cat = nasa_score(y_test, y_pred_cat)
print(f"[CAT]  RMSE={rmse_cat:.3f} NASA={nasa_cat:.1f}")

cat.save_model(os.path.join(MODELDIR, f"{FD}_catboost.cbm"))
with open(os.path.join(REPORTS, f"{FD}_catboost_metrics.json"), "w") as f:
    json.dump({"rmse": rmse_cat, "nasa": nasa_cat}, f, indent=2)

# Save per-row predictions for later analysis
preds = pd.DataFrame({
    "unit": test["unit"],
    "cycle": test["cycle"],
    "y_true": y_test,
    "lgbm_pred": y_pred_lgb,
    "cat_pred": y_pred_cat
})
preds.to_csv(os.path.join(REPORTS, f"{FD}_tabular_preds.csv"), index=False)

print("[DONE] Tabular models and SHAP saved.")
