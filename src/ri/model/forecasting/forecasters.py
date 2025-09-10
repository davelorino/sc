from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from xgboost import XGBRegressor

def _make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["year"] = d["ds"].dt.year
    d["woy"]  = d["ds"].dt.isocalendar().week.astype(int)
    d["dow"]  = d["ds"].dt.dayofweek
    d["month_sin"] = np.sin(2*np.pi*d["ds"].dt.month/12)
    d["month_cos"] = np.cos(2*np.pi*d["ds"].dt.month/12)
    d["woy_sin"]   = np.sin(2*np.pi*d["woy"]/52)
    d["woy_cos"]   = np.cos(2*np.pi*d["woy"]/52)
    return d

def fit_predict_series(
    hist_df: pd.DataFrame,  # columns: ds, y, [reg...]
    forecast_ds: List[pd.Timestamp],
    reg_cols: Optional[List[str]] = None,
    params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Train on all ds < min(forecast_ds), predict for forecast_ds (one-step or many).
    Uses XGB with time features + lags/rolls. Exogenous regressors are allowed.
    """
    if reg_cols is None: reg_cols = []
    if params is None:
        params = dict(n_estimators=500, learning_rate=0.05, max_depth=4,
                      subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                      objective="reg:squarederror", tree_method="hist",
                      n_jobs=4, random_state=42)

    df = hist_df.copy().sort_values("ds")
    df["ds"] = pd.to_datetime(df["ds"])
    cutoff = min(forecast_ds)

    tr = df[df["ds"] < cutoff].copy()
    te = pd.DataFrame({"ds": sorted(set(map(pd.Timestamp, forecast_ds)))})
    if len(tr) < 8:
        return te.assign(yhat=np.nan)

    tr = _make_time_features(tr)
    te = _make_time_features(te)

    # lags & rolls
    for L in (1,2,4): tr[f"lag_{L}"] = tr["y"].shift(L)
    for W in (4,8):   tr[f"roll_mean_{W}"] = tr["y"].shift(1).rolling(W, min_periods=1).mean()

    # design
    feat_cols = [c for c in tr.columns if c not in ("ds","y")]
    # Append exogenous regressors if present
    for c in reg_cols:
        if c not in feat_cols and c in df.columns:
            feat_cols.append(c)

    # Merge regressors
    if reg_cols:
        tr = tr.merge(df[["ds"]+reg_cols], on="ds", how="left")
        te = te.merge(df[["ds"]+reg_cols], on="ds", how="left").fillna(0.0)

    Xtr = tr[feat_cols].fillna(0.0); ytr = tr["y"]
    model = XGBRegressor(**params).fit(Xtr, ytr)

    # recursive preds not necessary if we only need given dates; still fine to predict directly
    Xte = te[feat_cols].fillna(0.0)
    te["yhat"] = model.predict(Xte)
    return te[["ds","yhat"]]
