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
    hist_df: pd.DataFrame,
    forecast_ds: List[pd.Timestamp],
    reg_cols: Optional[List[str]],
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Trains a simple regressor on (ds,y,+exog) and predicts yhat for the given ds list.
    Only uses feature columns that actually exist in hist_df.
    Falls back to an intercept-only model if no features are available.
    """
    df = hist_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    feat_cols = [c for c in (reg_cols or []) if c in df.columns]

    fc_dates = [pd.to_datetime(d) for d in (forecast_ds or [])]
    min_fc = min(fc_dates) if fc_dates else None

    if min_fc is None:
        tr = df[["ds","y"]].copy()
        te = pd.DataFrame(columns=["ds"])
    else:
        tr = df[df["ds"] < min_fc][["ds","y"]].copy()
        te = pd.DataFrame({"ds": fc_dates})

    if feat_cols:
        tr = tr.merge(df[["ds"] + feat_cols], on="ds", how="left")
        te = te.merge(df[["ds"] + feat_cols], on="ds", how="left")
    else:
        tr["_bias"] = 1.0
        te["_bias"] = 1.0
        feat_cols = ["_bias"]

    tr = tr.fillna(0.0)
    te = te.fillna(0.0)

    model = XGBRegressor(**(params or {})).fit(tr[feat_cols], tr["y"])

    if te.empty:
        return pd.DataFrame(columns=["ds","yhat"])

    out = te[["ds"]].copy()
    out["yhat"] = model.predict(te[feat_cols])
    return out