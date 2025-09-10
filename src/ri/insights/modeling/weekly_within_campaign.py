from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

def fit_weekly_within_campaign_model(
    weekly_df: pd.DataFrame,
    alpha: float = 10.0,
    min_weeks_per_campaign: int = 2
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Ridge with campaign fixed effects via within-campaign demeaning.
    Features:
      n_assets_cov, n_types, dose_type::<type>..., disc_depth_bc (exogenous),
      sov, esov, is_leader, running_carto_media (indicator), seasonality (woy sin/cos).
    """
    df = weekly_df.copy()
    df = df.groupby("campaign_id").filter(lambda g: g["week_start"].nunique() >= min_weeks_per_campaign)
    if df.empty:
        raise ValueError("No campaigns with enough weekly variation.")

    df["woy"] = pd.to_datetime(df["week_start"]).dt.isocalendar().week.astype(int)
    df["woy_sin"] = np.sin(2*np.pi*df["woy"]/52.0)
    df["woy_cos"] = np.cos(2*np.pi*df["woy"]/52.0)

    dose_cols = [c for c in df.columns if c.startswith("dose_type::")]
    base_cols = ["n_assets_cov","n_types","disc_depth_bc","sov","esov","is_leader",
                 "running_carto_media","woy_sin","woy_cos"]
    Xcols = base_cols + dose_cols

    # Within-campaign demeaning
    Xc = []; yc = []
    for cid, g in df.groupby("campaign_id"):
        Xg = g[Xcols].fillna(0.0).values
        yg = g["uplift"].values.astype(float)
        Xc.append(Xg - Xg.mean(axis=0, keepdims=True))
        yc.append(yg - yg.mean())
    Xc = np.vstack(Xc); yc = np.concatenate(yc)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=alpha, fit_intercept=False, random_state=42))
    ])
    pipe.fit(Xc, yc)
    coef = pipe.named_steps["ridge"].coef_
    coef_df = pd.DataFrame({"feature": Xcols, "coef": coef}).sort_values("coef", ascending=False)
    return pipe, coef_df
