from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.linear_model import LinearRegression, Ridge

def fit_campaign_level_model(
    weekly_df: pd.DataFrame,
    alpha: float | None = None
) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate to campaign totals and fit OLS (alpha=None) or Ridge (alpha>0).
    Features:
      duration_weeks, n_assets_total, n_types_avg, esov_avg, disc_depth_avg, is_leader (max), avg doses per type.
    """
    df = weekly_df.copy()
    # campaign duration
    dur = (df.groupby("campaign_id", as_index=False)
             .agg(duration_weeks=("week_start", lambda s: s.nunique())))

    dose_cols = [c for c in df.columns if c.startswith("dose_type::")]
    agg = (df.groupby("campaign_id", as_index=False)
             .agg(uplift_total=("uplift","sum"),
                  n_assets_total=("n_assets_cov","sum"),
                  n_types_avg=("n_types","mean"),
                  esov_avg=("esov","mean"),
                  disc_depth_avg=("disc_depth_bc","mean"),
                  is_leader=("is_leader","max"),
                  **{f"{c}_avg": (c,"mean") for c in dose_cols}))
    agg = agg.merge(dur, on="campaign_id", how="left")

    Xcols = ["duration_weeks","n_assets_total","n_types_avg","esov_avg","disc_depth_avg","is_leader"] + [f"{c}_avg" for c in dose_cols]
    X = agg[Xcols].fillna(0.0).values
    y = agg["uplift_total"].values

    if alpha is None:
        model = LinearRegression().fit(X, y)
        coef = model.coef_
    else:
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=42).fit(X, y)
        coef = model.coef_

    coef_df = pd.DataFrame({"feature": Xcols, "coef": coef}).sort_values("coef", ascending=False)
    return model, coef_df, agg
