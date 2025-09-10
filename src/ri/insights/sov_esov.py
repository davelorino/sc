from __future__ import annotations
import numpy as np
import pandas as pd

def compute_sov_esov(
    *,
    exposure_calendar_bc: pd.DataFrame,   # from build_media_exposure_calendar_detailed
    preperiod_share_bc: pd.DataFrame      # from compute_preperiod_market_share
) -> pd.DataFrame:
    """
    Compute SoV per (week, category) as brand_exp / total_exp in that category-week.
    Join preperiod market_share and form ESOV = SOV - market_share.
    """
    cal = exposure_calendar_bc.copy()
    denom = (cal.groupby(["week_start","category"], as_index=False)["exp_weight_brand_cat"].sum()
               .rename(columns={"exp_weight_brand_cat":"cat_week_exp"}))
    cal = cal.merge(denom, on=["week_start","category"], how="left")
    cal["sov"] = np.where(cal["cat_week_exp"] > 0,
                          cal["exp_weight_brand_cat"] / cal["cat_week_exp"], 0.0)

    ms = preperiod_share_bc.rename(columns={"brand":"brand","category":"category"})
    cal = cal.merge(ms[["brand","category","market_share","is_leader"]], on=["brand","category"], how="left")
    cal["market_share"] = cal["market_share"].fillna(0.0)
    cal["is_leader"]    = cal["is_leader"].fillna(0).astype(int)
    cal["esov"]         = cal["sov"] - cal["market_share"]
    return cal
