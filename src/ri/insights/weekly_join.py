from __future__ import annotations
import pandas as pd

def prepare_weekly_hypothesis_table_bc(
    *,
    cmp_sales_one_campaign: pd.DataFrame,    # columns: ds,y,yhat,group,level
    exposure_calendar_bc: pd.DataFrame,      # from build_media_exposure_calendar_detailed + SoV/ESOV
    discount_depth_bc: pd.DataFrame          # week_start, brand, category, disc_depth_bc
) -> pd.DataFrame:
    """
    Build weekly modeling table at brand×category grain:
      uplift, n_assets_cov, n_types, doses per type, SoV, ESOV, is_leader, discount depth, running_carto_media.
    """
    cmp = cmp_sales_one_campaign.copy()
    cmp = cmp[cmp["level"] == "brand_category"].copy()
    if cmp.empty:
        return pd.DataFrame()

    # Parse keys from group "brand=...|cat=..."
    def _parse(g):
        brand = None; cat = None
        parts = str(g).split("|")
        for p in parts:
            if p.startswith("brand="): brand = p.split("=",1)[1]
            if p.startswith("cat="):   cat   = p.split("=",1)[1]
        return pd.Series({"brand": brand, "category": cat})

    keys = cmp["group"].apply(_parse)
    cmp = pd.concat([cmp, keys], axis=1)
    cmp = cmp.rename(columns={"ds":"week_start"})
    cmp["uplift"] = cmp["y"] - cmp["yhat"]

    # Merge exposure (includes running_carto_media, types, SoV/ESOV)
    x = cmp.merge(exposure_calendar_bc, on=["week_start","brand","category"], how="left")

    # Merge discount depth
    x = x.merge(discount_depth_bc, on=["week_start","brand","category"], how="left")
    x["disc_depth_bc"] = x["disc_depth_bc"].fillna(0.0)

    # Fill exposure zeros where missing
    for c in x.columns:
        if c.startswith("has_type::") or c.startswith("dose_type::") or c in (
            "n_assets_cov","n_types","running_carto_media","sov","esov","coverage_frac","exp_weight_brand_cat","is_leader"
        ):
            x[c] = x[c].fillna(0.0)

    return x
