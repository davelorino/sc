from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.ri.insights.exposure_calendar import build_media_exposure_calendar_detailed
from src.ri.insights.market_share import compute_preperiod_market_share
from src.ri.utils.dates import week_wed

def exposure_by_grain(
    *,
    media_one_campaign: pd.DataFrame,
    tx_master_df: pd.DataFrame,
    product_dim: pd.DataFrame,        # ['product_id','brand','category','subcategory']
    fleet_size: int,
    grain: str,                       # 'sku'|'brand'|'brand_category'|'brand_subcategory'|'category'|'subcategory'
    promo_skus: List[str],
    train_end_date,
    media_type_weights: Optional[Dict[str,float]] = None
) -> pd.DataFrame:
    """
    Returns weekly exposure features for the requested grain with columns superset:
      ['week_start', keys..., 'running_carto_media','n_assets_cov','n_types',
       dose_type::*, 'sov','esov','is_leader','coverage_frac', 'exp_weight_brand_cat']
    Notes:
      - SoV/ESOV is native at brand_category; at other grains it is a weighted transform.
      - is_leader comes from brand's pre-period share in its category.
    """
    # Base: brand_category
    bc = build_media_exposure_calendar_detailed(
        media_df_one_campaign=media_one_campaign,
        product_dim=product_dim,
        fleet_size=fleet_size,
        media_type_weights=media_type_weights
    )  # columns: booking_number, week_start, brand, category, n_assets_cov, n_types, running_carto_media, coverage_frac, dose_type::*, exp_weight_brand_cat

    # Preperiod share for leader/challenger at brand×category
    cohort_stores = sorted(tx_master_df["store_id"].astype(str).unique().tolist())  # or pass weekly stores if you prefer
    pre = compute_preperiod_market_share(
        tx_master_df=tx_master_df, stores=cohort_stores, train_end_date=train_end_date, lookback_weeks=13
    )
    # SoV/ESOV at bc
    from src.ri.insights.sov_esov import compute_sov_esov
    bc = compute_sov_esov(exposure_calendar_bc=bc, preperiod_share_bc=pre)

    # Project to other grains
    dose_cols = [c for c in bc.columns if c.startswith("dose_type::")]
    keep_base = ["week_start","running_carto_media","n_assets_cov","n_types","coverage_frac","sov","esov","is_leader","exp_weight_brand_cat"] + dose_cols

    if grain == "brand_category":
        return bc[["brand","category"] + keep_base].copy()

    if grain == "brand_subcategory":
        # Map bc to bsc using product_dim (brand+category -> subcats promoted by SKUs)
        sku_pairs = product_dim[product_dim["product_id"].astype(str).isin(promo_skus)][["brand","category","subcategory"]].dropna().drop_duplicates()
        bsc = (bc.merge(sku_pairs.drop_duplicates(), on=["brand","category"], how="left")
                 .dropna(subset=["subcategory"]))
        # Within each subcategory-week, recompute SoV using exp weights aggregated by brand-subcat
        denom = (bsc.groupby(["week_start","subcategory"], as_index=False)["exp_weight_brand_cat"].sum()
                   .rename(columns={"exp_weight_brand_cat":"subcat_week_exp"}))
        bsc = bsc.merge(denom, on=["week_start","subcategory"], how="left")
        bsc["sov"] = np.where(bsc["subcat_week_exp"]>0, bsc["exp_weight_brand_cat"]/bsc["subcat_week_exp"], 0.0)
        # ESOV still uses brand's market share in its category (closest definition)
        return bsc[["brand","subcategory"] + keep_base].rename(columns={"subcategory":"subcat"}).copy()

    if grain == "brand":
        g = (bc.groupby(["week_start","brand"], as_index=False)
               .agg(n_assets_cov=("n_assets_cov","sum"),
                    n_types=("n_types","sum"),
                    running_carto_media=("running_carto_media","max"),
                    coverage_frac=("coverage_frac","max"),
                    exp_weight_brand_cat=("exp_weight_brand_cat","sum"),
                    **{c: (c,"sum") for c in dose_cols},
                    sov=("sov", lambda s: float(np.average(s, weights=None))),   # simple mean across cats
                    esov=("esov", lambda s: float(np.average(s, weights=None))),
                    is_leader=("is_leader","max")))
        return g[["brand"] + keep_base].copy()

    if grain == "category":
        # Sum exposures of the campaign's promoted brands per category-week
        # SoV here = sum of promoted brand SoVs within that category-week (bounded by 1)
        bc2 = bc.copy()
        g = (bc2.groupby(["week_start","category"], as_index=False)
                .agg(n_assets_cov=("n_assets_cov","sum"),
                     n_types=("n_types","sum"),
                     running_carto_media=("running_carto_media","max"),
                     coverage_frac=("coverage_frac","max"),
                     exp_weight_brand_cat=("exp_weight_brand_cat","sum"),
                     **{c: (c,"sum") for c in dose_cols},
                     sov=("sov","sum"),
                     esov=("esov","sum"),
                     is_leader=("is_leader","max")))
        g["sov"]  = g["sov"].clip(upper=1.0)
        return g[["category"] + keep_base].copy()

    if grain == "subcategory":
        # Build from brand_subcategory projection then aggregate brands
        bsc = exposure_by_grain(media_one_campaign=media_one_campaign, tx_master_df=tx_master_df,
                                product_dim=product_dim, fleet_size=fleet_size, grain="brand_subcategory",
                                promo_skus=promo_skus, train_end_date=train_end_date,
                                media_type_weights=media_type_weights)
        g = (bsc.groupby(["week_start","subcat"], as_index=False)
                .agg(n_assets_cov=("n_assets_cov","sum"),
                     n_types=("n_types","sum"),
                     running_carto_media=("running_carto_media","max"),
                     coverage_frac=("coverage_frac","max"),
                     exp_weight_brand_cat=("exp_weight_brand_cat","sum"),
                     **{c: (c,"sum") for c in dose_cols},
                     sov=("sov","sum"),
                     esov=("esov","sum"),
                     is_leader=("is_leader","max")))
        g["sov"]  = g["sov"].clip(upper=1.0)
        return g.rename(columns={"subcat":"subcategory"})[["subcategory"] + keep_base].copy()

    if grain == "sku":
        # attach brand/category to each sku, then join bc features
        pdim = product_dim[product_dim["product_id"].astype(str).isin(promo_skus)][["product_id","brand","category"]].dropna()
        bc2 = bc.rename(columns={"brand":"brand","category":"category"})
        sku_feat = (pdim.merge(bc2, on=["brand","category"], how="left"))
        # replicate per sku
        # no fresh SoV math; inherits brand-category SoV as driver
        keep = ["product_id","week_start"] + [c for c in bc2.columns if c not in ("booking_number","brand","category")]
        return sku_feat[keep].rename(columns={"product_id":"sku"}).copy()

    raise ValueError(grain)
