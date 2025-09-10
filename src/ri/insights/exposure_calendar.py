from __future__ import annotations
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional
from src.ri.utils.dates import week_range_wed
from src.ri.get_data.media.parse_media_table import to_list_flexible

def _type_weights(media_type: str, weights: Optional[Dict[str,float]]) -> float:
    if not weights:
        return 1.0
    return float(weights.get(str(media_type).lower(), weights.get("_default_", 1.0)))

def _infer_brand_category_from_skus(
    sorted_sku_list_str: str,
    product_dim: pd.DataFrame
) -> List[tuple[str,str]]:
    skus = to_list_flexible(sorted_sku_list_str)
    if not skus:
        return []
    m = product_dim[product_dim["product_id"].astype(str).isin([str(s) for s in skus])]
    if m.empty:
        return []
    pairs = m[["brand","category"]].dropna().drop_duplicates()
    return list(pairs.itertuples(index=False, name=None))

def build_media_exposure_calendar_detailed(
    *,
    media_df_one_campaign: pd.DataFrame,
    product_dim: pd.DataFrame,         # ['product_id','brand','category','subcategory']
    fleet_size: int,
    media_type_weights: Optional[Dict[str,float]] = None
) -> pd.DataFrame:
    """
    For a single campaign, return *weekly* exposure features at brand×category grain:

    Columns:
      booking_number, week_start, brand, category,
      n_assets_cov, n_types, running_carto_media,
      has_type::<type>, dose_type::<type>,
      exp_weight_brand_cat, coverage_frac

    Notes:
      - If sorted_store_list is empty => coverage_frac = 1.0 for that asset
      - Else coverage_frac = (#stores in list) / fleet_size
      - exp_weight_brand_cat sums type weights × coverage_frac across assets
    """
    rows: List[Dict] = []
    if media_df_one_campaign.empty:
        return pd.DataFrame(columns=[
            "booking_number","week_start","brand","category","n_assets_cov","n_types",
            "running_carto_media","coverage_frac","exp_weight_brand_cat"
        ])

    for _, r in media_df_one_campaign.iterrows():
        booking = r["booking_number"]
        types = [t.lower() for t in to_list_flexible(r.get("media_type_array"))]
        stores = to_list_flexible(r.get("sorted_store_list"))
        cov = 1.0 if len(stores) == 0 else min(1.0, len(stores) / float(max(1, fleet_size)))
        pairs = _infer_brand_category_from_skus(r.get("sorted_sku_list"), product_dim)
        wks = week_range_wed(r["media_start_date"], r["media_end_date"])
        if not types:
            # treat as one generic asset so counts still rise
            types = ["_unspecified_"]

        type_set = sorted(set(types))
        type_weight_sum = float(sum(_type_weights(mt, media_type_weights) for mt in type_set))

        for wk in wks:
            base = {
                "booking_number": booking,
                "week_start": wk,
                "n_assets_cov": 1.0 * cov,
                "n_types": float(len(type_set)),
                "running_carto_media": 1.0,     # media active this week
                "coverage_frac": float(cov),
            }
            for mt in type_set:
                base[f"has_type::{mt}"] = 1.0
                base[f"dose_type::{mt}"] = cov * _type_weights(mt, media_type_weights)

            if pairs:
                # attribute this asset to each brand×category it promotes
                for (b, c) in pairs:
                    rows.append({**base, "brand": b, "category": c,
                                 "exp_weight_brand_cat": type_weight_sum * cov})
            else:
                rows.append({**base, "brand": None, "category": None,
                             "exp_weight_brand_cat": type_weight_sum * cov})

    cal = pd.DataFrame(rows)
    if cal.empty:
        return cal

    # Aggregate if multiple assets overlap in same week/brand/category
    group_cols = ["booking_number","week_start","brand","category"]
    sum_cols = ["n_assets_cov","exp_weight_brand_cat","coverage_frac"]
    max_cols = ["running_carto_media"] + [c for c in cal.columns if c.startswith("has_type::")]
    dose_cols= [c for c in cal.columns if c.startswith("dose_type::")]

    agg_map = {**{c:"sum" for c in sum_cols + dose_cols},
               **{c:"max" for c in max_cols},
               **{"n_types":"sum"}}
    out = (cal.groupby(group_cols, as_index=False)
              .agg(agg_map)
              .sort_values(group_cols)
              .reset_index(drop=True))
    # Clip coverage to [0,1]
    out["coverage_frac"] = out["coverage_frac"].clip(upper=1.0)
    # Ensure running_carto_media exists for weeks without assets (fill later if you outer-join)
    out["running_carto_media"] = out["running_carto_media"].fillna(0.0)
    return out
