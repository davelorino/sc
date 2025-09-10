from __future__ import annotations
import pandas as pd
from typing import Dict, List
from src.ri.get_data.media.build_weekly_store_calendar import build_weekly_store_calendar_for_campaign, get_fleet_stores, weeks_to_store_sets
from src.ri.get_data.media.parse_media_table import to_list_flexible
from src.ri.model.structures import NodeSpec
from src.ri.model.forecasting.cohort_forecasting import forecast_per_week_dynamic_cohorts
from src.ri.model.forecasting.aggregate_nodes import build_weekly_series_for_stores  # used internally
# Node registry helper
from src.ri.model.orchestration.build_grids_v3 import _make_registry  # reuse

# Which targets (metrics) to compute
ALL_TARGETS = [
    "sales","shoppers",
    "new_to_sku_sales","new_to_sku_shoppers",
    "new_to_brand_sales","new_to_brand_shoppers",
    "new_to_category_sales","new_to_category_shoppers",
    "new_to_subcategory_sales","new_to_subcategory_shoppers",
]

# Which node groups to use for each report band
NODE_BANDS = {
    "promoted_skus": ["sku"],
    "brand_skus":    ["brand"],
    "brand_category":["brand_category"],
    "brand_subcat":  ["brand_subcategory"],
    "category":      ["category"],
    "subcategory":   ["subcategory"],
}

def run_counterfactuals_full(
    *,
    media_master_df: pd.DataFrame,
    tx_master_df: pd.DataFrame,
    booking_numbers: List[str],
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Returns nested dict:
      cf[campaign_id][metric_key][band] = compare_df with columns [ds,y,yhat,group,level]
    """
    out: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    fleet = get_fleet_stores(tx_master_df)

    # product dimension
    dim = (tx_master_df.rename(columns={"product_number":"product_id",
                                        "BrandDescription":"brand",
                                        "CategoryDescription":"category",
                                        "SubCategoryDescription":"subcategory"})
           [["product_id","brand","category","subcategory"]].drop_duplicates())

    for cid in booking_numbers:
        m = media_master_df[media_master_df["booking_number"]==cid].copy()
        if m.empty: 
            continue
        media_start = pd.to_datetime(m["media_start_date"]).min()
        train_end   = media_start - pd.Timedelta(days=1)

        promo_skus = sorted(set(s for row in m["sorted_sku_list"].dropna() for s in to_list_flexible(row)))
        if not promo_skus: 
            continue

        registry = _make_registry(promo_skus, dim)
        cal = build_weekly_store_calendar_for_campaign(m, fleet)
        if cal.empty: 
            continue
        wk_to_stores = weeks_to_store_sets(cal)

        out.setdefault(cid, {})

        # Loop metrics and node bands
        for target in ALL_TARGETS:
            out[cid].setdefault(target, {})
            for band, levels in NODE_BANDS.items():
                nodes: List[NodeSpec] = sum((registry[level] for level in levels), [])
                if not nodes:
                    continue
                cmp_df = forecast_per_week_dynamic_cohorts(
                    tx_master_df=tx_master_df,
                    week_to_storelist=wk_to_stores,
                    nodes=nodes,
                    train_end_date=train_end,
                    target=target,
                    exog_regressors=[
                        # keep exogenous; discount/brochure/multibuy ARE exogenous per your directive
                        "discountpercent",
                        "max_internal_competitor_discount_percent",
                        "n_competitors","n_any_cheaper","n_shelf_cheaper",
                        "n_promo_cheaper_no_hurdle","n_promo_cheaper_hurdle",
                        "avg_cheaper_gap","worst_gap","p90_gap",
                        "brochure_Not on brochure","multibuy_Not on Multibuy"
                    ]
                )
                if cmp_df.empty:
                    continue
                cmp_df["uplift"] = cmp_df["y"] - cmp_df["yhat"]
                out[cid][target][band] = cmp_df
    return out
