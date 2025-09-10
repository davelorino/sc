from __future__ import annotations
import pandas as pd
from typing import Dict, List, Tuple
from src.ri.get_data.media.build_weekly_store_calendar import build_weekly_store_calendar_for_campaign, get_fleet_stores, weeks_to_store_sets
from src.ri.model.structures import NodeSpec
from src.ri.model.forecasting.cohort_forecasting import forecast_per_week_dynamic_cohorts
from src.ri.model.reconciliation.non_overshoot import enforce_non_overshoot_grid

def _make_registry(promo_skus: List[str], dim_df: pd.DataFrame) -> Dict[str, List[NodeSpec]]:
    # Create NodeSpec lists for sku, brand_category, brand_subcategory, brand, category, subcategory
    from ri.model.structures import NodeSpec
    d = dim_df[dim_df["product_id"].astype(str).isin(set(map(str, promo_skus)))]
    brands = sorted(d["brand"].dropna().unique())
    cats   = sorted(d["category"].dropna().unique())
    subs   = sorted(d["subcategory"].dropna().unique())
    brand_cats    = sorted(d[["brand","category"]].dropna().drop_duplicates().itertuples(index=False, name=None))
    brand_subcats = sorted(d[["brand","subcategory"]].dropna().drop_duplicates().itertuples(index=False, name=None))
    return {
        "sku":               [NodeSpec("sku",(s,)) for s in sorted(set(map(str, promo_skus)))],
        "brand_category":    [NodeSpec("brand_category", t) for t in brand_cats],
        "brand_subcategory": [NodeSpec("brand_subcategory", t) for t in brand_subcats],
        "brand":             [NodeSpec("brand",(b,)) for b in brands],
        "category":          [NodeSpec("category",(c,)) for c in cats],
        "subcategory":       [NodeSpec("subcategory",(s,)) for s in subs],
    }

def _compute_uplift_totals(cmp_df: pd.DataFrame) -> Dict[str,float]:
    d = {}
    df = cmp_df.copy()
    d["promo_sku_total"]        = df[df["level"]=="sku"]["uplift"].sum()
    d["brand_category_total"]   = df[df["level"]=="brand_category"]["uplift"].sum()
    d["brand_subcategory_total"]= df[df["level"]=="brand_subcategory"]["uplift"].sum()
    d["brand_total"]            = df[df["level"]=="brand"]["uplift"].sum()
    d["category_total"]         = df[df["level"]=="category"]["uplift"].sum()
    d["subcategory_total"]      = df[df["level"]=="subcategory"]["uplift"].sum()
    return d

def _build_grid_dataframe(campaign_id: str, U_sales: Dict[str,float]) -> pd.DataFrame:
    return pd.DataFrame([{
        "campaign_id": campaign_id,
        "Metric": "Incremental Sales / Dollars",
        "Campaign SKUs":       U_sales.get("promo_sku_total"),
        "Brand- Category":     U_sales.get("brand_category_total"),
        "Brand- Subcat":       U_sales.get("brand_subcategory_total"),
        "Whole Brand":         U_sales.get("brand_total"),
        "Whole Category":      U_sales.get("category_total"),
        "Whole Sub- category": U_sales.get("subcategory_total"),
    }])

def build_grids_for_campaigns_v3(
    *,
    media_master_df: pd.DataFrame,
    tx_master_df: pd.DataFrame,   # must include: week_start, store_id, product_number, article_id, brand, category, subcategory, sales, discount/brochure/multibuy, competitor regs
    booking_numbers: List[str]
) -> tuple[pd.DataFrame, Dict[str,pd.DataFrame]]:
    """
    Dynamic-cohort counterfactuals: for each campaign *week*, forecast for that week's treated store subset.
    """
    grids = []
    artifacts: Dict[str,pd.DataFrame] = {}
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
        media_end   = pd.to_datetime(m["media_end_date"]).max()
        train_end   = media_start - pd.Timedelta(days=1)

        # SKUs union
        from ri.get_data.media.parse_media_table import to_list_flexible
        promo_skus = sorted(set(s for row in m["sorted_sku_list"].dropna() for s in to_list_flexible(row)))
        if not promo_skus:
            continue

        registry = _make_registry(promo_skus, dim)
        nodes = registry["sku"] + registry["brand_category"] + registry["brand_subcategory"] + registry["brand"] + registry["category"] + registry["subcategory"]

        # weekly treated-store calendar
        cal = build_weekly_store_calendar_for_campaign(m, fleet)
        if cal.empty:
            continue
        wk_to_stores = weeks_to_store_sets(cal)
        # restrict to campaign window weeks
        wk_to_stores = {wk: s for wk, s in wk_to_stores.items() if (wk >= cal["week_start"].min()) and (wk <= cal["week_start"].max())}

        # Forecast per-week dynamic cohorts (sales only here; extend similarly for shoppers/new-to-brand)
        cmp_sales = forecast_per_week_dynamic_cohorts(
            tx_master_df=tx_master_df,
            week_to_storelist=wk_to_stores,
            nodes=nodes,
            train_end_date=train_end,
            target="sales",
            exog_regressors=[
                "discountpercent",
                "max_internal_competitor_discount_percent",
                "n_competitors","n_any_cheaper","n_shelf_cheaper",
                "n_promo_cheaper_no_hurdle","n_promo_cheaper_hurdle",
                "avg_cheaper_gap","worst_gap","p90_gap",
                "brochure_Not on brochure","multibuy_Not on Multibuy"  # created via dummies
            ]
        )

        if cmp_sales.empty:
            continue

        # Uplift = y - yhat for campaign weeks
        cmp_sales["uplift"] = cmp_sales["y"] - cmp_sales["yhat"]
        U_raw = _compute_uplift_totals(cmp_sales)
        U_adj = enforce_non_overshoot_grid(U_raw)
        grid = _build_grid_dataframe(cid, U_adj)
        grids.append(grid)
        artifacts[cid] = cmp_sales

    grids_df = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(
        columns=["campaign_id","Metric","Campaign SKUs","Brand- Category","Brand- Subcat","Whole Brand","Whole Category","Whole Sub- category"]
    )
    return grids_df, artifacts
