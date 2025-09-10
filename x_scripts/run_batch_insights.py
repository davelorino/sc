#!/usr/bin/env python
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd

from src.ri.get_data.media.build_weekly_store_calendar import build_weekly_store_calendar_for_campaign, get_fleet_stores, weeks_to_store_sets
from src.ri.get_data.media.parse_media_table import to_list_flexible
from src.ri.model.orchestration.build_grids_v3 import build_grids_for_campaigns_v3
from src.ri.insights.exposure_calendar import build_media_exposure_calendar_detailed
from src.ri.insights.market_share import compute_preperiod_market_share
from src.ri.insights.sov_esov import compute_sov_esov
from src.ri.insights.discount_depth import compute_discount_depth
from src.ri.insights.weekly_join import prepare_weekly_hypothesis_table_bc
from src.ri.insights.modeling.weekly_within_campaign import fit_weekly_within_campaign_model
from src.ri.insights.modeling.campaign_level import fit_campaign_level_model
from src.ri.insights.modeling.predict import predict_weekly_uplift, predict_campaign_total

def _product_dim_from_tx(tx_master_df: pd.DataFrame) -> pd.DataFrame:
    return (tx_master_df.rename(columns={
        "product_number":"product_id",
        "BrandDescription":"brand",
        "CategoryDescription":"category",
        "SubCategoryDescription":"subcategory"
    })[["product_id","brand","category","subcategory"]].drop_duplicates())

def build_weekly_insight_table_for_campaign(
    *,
    cid: str,
    media_one: pd.DataFrame,
    tx_master_df: pd.DataFrame,
    cmp_sales_one: pd.DataFrame,
    media_type_weights: dict | None = None
) -> pd.DataFrame:
    """
    Build weekly brand×category insight table for a single campaign.
    """
    product_dim = _product_dim_from_tx(tx_master_df)
    fleet = get_fleet_stores(tx_master_df)

    # cohort stores (union over campaign weeks)
    cal_store = build_weekly_store_calendar_for_campaign(media_one, fleet)
    wk_to_stores = weeks_to_store_sets(cal_store)
    cohort_stores = sorted({sid for s in wk_to_stores.values() for sid in s})

    # preperiod cutoff
    media_start = pd.to_datetime(media_one["media_start_date"]).min()
    train_end = media_start - pd.Timedelta(days=1)

    # preperiod market share and leader flag (brand×category)
    ms = compute_preperiod_market_share(
        tx_master_df=tx_master_df, stores=cohort_stores, train_end_date=train_end, lookback_weeks=13
    )

    # exposure calendar (detailed, brand×category)
    cal_bc = build_media_exposure_calendar_detailed(
        media_df_one_campaign=media_one,
        product_dim=product_dim,
        fleet_size=len(fleet),
        media_type_weights=media_type_weights
    )
    # add SoV & ESOV
    cal_bc = compute_sov_esov(exposure_calendar_bc=cal_bc, preperiod_share_bc=ms)

    # weekly discount depth (brand×category) – exogenous and always modeled
    disc_bc = compute_discount_depth(tx=tx_master_df, stores=cohort_stores)

    # join to weekly uplift from compare (brand×category rows)
    weekly_bc = prepare_weekly_hypothesis_table_bc(
        cmp_sales_one_campaign=cmp_sales_one,
        exposure_calendar_bc=cal_bc,
        discount_depth_bc=disc_bc
    )
    weekly_bc["campaign_id"] = cid
    return weekly_bc

def summarize_soundbites(coef_weekly: pd.DataFrame, coef_campaign: pd.DataFrame) -> list[str]:
    """
    Turn coefficients into a few readable insights.
    """
    out = []
    # Weekly: effects per unit (already standardized in Ridge; we approximate interpretation simply)
    def topk(df, prefix, k=5):
        # Exclude seasonality terms
        filt = df[~df["feature"].isin(["woy_sin","woy_cos"])]
        pos = filt.sort_values("coef", ascending=False).head(k)
        neg = filt.sort_values("coef", ascending=True).head(k)
        return pos, neg

    pos, neg = topk(coef_weekly, "Weekly")
    out.append("WEEKLY DRIVERS (Ridge, within-campaign):")
    for _, r in pos.iterrows():
        out.append(f"  + {r['feature']}: positive effect (coef {r['coef']:.3f})")
    for _, r in neg.iterrows():
        out.append(f"  - {r['feature']}: negative effect (coef {r['coef']:.3f})")

    # Campaign-level totals
    pos2 = coef_campaign.sort_values("coef", ascending=False).head(5)
    neg2 = coef_campaign.sort_values("coef", ascending=True).head(5)
    out.append("CAMPAIGN TOTAL DRIVERS (OLS):")
    for _, r in pos2.iterrows():
        out.append(f"  + {r['feature']}: increases total uplift (coef {r['coef']:.2f})")
    for _, r in neg2.iterrows():
        out.append(f"  - {r['feature']}: decreases total uplift (coef {r['coef']:.2f})")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media_csv", required=True, help="Media campaign table CSV")
    ap.add_argument("--tx_parquet", required=True, help="Transactions master (parquet or CSV)")
    ap.add_argument("--bookings", nargs="+", required=True, help="List of campaign IDs (100+ ok)")
    ap.add_argument("--outdir", default="4_outputs")
    ap.add_argument("--weights_yaml", default=None, help="Optional YAML of media type weights")
    args = ap.parse_args()

    os.makedirs(f"{args.outdir}/reports", exist_ok=True)
    os.makedirs(f"{args.outdir}/artifacts", exist_ok=True)

    media = pd.read_csv(
        args.media_csv,
        parse_dates=["campaign_start_date","campaign_end_date","media_start_date","media_end_date"]
    )
    tx = (pd.read_parquet(args.tx_parquet) if args.tx_parquet.endswith(".parquet")
          else pd.read_csv(args.tx_parquet, parse_dates=["week_start"]))

    # ---- 1) Run dynamic-cohort counterfactuals (sales); reuse existing orchestrator
    from ri.model.orchestration.build_grids_v3 import build_grids_for_campaigns_v3
    grids, cmp_by_campaign = build_grids_for_campaigns_v3(
        media_master_df=media, tx_master_df=tx, booking_numbers=args.bookings
    )
    grids.to_csv(f"{args.outdir}/reports/grids_sales.csv", index=False)

    # ---- 2) Build weekly insight table across campaigns (brand×category)
    product_dim = (tx.rename(columns={"product_number":"product_id",
                                      "BrandDescription":"brand",
                                      "CategoryDescription":"category",
                                      "SubCategoryDescription":"subcategory"})
                     [["product_id","brand","category","subcategory"]].drop_duplicates())
    # Optional weights
    media_type_weights = None
    if args.weights_yaml and os.path.exists(args.weights_yaml):
        import yaml
        with open(args.weights_yaml, "r") as f:
            cfg = yaml.safe_load(f)
            media_type_weights = cfg.get("weights", None)

    weekly_all = []
    for cid in args.bookings:
        m = media[media["booking_number"]==cid].copy()
        cmp_sales = cmp_by_campaign.get(cid)
        if m.empty or cmp_sales is None or cmp_sales.empty:
            continue
        weekly = build_weekly_insight_table_for_campaign(
            cid=cid, media_one=m, tx_master_df=tx, cmp_sales_one=cmp_sales, media_type_weights=media_type_weights
        )
        if weekly.empty: 
            continue
        weekly_all.append(weekly)

    if not weekly_all:
        print("No weekly insight rows; exiting.")
        return

    weekly_df = pd.concat(weekly_all, ignore_index=True)
    weekly_df.to_parquet(f"{args.outdir}/artifacts/weekly_bc.parquet", index=False)

    # ---- 3) Fit models
    weekly_model, coef_weekly = fit_weekly_within_campaign_model(weekly_df, alpha=10.0)
    model, coef_campaign, campaign_agg = fit_campaign_level_model(weekly_df, alpha=None)

    coef_weekly.to_csv(f"{args.outdir}/reports/coef_weekly.csv", index=False)
    coef_campaign.to_csv(f"{args.outdir}/reports/coef_campaign.csv", index=False)
    campaign_agg.to_csv(f"{args.outdir}/reports/campaign_agg.csv", index=False)

    # ---- 4) Soundbites
    bites = summarize_soundbites(coef_weekly, coef_campaign)
    with open(f"{args.outdir}/reports/soundbites.txt","w") as f:
        f.write("\n".join(bites))
    print("\n".join(bites))

    # ---- 5) Interactive examples
    # a) Weekly scenario (e.g., 8 assets across 3 types, moderate ESOV, leader brand)
    from ri.insights.modeling.predict import predict_weekly_uplift, predict_campaign_total

    # Build a column template for predictors
    template_cols = weekly_df[[c for c in weekly_df.columns if c.startswith("dose_type::") or c in
                              ["n_assets_cov","n_types","disc_depth_bc","sov","esov","is_leader",
                               "running_carto_media","woy_sin","woy_cos"]]]

    scenario_weekly = {
        "n_assets_cov": 8.0,
        "n_types": 3.0,
        "disc_depth_bc": 0.15,    # 15% average depth
        "sov": 0.35,
        "esov": 0.10,
        "is_leader": 1.0,
        "running_carto_media": 1.0,
        "woy_sin": 0.0, "woy_cos": 1.0,
        # turn on doses for some media types if present:
        # e.g., "dose_type::aisle fin": 1.0, "dose_type::digital screens supers": 0.7
    }
    weekly_pred = predict_weekly_uplift(weekly_model, template_cols, scenario_weekly)
    print(f"[Interactive] Predicted weekly uplift under scenario: {weekly_pred:,.0f}")

    # b) Campaign-level scenario
    template_agg = campaign_agg.copy()
    dose_cols_avg = [c for c in template_agg.columns if c.startswith("dose_type::") and c.endswith("_avg")]
    scenario_campaign = {
        "duration_weeks": 6,
        "n_assets_total": 42,
        "n_types_avg": 3.2,
        "esov_avg": 0.08,
        "disc_depth_avg": 0.12,
        "is_leader": 1.0,
        # e.g., "dose_type::aisle fin_avg": 0.9, "dose_type::digital screens supers_avg": 0.6
    }
    total_pred = predict_campaign_total(model, template_agg, scenario_campaign)
    print(f"[Interactive] Predicted campaign total uplift under scenario: {total_pred:,.0f}")

if __name__ == "__main__":
    main()
