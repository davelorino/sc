#!/usr/bin/env python
from __future__ import annotations
import os, argparse
import pandas as pd
from collections import defaultdict

from src.ri.get_data.media.parse_media_table import to_list_flexible
from src.ri.model.orchestration.run_counterfactuals_full import run_counterfactuals_full, ALL_TARGETS
from src.ri.insights.orchestration.build_full_grid import build_weekly_table_for_metric_grain, fit_models_for_metric_grain
from src.ri.insights.modeling.predict import predict_weekly_uplift, predict_campaign_total

GRAINS = ["sku","brand","brand_category","brand_subcategory","category","subcategory"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media_csv", required=True)
    ap.add_argument("--tx_parquet", required=True)
    ap.add_argument("--bookings", nargs="+", required=True)
    ap.add_argument("--outdir", default="4_outputs")
    args = ap.parse_args()

    os.makedirs(f"{args.outdir}/reports", exist_ok=True)
    os.makedirs(f"{args.outdir}/artifacts", exist_ok=True)

    media = pd.read_csv(args.media_csv, parse_dates=["campaign_start_date","campaign_end_date","media_start_date","media_end_date"])
    tx = (pd.read_parquet(args.tx_parquet) if args.tx_parquet.endswith(".parquet")
          else pd.read_csv(args.tx_parquet, parse_dates=["week_start"]))

    # product dim
    product_dim = (tx.rename(columns={"product_number":"product_id",
                                      "BrandDescription":"brand",
                                      "CategoryDescription":"category",
                                      "SubCategoryDescription":"subcategory"})
                    [["product_id","brand","category","subcategory"]].drop_duplicates())
    # promo SKUs per campaign
    promo_skus = {cid: sorted(set(s for row in media.loc[media["booking_number"]==cid, "sorted_sku_list"].dropna()
                                  for s in to_list_flexible(row)))
                  for cid in args.bookings}

    # ---- A) Counterfactuals for ALL metrics × ALL node bands
    cf = run_counterfactuals_full(media_master_df=media, tx_master_df=tx, booking_numbers=args.bookings)

    # ---- B) Build weekly tables and fit models for each (metric, grain)
    soundbites = []
    model_registry = {}  # (metric, grain) -> (weekly_pipe, campaign_model, campaign_agg, coef tables)
    for target in ALL_TARGETS:
        if target not in cf.get(next(iter(cf)), {}):
            # if first campaign lacks this metric entirely, skip
            pass
        for grain in GRAINS:
            weekly = build_weekly_table_for_metric_grain(
                cmp_by_campaign_metric_band={cid: cf[cid].get(target, {}) for cid in cf.keys()},
                media_master_df=media,
                tx_master_df=tx,
                product_dim=product_dim,
                promo_skus_by_campaign=promo_skus,
                grain=grain,
                media_type_weights=None
            )
            if weekly.empty:
                continue
            weekly.to_parquet(f"{args.outdir}/artifacts/weekly_{target}_{grain}.parquet", index=False)

            coef_weekly, coef_campaign, weekly_pipe, campaign_model, camp_agg = fit_models_for_metric_grain(weekly)
            coef_weekly.to_csv(f"{args.outdir}/reports/coef_weekly_{target}_{grain}.csv", index=False)
            coef_campaign.to_csv(f"{args.outdir}/reports/coef_campaign_{target}_{grain}.csv", index=False)
            camp_agg.to_csv(f"{args.outdir}/reports/campaign_agg_{target}_{grain}.csv", index=False)

            # minimal soundbites
            top_pos = coef_weekly.sort_values("coef", ascending=False).head(3)
            top_neg = coef_weekly.sort_values("coef", ascending=True).head(3)
            soundbites.append(f"[{target} @ {grain}] Weekly positive drivers:")
            for _, r in top_pos.iterrows():
                soundbites.append(f"  + {r['feature']} ({r['coef']:.3f})")
            soundbites.append(f"[{target} @ {grain}] Weekly negative drivers:")
            for _, r in top_neg.iterrows():
                soundbites.append(f"  - {r['feature']} ({r['coef']:.3f})")

            model_registry[(target, grain)] = {
                "weekly_model": weekly_pipe,
                "campaign_model": campaign_model,
                "campaign_agg": camp_agg,
                "template_weekly": weekly[[c for c in weekly.columns if c.startswith('dose_type::') or c in
                                           ["n_assets_cov","n_types","disc_depth","sov","esov","is_leader",
                                            "running_carto_media","woy_sin","woy_cos"]]].head(1) # schema only
            }

    with open(f"{args.outdir}/reports/soundbites_full_grid.txt","w") as f:
        f.write("\n".join(soundbites))
    print("Wrote soundbites for full grid.")

    # ---- C) Interactive examples
    # Example: predict weekly uplift for "new_to_brand_shoppers" at brand_subcategory under a scenario
    key = ("new_to_brand_shoppers","brand_subcategory")
    if key in model_registry:
        mr = model_registry[key]
        tmpl = mr["template_weekly"]
        scenario = {
            "n_assets_cov": 6.0,
            "n_types": 3.0,
            "disc_depth": 0.12,
            "sov": 0.30,
            "esov": 0.07,
            "is_leader": 1.0,
            "running_carto_media": 1.0,
            "woy_sin": 0.0, "woy_cos": 1.0,
            # if your campaign uses specific media types, include their doses, e.g.:
            # "dose_type::aisle fin": 0.8, "dose_type::digital screens supers": 0.5
        }
        wk_pred = predict_weekly_uplift(mr["weekly_model"], tmpl, scenario)
        print(f"[Interactive] Weekly uplift for {key}: {wk_pred:,.0f}")

    # Example: predict campaign total uplift for "sales" at brand_category
    key2 = ("sales","brand_category")
    if key2 in model_registry:
        mr2 = model_registry[key2]
        camp_scenario = {
            "duration_weeks": 6,
            "n_assets_total": 40,
            "n_types_avg": 3.1,
            "esov_avg": 0.08,
            "disc_depth_avg": 0.10,
            "is_leader": 1.0,
            # optional: add avg doses per type if present, e.g. "dose_type::aisle fin_avg": 0.6
        }
        camp_pred = predict_campaign_total(mr2["campaign_model"], mr2["campaign_agg"], camp_scenario)
        print(f"[Interactive] Campaign total uplift for {key2}: {camp_pred:,.0f}")

if __name__ == "__main__":
    main()
