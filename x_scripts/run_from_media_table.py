#!/usr/bin/env python
from __future__ import annotations
import os, sys, time, argparse
from typing import List, Optional, Dict
from collections import defaultdict
import pandas as pd
from google.cloud import bigquery

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# Your helpers
from src.ri.get_data.media.parse_media_table import to_list_flexible
from src.ri.get_data.transaction_pipeline import get_transaction_data_by_scope
from src.ri.get_data.media.build_weekly_store_calendar import (
    build_weekly_store_calendar_for_campaign,
    weeks_to_store_sets,
)
from src.ri.model.orchestration.build_grids_v3 import _make_registry
from src.ri.model.forecasting.cohort_forecasting import forecast_per_week_dynamic_cohorts

# Exogenous regressors (treatment flag is NOT included)
EXOG = [
    "discountpercent",
    "max_internal_competitor_discount_percent",
    "n_competitors","n_any_cheaper","n_shelf_cheaper",
    "n_promo_cheaper_no_hurdle","n_promo_cheaper_hurdle",
    "avg_cheaper_gap","worst_gap","p90_gap",
    "brochure_Not on brochure","multibuy_Not on Multibuy",
]

ALL_TARGETS = [
    "sales",
    "shoppers",
    "new_to_sku_sales",
    "new_to_sku_shoppers",
    "new_to_brand_sales",
    "new_to_brand_shoppers",
    #"new_to_category_sales","new_to_category_shoppers",
    #"new_to_subcategory_sales","new_to_subcategory_shoppers",
]

GRAINS = ["sku","brand","brand_category","brand_subcategory","category","subcategory"]

def load_media_df(*, media_table: Optional[str], media_csv: Optional[str], project: Optional[str]) -> pd.DataFrame:
    if media_table:
        from google.cloud import bigquery
        client = bigquery.Client(project=project)
        sql = f"""
        SELECT booking_number, opportunity_name,
               campaign_start_date, campaign_end_date,
               media_start_date, media_end_date,
               media_array, media_location_array, media_type_array,
               sorted_store_list, sorted_sku_list,
               campaign_week, campaign_week_split
        FROM `{media_table}`
        ORDER BY booking_number, media_start_date, media_end_date
        """
        df = client.query(sql).result().to_dataframe()
    elif media_csv:
        df = pd.read_csv(
            media_csv,
            parse_dates=["campaign_start_date","campaign_end_date","media_start_date","media_end_date"]
        )
    else:
        raise ValueError("Provide --media-table (BQ) or --media-csv (file).")

    for c in ("media_array","media_location_array","media_type_array","sorted_store_list","sorted_sku_list"):
        if c in df.columns:
            df[c] = df[c].apply(to_list_flexible)
    return df

def select_bookings(media: pd.DataFrame, mode: str, start: Optional[str], end: Optional[str]) -> List[str]:
    if mode == "all":
        return sorted(media["booking_number"].astype(str).unique().tolist())
    if mode == "window":
        if not start or not end:
            raise ValueError("--mode window requires --start and --end")
        m = media[(pd.to_datetime(media["campaign_start_date"]) >= pd.to_datetime(start)) &
                  (pd.to_datetime(media["campaign_end_date"])   <= pd.to_datetime(end))]
        return sorted(m["booking_number"].astype(str).unique().tolist())
    raise ValueError(mode)

def _date_window(media_one: pd.DataFrame, lookback_weeks: int = 104, pad_weeks: int = 2):
    min_start = pd.to_datetime(media_one["media_start_date"]).min()
    max_end   = pd.to_datetime(media_one["media_end_date"]).max()
    min_d = (min_start - pd.Timedelta(weeks=lookback_weeks)).to_period("W-WED").start_time
    max_d = (max_end   + pd.Timedelta(weeks=pad_weeks)).to_period("W-WED").start_time
    return min_d, max_d

# -------- BQ on-demand fetch (recommended) --------

def fetch_campaign_tx_slice_bq(
    *,
    client,
    tx_table_fq: str,               # base weekly table (lean _base or old _pn2)
    ranked_table_fq: str,           # kept for signature compatibility; not used
    promo_skus_article_ids: list[str],
    stores_union: list[str],
    min_date, max_date
) -> pd.DataFrame:

    # 1) Expand promo SKUs -> universe ARTICLE IDs for this window (brand / cat / subcat expansions)
    params = [
        bigquery.ArrayQueryParameter("skus", "STRING", [str(x) for x in promo_skus_article_ids]),
        bigquery.ScalarQueryParameter("min_d", "DATE", str(pd.to_datetime(min_date).date())),
        bigquery.ScalarQueryParameter("max_d", "DATE", str(pd.to_datetime(max_date).date())),
    ]
    universe_sql = f"""
    WITH skus AS (SELECT id FROM UNNEST(@skus) AS id),
    promo AS (
      SELECT DISTINCT t.BrandDescription, t.CategoryDescription, t.SubCategoryDescription
      FROM `{tx_table_fq}` t
      JOIN skus ON CAST(t.article_id AS STRING) = skus.id
      WHERE t.week_start BETWEEN @min_d AND @max_d
    ),
    universe_articles AS (
      -- promo SKUs directly
      SELECT DISTINCT CAST(t.article_id AS STRING) AS article_id
      FROM `{tx_table_fq}` t
      JOIN skus s ON CAST(t.article_id AS STRING) = s.id
      WHERE t.week_start BETWEEN @min_d AND @max_d

      UNION DISTINCT
      -- whole brand(s)
      SELECT DISTINCT CAST(t.article_id AS STRING)
      FROM `{tx_table_fq}` t
      JOIN (SELECT DISTINCT BrandDescription FROM promo) b
        ON t.BrandDescription = b.BrandDescription
      WHERE t.week_start BETWEEN @min_d AND @max_d

      UNION DISTINCT
      -- brand × category
      SELECT DISTINCT CAST(t.article_id AS STRING)
      FROM `{tx_table_fq}` t
      JOIN (SELECT DISTINCT BrandDescription, CategoryDescription FROM promo) bc
        ON t.BrandDescription = bc.BrandDescription
       AND t.CategoryDescription = bc.CategoryDescription
      WHERE t.week_start BETWEEN @min_d AND @max_d

      UNION DISTINCT
      -- brand × subcategory
      SELECT DISTINCT CAST(t.article_id AS STRING)
      FROM `{tx_table_fq}` t
      JOIN (SELECT DISTINCT BrandDescription, SubCategoryDescription FROM promo) bs
        ON t.BrandDescription = bs.BrandDescription
       AND t.SubCategoryDescription = bs.SubCategoryDescription
      WHERE t.week_start BETWEEN @min_d AND @max_d

      UNION DISTINCT
      -- whole category(ies)
      SELECT DISTINCT CAST(t.article_id AS STRING)
      FROM `{tx_table_fq}` t
      JOIN (SELECT DISTINCT CategoryDescription FROM promo) c
        ON t.CategoryDescription = c.CategoryDescription
      WHERE t.week_start BETWEEN @min_d AND @max_d

      UNION DISTINCT
      -- whole subcategory(ies)
      SELECT DISTINCT CAST(t.article_id AS STRING)
      FROM `{tx_table_fq}` t
      JOIN (SELECT DISTINCT SubCategoryDescription FROM promo) sc
        ON t.SubCategoryDescription = sc.SubCategoryDescription
      WHERE t.week_start BETWEEN @min_d AND @max_d
    )
    SELECT article_id FROM universe_articles
    """
    universe_ids = client.query(
        universe_sql, job_config=bigquery.QueryJobConfig(query_parameters=params)
    ).result().to_dataframe()["article_id"].astype(str).tolist()

    # 2) Call the SSOT to build the exact slice the pipeline expects
    return get_transaction_data_by_scope(
        client=client,
        base_table_fq=tx_table_fq,             # pass the same base
        sku_list=universe_ids,                 # ARTICLE IDs (expanded)
        store_list=[str(s) for s in stores_union],
        start_date=str(pd.to_datetime(min_date).date()),
        end_date=str(pd.to_datetime(max_date).date()),
    )

# -------- Parquet fallback (loads once; filters in memory) --------
def fetch_campaign_tx_slice_parquet(
    *,
    tx_master_df: pd.DataFrame,
    product_dim_df: pd.DataFrame,
    promo_skus_article_ids: List[str],
    stores_union: List[str],
    min_date, max_date
) -> pd.DataFrame:
    # Expand universe in memory using product_dim_df (must contain all products)
    promo = product_dim_df[product_dim_df["article_id"].astype(str).isin([str(x) for x in promo_skus_article_ids])]
    brands = promo["BrandDescription"].unique().tolist()
    cats   = promo["CategoryDescription"].unique().tolist()
    subcs  = promo["SubCategoryDescription"].unique().tolist()

    uni = set(promo["product_number"].astype(str))
    uni |= set(product_dim_df.loc[product_dim_df["BrandDescription"].isin(brands), "product_number"].astype(str))
    uni |= set(product_dim_df.loc[(product_dim_df["BrandDescription"].isin(brands)) &
                                  (product_dim_df["CategoryDescription"].isin(cats)), "product_number"].astype(str))
    uni |= set(product_dim_df.loc[(product_dim_df["BrandDescription"].isin(brands)) &
                                  (product_dim_df["SubCategoryDescription"].isin(subcs)), "product_number"].astype(str))
    uni |= set(product_dim_df.loc[product_dim_df["CategoryDescription"].isin(cats), "product_number"].astype(str))
    uni |= set(product_dim_df.loc[product_dim_df["SubCategoryDescription"].isin(subcs), "product_number"].astype(str))

    df = tx_master_df[
        (tx_master_df["product_number"].astype(str).isin(uni)) &
        (tx_master_df["store_id"].astype(str).isin(set(map(str, stores_union)))) &
        (pd.to_datetime(tx_master_df["week_start"]) >= pd.to_datetime(min_date)) &
        (pd.to_datetime(tx_master_df["week_start"]) <= pd.to_datetime(max_date))
    ].copy()
    return df

def run_pipeline(
    *,
    media: pd.DataFrame,
    tx_table: Optional[str],
    tx_parquet: Optional[str],
    project: Optional[str],
    outdir: str,
    metrics: List[str],
    grains: List[str],
) -> None:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "artifacts"), exist_ok=True)

    # If parquet mode, load once
    tx_master_df = None
    product_dim_df = None
    bq_client = None
    ranked_table = None

    if tx_table:
        from google.cloud import bigquery
        bq_client = bigquery.Client(project=project)
        # Infer ranked table from base table name by convention
        # e.g. dataset.sku_store_week_sales_pn2 -> dataset.sku_store_week_sales_ranked
        parts = tx_table.split(".")
        ranked_table = ".".join(parts[:-1] + [parts[-1].replace("_pn2", "_ranked")])
    else:
        # Parquet fallback
        tx_master_df = (pd.read_parquet(tx_parquet) if tx_parquet.endswith(".parquet")
                        else pd.read_csv(tx_parquet, parse_dates=["week_start"]))
        # Minimal product dim for universe expansion
        product_dim_df = tx_master_df[[
            "product_number","article_id",
            "BrandDescription","CategoryDescription","SubCategoryDescription"
        ]].drop_duplicates()

    t0_all = time.time(); per_stage = defaultdict(list); rows_prog = []
    bookings = sorted(media["booking_number"].astype(str).unique().tolist())
    iterator = tqdm(bookings, total=len(bookings), dynamic_ncols=True) if tqdm else bookings

    for i, cid in enumerate(iterator, 1):
        t0 = time.time()
        mezzo = media[media["booking_number"].astype(str) == cid].copy()
        if mezzo.empty:
            continue

        # promo SKUs (article ids)
        promo_skus = sorted({s for row in mezzo["sorted_sku_list"].dropna() for s in to_list_flexible(row)})
        if not promo_skus:
            continue

        # dynamic cohorts
        t_cal = time.time()
        fleet = sorted(set(str(x) for x in mezzo["sorted_store_list"].explode().dropna().tolist() if str(x).strip()))
        # The calendar builder already handles "empty list means all stores" using fleet argument:
        from src.ri.get_data.media.build_weekly_store_calendar import get_fleet_stores
        if not fleet and tx_master_df is not None:
            fleet = sorted(tx_master_df["store_id"].astype(str).unique().tolist())
        elif not fleet and tx_master_df is None and bq_client is not None:
            # get fleet from tx table (cheap)
            fleet_sql = f"SELECT DISTINCT CAST(store_id AS STRING) AS sid FROM `{tx_table}`"
            fleet = bq_client.query(fleet_sql).result().to_dataframe()["sid"].tolist()

        cal = build_weekly_store_calendar_for_campaign(mezzo, fleet)
        wk_to_stores = weeks_to_store_sets(cal)
        stores_union = sorted({sid for s in wk_to_stores.values() for sid in s})
        per_stage["calendar"].append(time.time() - t_cal)

        # date window
        min_d, max_d = _date_window(mezzo, lookback_weeks=104, pad_weeks=2)

        # fetch campaign slice
        t_fetch = time.time()
        if bq_client is not None:
            tx_slice = fetch_campaign_tx_slice_bq(
                client=bq_client,
                tx_table_fq=tx_table,
                ranked_table_fq=ranked_table,
                promo_skus_article_ids=promo_skus,
                stores_union=stores_union,
                min_date=min_d, max_date=max_d
            )
        else:
            tx_slice = fetch_campaign_tx_slice_parquet(
                tx_master_df=tx_master_df,
                product_dim_df=product_dim_df,
                promo_skus_article_ids=promo_skus,
                stores_union=stores_union,
                min_date=min_d, max_date=max_d
            )
        per_stage["fetch_slice"].append(time.time() - t_fetch)
        if tx_slice.empty:
            continue

        # product dimension for registry
        product_dim = (tx_slice.rename(columns={
            "product_number":"product_id",
            "BrandDescription":"brand",
            "CategoryDescription":"category",
            "SubCategoryDescription":"subcategory"
        })[["product_id","brand","category","subcategory"]].drop_duplicates())
        registry = _make_registry(promo_skus, product_dim)

        # training cutoff
        train_end = (pd.to_datetime(mezzo["media_start_date"]).min() - pd.Timedelta(days=1))

        for tgt in metrics:
            for grain in grains:
                nodes = registry.get(grain, [])
                if not nodes:
                    continue
                t_fc = time.time()
                cmp_df = forecast_per_week_dynamic_cohorts(
                    tx_master_df=tx_slice,
                    week_to_storelist=wk_to_stores,
                    nodes=nodes,
                    train_end_date=train_end,
                    target=tgt,
                    exog_regressors=EXOG
                )
                out_dir = os.path.join(outdir, "artifacts", cid)
                os.makedirs(out_dir, exist_ok=True)
                cmp_df.to_parquet(os.path.join(out_dir, f"compare_{tgt}_{grain}.parquet"), index=False)
                per_stage[f"fc_{tgt}@{grain}"].append(time.time() - t_fc)

        dur = time.time() - t0
        rows_prog.append({"campaign_id": cid, "seconds_total": round(dur,2)})
        if tqdm:
            avg = (time.time() - t0_all) / max(1,i)
            remain = (len(bookings) - i) * avg
            iterator.set_postfix_str(f"{dur/60:.1f}m | avg {avg/60:.1f}m/c | ~rem {remain/60:.1f}m")

    # progress/bottlenecks
    pd.DataFrame(rows_prog).to_csv(os.path.join(outdir, "progress_campaigns.csv"), index=False)
    stage_avg = pd.DataFrame(
        [{"stage":k, "avg_seconds": sum(v)/len(v), "n": len(v)} for k,v in per_stage.items() if v]
    ).sort_values("avg_seconds", ascending=False)
    stage_avg.to_csv(os.path.join(outdir, "progress_stages.csv"), index=False)
    if tqdm:
        print("Top bottlenecks (avg seconds):")
        print(stage_avg.head(10).to_string(index=False))

def parse_args():
    ap = argparse.ArgumentParser("Run forecasting off media table with on-demand per-campaign transaction slices.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--media-table", help="BQ table, e.g. project.dataset.media_master_v1")
    src.add_argument("--media-csv", help="CSV export of media master")

    # choose either tx-table (BQ live slicing) or tx-parquet (fallback)
    txsrc = ap.add_mutually_exclusive_group(required=True)
    txsrc.add_argument("--tx-table", help="BQ table e.g. project.dataset.sku_store_week_sales_pn2")
    txsrc.add_argument("--tx-parquet", help="Path to parquet/csv of the transactions master (fallback)")

    ap.add_argument("--project", help="GCP project (required if using BQ)")
    ap.add_argument("--outdir", default="4_outputs")

    ap.add_argument("--mode", choices=["all","window"], default="all")
    ap.add_argument("--start", help="YYYY-MM-DD (campaign_start_date >= start) for --mode window")
    ap.add_argument("--end", help="YYYY-MM-DD (campaign_end_date <= end) for --mode window")

    ap.add_argument("--metrics", nargs="+", default=["sales"])
    ap.add_argument("--grains",  nargs="+", default=["brand_category"])

    return ap.parse_args()

def main():
    args = parse_args()
    media = load_media_df(media_table=args.media_table, media_csv=args.media_csv, project=args.project)
    bookings = select_bookings(media, args.mode, args.start, args.end)
    if not bookings:
        print("No campaigns match your selection.")
        sys.exit(0)
    media = media[media["booking_number"].astype(str).isin(set(bookings))].copy()

    metrics = (ALL_TARGETS if (len(args.metrics)==1 and args.metrics[0].lower()=="all") else args.metrics)
    grains  = (GRAINS      if (len(args.grains)==1  and args.grains[0].lower()=="all")  else args.grains)

    run_pipeline(
        media=media,
        tx_table=args.tx_table,
        tx_parquet=args.tx_parquet,
        project=args.project,
        outdir=args.outdir,
        metrics=metrics,
        grains=grains
    )

if __name__ == "__main__":
    main()
