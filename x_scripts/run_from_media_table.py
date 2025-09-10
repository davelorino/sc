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
from src.ri.get_data.transaction_pipeline import get_transaction_data_by_scope, get_transaction_data_by_scope_fast, get_transaction_data_by_scope_fast2
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



def _sanitize_ids(values: List[str], type_: str, name: str) -> List[str]:
    out = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        # optional: allow only digits for product_numbers, but article_id may be alphanumeric
        out.append(s if type_ == "STRING" else str(int(s)))  # coerce stores to digits if desired
    # de-dupe while preserving order
    seen = set()
    deduped = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return deduped

def preflight_scope(
    client: bigquery.Client,
    base_table_fq: str,
    sku_list: List[str],
    store_list: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
    log_dir: str = "preflight_logs",
    print_full: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    os.makedirs(log_dir, exist_ok=True)
    tstamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Sanitize lists
    skus = _sanitize_ids(sku_list, "STRING", "sku_ids")
    stores = _sanitize_ids(store_list, "INT64",  "store_ids")  # change to STRING if your store_id is string in BQ

    # 2) Get brands implied by base for these (skus × stores × dates)
    sql_brands = f"""
    DECLARE min_d DATE DEFAULT @min_d;
    DECLARE max_d DATE DEFAULT @max_d;
    WITH skus AS (
      SELECT DISTINCT id
      FROM UNNEST(@sku_ids) id
      WHERE id IS NOT NULL AND TRIM(id) <> ''
    ),
    stores AS (
      SELECT DISTINCT CAST(id AS STRING) AS id
      FROM UNNEST(@store_ids) id
      WHERE id IS NOT NULL AND TRIM(CAST(id AS STRING)) <> ''
    )
    SELECT DISTINCT BrandDescription
    FROM `{base_table_fq}`
    WHERE week_start BETWEEN min_d AND max_d
      AND CAST(article_id AS STRING) IN (SELECT id FROM skus)
      AND CAST(store_id  AS STRING)  IN (SELECT id FROM stores)
      AND BrandDescription IS NOT NULL AND TRIM(BrandDescription) <> ''
    ORDER BY BrandDescription
    """
    job = client.query(
        sql_brands,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("sku_ids", "STRING", skus),
                bigquery.ArrayQueryParameter("store_ids", "STRING", stores), # casted to STRING above
                bigquery.ScalarQueryParameter("min_d", "DATE", pd.to_datetime(start_date).date()),
                bigquery.ScalarQueryParameter("max_d", "DATE", pd.to_datetime(end_date).date()),
            ]
        ),
    )
    brands = job.result().to_dataframe()["BrandDescription"].astype(str).tolist()

    # 3) Print / log
    print(f"[Preflight] SKUs ({len(skus)}):")
    if print_full or len(skus) <= 200:
        print(skus)
    else:
        print(skus[:200], f"...(+{len(skus)-200} more)")

    print(f"[Preflight] Stores ({len(stores)}):")
    if print_full or len(stores) <= 200:
        print(stores)
    else:
        print(stores[:200], f"...(+{len(stores)-200} more)")

    print(f"[Preflight] Brands ({len(brands)}):")
    if print_full or len(brands) <= 200:
        print(brands)
    else:
        print(brands[:200], f"...(+{len(brands)-200} more)")

    # Also persist to disk for forensic debugging
    with open(os.path.join(log_dir, f"scope_{tstamp}.json"), "w") as f:
        json.dump({"skus": skus, "stores": stores, "brands": brands}, f, indent=2)

    # 4) Optional: Dry-run bytes on the main query (replace with your actual SQL builder)
    # cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, query_parameters=[...])
    # job = client.query(main_sql, job_config=cfg)
    # print(f"[Preflight] Estimated bytes processed: {job.total_bytes_processed:,}")

    return skus, stores, brands

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
    print("Initiating...")
    # --- 1) Expand promo SKUs -> universe ARTICLE IDs for this window (same logic as before) ---
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
    print("Getting universe ids...", end=" ", flush=True)
    universe_ids = client.query(
        universe_sql, job_config=bigquery.QueryJobConfig(query_parameters=params)
    ).result().to_dataframe()["article_id"].astype(str).tolist()
    print("Done!")
    # --- 2) Clean lists (order-preserving de-dupe + drop blanks) ---
    def _clean_list(seq):
        out, seen = [], set()
        for x in seq or []:
            s = str(x).strip()
            if not s: 
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    universe_ids = _clean_list(universe_ids)
    stores_union = _clean_list(stores_union)
    print(f"Skus: {universe_ids}")
    print(f"stores union: {stores_union.sort()}")
    # --- 3) PREFLIGHT DIAGNOSTICS (explicit) BEFORE the heavy query ---
    # Derive the Brand list in-scope from the base table for this (SKUs × stores × window)
    pre_params = [
        bigquery.ArrayQueryParameter("sku_ids",   "STRING", universe_ids),
        bigquery.ArrayQueryParameter("store_ids", "STRING", stores_union),
        bigquery.ScalarQueryParameter("min_d", "DATE", str(pd.to_datetime(min_date).date())),
        bigquery.ScalarQueryParameter("max_d", "DATE", str(pd.to_datetime(max_date).date())),
    ]
    pre_sql = f"""
    WITH
      skus AS (
        SELECT DISTINCT id
        FROM UNNEST(@sku_ids) id
        WHERE id IS NOT NULL AND TRIM(id) <> ''
      )
    #  stores AS (
    #    SELECT DISTINCT CAST(id AS STRING) AS id
    #    FROM UNNEST(@store_ids) id
    #    WHERE id IS NOT NULL AND TRIM(CAST(id AS STRING)) <> ''
    #  )
    SELECT DISTINCT BrandDescription
    FROM `{tx_table_fq}`
    WHERE week_start BETWEEN @min_d AND @max_d
      AND CAST(article_id AS STRING) IN (SELECT id FROM skus)
      #AND CAST(store_id  AS STRING)  IN (SELECT id FROM stores)
      AND BrandDescription IS NOT NULL AND TRIM(BrandDescription) <> ''
    ORDER BY BrandDescription
    """
    print("Getting brands...", end=" ", flush=True)
    brands = client.query(
        pre_sql, job_config=bigquery.QueryJobConfig(query_parameters=pre_params)
    ).result().to_dataframe()["BrandDescription"].astype(str).tolist()
    print("Done!")
    # Print FULL lists
    print(f"[Preflight] SKUs ({len(universe_ids)}): {universe_ids}")
    print(f"[Preflight] Stores ({len(stores_union)}): {stores_union}")
    print(f"[Preflight] Brands ({len(brands)}): {brands}")

    if not universe_ids:
        raise ValueError("[Preflight] SKU universe is empty after expansion/clean.")
    if not stores_union:
        raise ValueError("[Preflight] Store list is empty after clean.")
    if not brands:
        raise ValueError("[Preflight] Brand list empty for this scope; check inputs.")
    print("Begininng get_transaction_data_by_scope_fast ...", end=" ", flush=True)
    # --- 4) Execute the main pull (this function also guards against explosions) ---
    return get_transaction_data_by_scope_fast(
        client=client,
        base_table_fq=tx_table_fq,
        sku_list=universe_ids,                 # ARTICLE IDs (expanded, cleaned)
        store_list=stores_union,               # cleaned store list
        start_date=str(pd.to_datetime(min_date).date()),
        end_date=str(pd.to_datetime(max_date).date()),
    )

def fetch_campaign_tx_slice_bq2(
    *,
    client,
    tx_table_fq: str,
    ranked_table_fq: str,           # kept for signature compatibility; not used
    promo_skus_article_ids: List[str],
    stores_union: List[str],
    min_date, max_date,
    grains: List[str],              # drives conditional SKU expansion
) -> pd.DataFrame:
    """
    Campaign slice fetcher that enforces:
      - Promo SKUs must belong to EXACTLY ONE brand (0 or >1 => skip campaign).
      - No whole-category / whole-subcategory expansion.
      - Universe = promo SKUs (+ brand-subcategory SKUs that share (Brand, SubCategory) with the promo SKUs)
        only if grains require brand-level modeling (brand_subcategory).
      - Delegates to get_transaction_data_by_scope_fast2 which computes:
          * new_to_brand_* as brand-subcategory.
    """
    print("Initiating...")

    def _clean_list(seq):
        out, seen = [], set()
        for x in seq or []:
            s = str(x).strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _as_date(x):
        return pd.to_datetime(x).date()

    promo_skus_article_ids = _clean_list(promo_skus_article_ids)
    stores_union           = _clean_list(stores_union)

    min_d = _as_date(min_date)
    max_d = _as_date(max_date)

    # ---------- Enforce: promo SKUs map to exactly ONE brand ----------
    promo_brands_sql = f"""
    WITH skus AS (
      SELECT DISTINCT id FROM UNNEST(@skus) AS id
      WHERE id IS NOT NULL AND TRIM(id) <> ''
    )
    SELECT DISTINCT t.BrandDescription AS brand
    FROM `{tx_table_fq}` t
    JOIN skus s ON CAST(t.article_id AS STRING) = s.id
    WHERE t.week_start BETWEEN @min_d AND @max_d
      AND t.BrandDescription IS NOT NULL AND TRIM(t.BrandDescription) <> ''
    """
    brand_params = [
        bigquery.ArrayQueryParameter("skus", "STRING", promo_skus_article_ids),
        bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
        bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
    ]
    promo_brands_df = client.query(
        promo_brands_sql, job_config=bigquery.QueryJobConfig(query_parameters=brand_params)
    ).result().to_dataframe()

    n_brands_promo = int(promo_brands_df.shape[0])
    brands_pretty  = promo_brands_df["brand"].astype(str).tolist()
    print(f"[Diag] Promo-SKU brand count: {n_brands_promo} | {brands_pretty}")

    if n_brands_promo != 1:
        raise ValueError(f"[SkipCampaign] Promo SKUs span {n_brands_promo} brand(s); require exactly 1. Found: {brands_pretty}")

    # ---------- Build universe with NO whole-category/whole-subcategory ----------
    # Allow brand_subcategory expansion ONLY if grains require "brand" modeling (we alias brand -> brand_subcategory).
    wants_brand_sub = any(g in {"brand","brand_subcategory"} for g in grains)

    with_parts = []

    with_parts.append("""
skus AS (
  SELECT DISTINCT id FROM UNNEST(@skus) AS id
  WHERE id IS NOT NULL AND TRIM(id) <> ''
)""")

    # Always include the listed promo SKUs
    with_parts.append(f"""
sku_only AS (
  SELECT DISTINCT CAST(t.article_id AS STRING) AS article_id
  FROM `{tx_table_fq}` t
  JOIN skus s ON CAST(t.article_id AS STRING) = s.id
  WHERE t.week_start BETWEEN @min_d AND @max_d
)""")

    # Expand ONLY to brand-subcategory pairs implied by the promo SKUs
    if wants_brand_sub:
        with_parts.append(f"""
promo_brand_sub AS (
  SELECT DISTINCT t.BrandDescription, t.SubCategoryDescription
  FROM `{tx_table_fq}` t
  JOIN skus s ON CAST(t.article_id AS STRING) = s.id
  WHERE t.week_start BETWEEN @min_d AND @max_d
    AND t.BrandDescription IS NOT NULL AND TRIM(t.BrandDescription) <> ''
    AND t.SubCategoryDescription IS NOT NULL AND TRIM(t.SubCategoryDescription) <> ''
),
brand_subcat AS (
  SELECT DISTINCT CAST(t.article_id AS STRING) AS article_id
  FROM `{tx_table_fq}` t
  JOIN promo_brand_sub pbs
    ON t.BrandDescription       = pbs.BrandDescription
   AND t.SubCategoryDescription = pbs.SubCategoryDescription
  WHERE t.week_start BETWEEN @min_d AND @max_d
)""")

    union_selects = ["SELECT article_id FROM sku_only"]
    if wants_brand_sub:
        union_selects.append("SELECT article_id FROM brand_subcat")

    universe_sql = "WITH\n" + ",\n".join(with_parts) + "\n" + (
        "SELECT article_id\nFROM (\n  " + "\n  UNION DISTINCT\n  ".join(union_selects) + "\n)"
    )

    params = [
        bigquery.ArrayQueryParameter("skus", "STRING", promo_skus_article_ids),
        bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
        bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
    ]

    print("Getting universe ids...", end=" ", flush=True)
    universe_ids = client.query(
        universe_sql, job_config=bigquery.QueryJobConfig(query_parameters=params)
    ).result().to_dataframe()["article_id"].astype(str).tolist()
    print("Done!")

    universe_ids = _clean_list(universe_ids)

    # Optional, cheap sampled brand log (visibility only)
    pre_sql = f"""
    DECLARE sample_start DATE DEFAULT GREATEST(@min_d, DATE_SUB(@max_d, INTERVAL 28 DAY));
    SELECT DISTINCT BrandDescription
    FROM `{tx_table_fq}`
    WHERE week_start BETWEEN sample_start AND @max_d
      AND CAST(article_id AS STRING) IN UNNEST(@sku_ids)
      AND BrandDescription IS NOT NULL AND TRIM(BrandDescription) <> ''
    ORDER BY 1
    """
    pre_params = [
        bigquery.ArrayQueryParameter("sku_ids", "STRING", universe_ids),
        bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
        bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
    ]
    print("Getting brands (sampled)...", end=" ", flush=True)
    brands = client.query(
        pre_sql, job_config=bigquery.QueryJobConfig(query_parameters=pre_params)
    ).result().to_dataframe()["BrandDescription"].astype(str).tolist()
    print("Done!")
    print(f"[Preflight] SKUs ({len(universe_ids)}): {universe_ids[:20]}{' ...' if len(universe_ids)>20 else ''}")
    print(f"[Preflight] Stores ({len(stores_union)}; all_stores={len(stores_union)==0})")
    print(f"[Preflight] Brands (sampled) ({len(brands)}): {brands[:20]}{' ...' if len(brands)>20 else ''}")

    if not universe_ids:
        raise ValueError("[SkipCampaign] SKU universe empty after expansion/clean.")

    # Pull the scoped transactions (fast) — new_to_brand_* is brand-subcategory by construction
    print("Begin get_transaction_data_by_scope_fast2 ...", end=" ", flush=True)
    from src.ri.get_data.transaction_pipeline import get_transaction_data_by_scope_fast2
    df = get_transaction_data_by_scope_fast2(
        client=client,
        base_table_fq=tx_table_fq,
        sku_list=universe_ids,
        store_list=stores_union,     # empty => all stores inside SQL
        start_date=str(min_d),
        end_date=str(max_d),
    )
    print("Done!")
    return df


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
    print("run_pipleline setup...", end=" ", flush=True)
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
    print("Done!")
    for i, cid in enumerate(iterator, 1):
        print(f"booking_number {cid}")
        
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
        print(f"Building weekly store calendar...", end=" ", flush=True)
        cal = build_weekly_store_calendar_for_campaign(mezzo, fleet)
        wk_to_stores = weeks_to_store_sets(cal)
        stores_union = sorted({sid for s in wk_to_stores.values() for sid in s})
        per_stage["calendar"].append(time.time() - t_cal)
        print("Done!")

        # date window
        min_d, max_d = _date_window(mezzo, lookback_weeks=104, pad_weeks=2)

        # fetch campaign slice
        t_fetch = time.time()
        if bq_client is not None:
            print(f"Entering fetch_campaign_tx_slice_bq...", end=" ", flush=True)
            tx_slice = fetch_campaign_tx_slice_bq(
                client=bq_client,
                tx_table_fq=tx_table,
                ranked_table_fq=ranked_table,
                promo_skus_article_ids=promo_skus,
                stores_union=stores_union,
                min_date=min_d, max_date=max_d
            )
            print("Done!")
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

def run_pipeline2(
    *,
    media: pd.DataFrame,
    tx_table: Optional[str],
    tx_parquet: Optional[str],
    project: Optional[str],
    outdir: str,
    metrics: List[str],
    grains: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates per-campaign slicing + forecasting with:
      - 'brand' treated as brand_subcategory
      - only 'sku' and 'brand_subcategory' grains honored
      - campaigns skipped unless promo SKUs belong to exactly one brand
      - NO whole-category / whole-subcategory modeling anywhere

    Side effects:
      - Writes d_outputs/.../artifacts per campaign
      - Writes progress CSVs in `outdir`

    Returns:
      {
        "progress": <DataFrame of per-campaign durations>,
        "stages":   <DataFrame of average seconds per stage>
      }
    """
    import os, time
    from collections import defaultdict
    import pandas as pd
    from google.cloud import bigquery

    # Ensure output dirs exist
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "artifacts"), exist_ok=True)

    tx_master_df = None
    product_dim_df = None
    bq_client = None
    ranked_table = None

    print("run_pipeline2 setup...", end=" ", flush=True)
    if tx_table:
        bq_client = bigquery.Client(project=project)
        parts = tx_table.split(".")
        ranked_table = ".".join(parts[:-1] + [parts[-1].replace("_pn2", "_ranked")])
    else:
        # Parquet fallback (rare)
        tx_master_df = (pd.read_parquet(tx_parquet) if tx_parquet.endswith(".parquet")
                        else pd.read_csv(tx_parquet, parse_dates=["week_start"]))
        product_dim_df = tx_master_df[[
            "product_number","article_id",
            "BrandDescription","CategoryDescription","SubCategoryDescription"
        ]].drop_duplicates()
    print("Done!")

    # ---- sanitize grains: only sku and brand_subcategory ----
    cleaned_grains = []
    for g in (grains or []):
        g = g.strip().lower()
        if g == "brand":
            g = "brand_subcategory"
        if g in {"sku","brand_subcategory"} and g not in cleaned_grains:
            cleaned_grains.append(g)
    if not cleaned_grains:
        cleaned_grains = ["sku","brand_subcategory"]

    t0_all = time.time()
    per_stage: Dict[str, List[float]] = defaultdict(list)
    rows_prog: List[Dict[str, object]] = []

    bookings = sorted(media["booking_number"].astype(str).unique().tolist())
    iterator = tqdm(bookings, total=len(bookings), dynamic_ncols=True) if tqdm else bookings

    for i, cid in enumerate(iterator, 1):
        print(f"booking_number {cid}")

        t0 = time.time()
        mezzo = media[media["booking_number"].astype(str) == cid].copy()
        if mezzo.empty:
            continue

        # promo SKUs (article ids) straight from media
        from src.ri.get_data.media.parse_media_table import to_list_flexible
        promo_skus = sorted({s for row in mezzo["sorted_sku_list"].dropna() for s in to_list_flexible(row)})
        if not promo_skus:
            continue

        # fleet/all-stores handling
        t_cal = time.time()
        raw = mezzo["sorted_store_list"].explode().dropna().tolist() if "sorted_store_list" in mezzo.columns else []
        fleet_input = [str(x).strip() for x in raw if str(x).strip()]
        if not fleet_input:
            if tx_master_df is not None:
                fleet_input = sorted(tx_master_df["store_id"].astype(str).unique().tolist())
            elif bq_client is not None:
                fleet_sql = f"SELECT DISTINCT CAST(store_id AS STRING) AS sid FROM `{tx_table}`"
                fleet_input = bq_client.query(fleet_sql).result().to_dataframe()["sid"].astype(str).tolist()

        print("Building weekly store calendar...", end=" ", flush=True)
        from src.ri.get_data.media.build_weekly_store_calendar import (
            build_weekly_store_calendar_for_campaign,
            weeks_to_store_sets,
        )
        cal = build_weekly_store_calendar_for_campaign(mezzo, fleet_input)
        wk_to_stores = weeks_to_store_sets(cal)

        # empty dict or per-week None => interpret as "all stores"
        if not wk_to_stores:
            stores_union = []  # empty => all stores downstream
        else:
            stores_union = sorted({sid for s in wk_to_stores.values() for sid in (s or [])})
        per_stage["calendar"].append(time.time() - t_cal)
        print("Done!")

        # window (13m back + 2w pad)
        min_start = pd.to_datetime(mezzo["media_start_date"]).min()
        max_end   = pd.to_datetime(mezzo["media_end_date"]).max()
        min_d = (min_start - pd.Timedelta(weeks=104)).to_period("W-WED").start_time
        max_d = (max_end   + pd.Timedelta(weeks=2)).to_period("W-WED").start_time

        # fetch campaign slice
        t_fetch = time.time()
        if bq_client is not None:
            print("Entering fetch_campaign_tx_slice_bq2...", end=" ", flush=True)
            try:
                tx_slice = fetch_campaign_tx_slice_bq2(
                    client=bq_client,
                    tx_table_fq=tx_table,
                    ranked_table_fq=ranked_table,
                    promo_skus_article_ids=promo_skus,
                    stores_union=stores_union,          # [] => all stores
                    min_date=min_d, max_date=max_d,
                    grains=cleaned_grains,              # ONLY sku/brand_subcategory
                )
            except ValueError as e:
                msg = str(e)
                if msg.startswith("[SkipCampaign]"):
                    print(f"\n[Skip] booking {cid}: {msg}")
                    per_stage["fetch_slice"].append(time.time() - t_fetch)
                    continue
                raise
        else:
            # Parquet branch
            tx_slice = fetch_campaign_tx_slice_parquet(
                tx_master_df=tx_master_df,
                product_dim_df=product_dim_df,
                promo_skus_article_ids=promo_skus,
                stores_union=stores_union if stores_union else list(tx_master_df["store_id"].astype(str).unique()),
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

        from src.ri.model.orchestration.build_grids_v3 import _make_registry
        registry = _make_registry(promo_skus, product_dim)

        # training cutoff
        train_end = (pd.to_datetime(mezzo["media_start_date"]).min() - pd.Timedelta(days=1))

        # forecast — only metrics in ALL_TARGETS (no category/subcategory totals anywhere)
        full_metrics = ["sales","shoppers","new_to_sku_sales","new_to_sku_shoppers",
                        "new_to_brand_sales","new_to_brand_shoppers"]
        targets = (metrics if metrics != ["all"] else full_metrics)

        for tgt in targets:
            for grain in cleaned_grains:
                nodes = registry.get(grain, [])
                if not nodes:
                    continue

                from src.ri.model.forecasting.cohort_forecasting import forecast_per_week_dynamic_cohorts
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
    progress_df = pd.DataFrame(rows_prog)
    progress_path = os.path.join(outdir, "progress_campaigns.csv")
    progress_df.to_csv(progress_path, index=False)

    stage_rows = [{"stage":k, "avg_seconds": (sum(v)/len(v)), "n": len(v)} for k,v in per_stage.items() if v]
    stages_df = pd.DataFrame(stage_rows).sort_values("avg_seconds", ascending=False) if stage_rows else pd.DataFrame(columns=["stage","avg_seconds","n"])
    stages_path = os.path.join(outdir, "progress_stages.csv")
    stages_df.to_csv(stages_path, index=False)

    if tqdm and not stages_df.empty:
        print("Top bottlenecks (avg seconds):")
        print(stages_df.head(10).to_string(index=False))

    # Return a summary so notebook callers can inspect without reading CSVs
    return {"progress": progress_df, "stages": stages_df}


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
