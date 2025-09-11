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
from src.ri.get_data.transaction_pipeline import get_transaction_data_by_scope, get_transaction_data_by_scope_fast, get_transaction_data_by_scope_fast2, get_transaction_data_by_scope_fast2_cohort_total
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


def _clean_list(seq):
    out, seen = [], set()
    for x in (seq or []):
        s = str(x).strip() if x is not None else ""
        if s and s.lower() != "nan" and s not in seen:
            seen.add(s); out.append(s)
    return out

def preflight_scope(
    *,
    client: bigquery.Client,
    base_table_fq: str,
    sku_list: List[str],
    store_list: List[str],   # [] means ALL STORES
    start_date,
    end_date,
    print_full: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Prints counts/lists for SKUs, Stores, Brands and returns (skus, stores, brands).
    If stores == [], treats as ALL STORES (doesn't filter by store_id in brand check).
    Skips nothing by itself; caller decides based on len(brands).
    """
    skus = _clean_list(sku_list)
    stores = _clean_list(store_list)

    min_d = pd.to_datetime(start_date).date()
    max_d = pd.to_datetime(end_date).date()

    # ---- Brands in scope (store filter only if stores were given) ----
    if stores:
        sql_brands = f"""
        WITH skus AS (
          SELECT DISTINCT id FROM UNNEST(@sku_ids) id
          WHERE id IS NOT NULL AND TRIM(id) <> ''
        ),
        stores AS (
          SELECT DISTINCT CAST(id AS STRING) AS id
          FROM UNNEST(@store_ids) id
          WHERE id IS NOT NULL AND TRIM(CAST(id AS STRING)) <> ''
        )
        SELECT DISTINCT BrandDescription
        FROM `{base_table_fq}`
        WHERE week_start BETWEEN @min_d AND @max_d
          AND CAST(article_id AS STRING) IN (SELECT id FROM skus)
          AND CAST(store_id  AS STRING)  IN (SELECT id FROM stores)
          AND BrandDescription IS NOT NULL AND TRIM(BrandDescription) <> ''
        ORDER BY BrandDescription
        """
        params = [
            bigquery.ArrayQueryParameter("sku_ids", "STRING", skus),
            bigquery.ArrayQueryParameter("store_ids", "STRING", stores),
            bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
            bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
        ]
    else:
        sql_brands = f"""
        WITH skus AS (
          SELECT DISTINCT id FROM UNNEST(@sku_ids) id
          WHERE id IS NOT NULL AND TRIM(id) <> ''
        )
        SELECT DISTINCT BrandDescription
        FROM `{base_table_fq}`
        WHERE week_start BETWEEN @min_d AND @max_d
          AND CAST(article_id AS STRING) IN (SELECT id FROM skus)
          AND BrandDescription IS NOT NULL AND TRIM(BrandDescription) <> ''
        ORDER BY BrandDescription
        """
        params = [
            bigquery.ArrayQueryParameter("sku_ids", "STRING", skus),
            bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
            bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
        ]

    brands_df = client.query(sql_brands, job_config=bigquery.QueryJobConfig(query_parameters=params)).result().to_dataframe()
    brands = brands_df["BrandDescription"].astype(str).tolist()

    # ---- Diagnostics (print counts + truncated lists) ----
    def _print_list(label, items):
        n = len(items)
        head = items if print_full or n <= 200 else (items[:200] + [f"...(+{n-200} more)"])
        print(f"[Preflight] {label} ({n}): {head}")

    _print_list("SKUs", skus)

    if stores:
        _print_list("Stores", stores)
    else:
        # ALL STORES: estimate count quickly (filter by SKUs for relevance)
        sql_store_count = f"""
        WITH skus AS (
          SELECT DISTINCT id FROM UNNEST(@sku_ids) id
          WHERE id IS NOT NULL AND TRIM(id) <> ''
        )
        SELECT COUNT(DISTINCT CAST(store_id AS STRING)) AS n_stores
        FROM `{base_table_fq}`
        WHERE week_start BETWEEN @min_d AND @max_d
          AND CAST(article_id AS STRING) IN (SELECT id FROM skus)
        """
        sc_params = [
            bigquery.ArrayQueryParameter("sku_ids", "STRING", skus),
            bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
            bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
        ]
        n_stores = int(client.query(sql_store_count, job_config=bigquery.QueryJobConfig(query_parameters=sc_params)).result().to_dataframe()["n_stores"].iloc[0] or 0)
        print(f"[Preflight] Stores (ALL STORES): ~{n_stores}")

    _print_list("Brands", brands)

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
    grains: List[str],              # drives conditional brand-subcat expansion (for sku universe logging only)
) -> pd.DataFrame:
    """
    Enforces that promo SKUs belong to EXACTLY ONE brand (0 or >1 => SkipCampaign).
    No category/subcategory-wide expansions. Optionally expands to brand-subcategory
    for visibility when requested via 'grains'.

    Returns cohort×week totals by calling get_transaction_data_by_scope_fast2_cohort_total.
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

    # ---------- Strict brand check: EXACTLY ONE brand among promo SKUs over window ----------
    brand_check_sql = f"""
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
        brand_check_sql, job_config=bigquery.QueryJobConfig(query_parameters=brand_params)
    ).result().to_dataframe()

    n_brands = int(promo_brands_df.shape[0])
    brands_list = promo_brands_df["brand"].astype(str).tolist()
    print(f"[Diag] Promo-SKU brand count: {n_brands} | {brands_list}")

    if n_brands != 1:
        raise ValueError(f"[SkipCampaign] Promo SKUs span {n_brands} brand(s); require exactly 1. Found: {brands_list}")

    # ---------- OPTIONAL: build a small SKU universe for logging (no category-wide expansion) ----------
    wants_brand_sub = any(g.lower() in {"brand","brand_subcategory"} for g in (grains or []))

    with_parts = []
    with_parts.append("""
skus AS (
  SELECT DISTINCT id FROM UNNEST(@skus) AS id
  WHERE id IS NOT NULL AND TRIM(id) <> ''
)""")
    with_parts.append(f"""
sku_only AS (
  SELECT DISTINCT CAST(t.article_id AS STRING) AS article_id
  FROM `{tx_table_fq}` t
  JOIN skus s ON CAST(t.article_id AS STRING) = s.id
  WHERE t.week_start BETWEEN @min_d AND @max_d
)""")

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
    u_params = [
        bigquery.ArrayQueryParameter("skus", "STRING", promo_skus_article_ids),
        bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
        bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
    ]
    print("Getting universe ids...", end=" ", flush=True)
    universe_ids = client.query(
        universe_sql, job_config=bigquery.QueryJobConfig(query_parameters=u_params)
    ).result().to_dataframe()["article_id"].astype(str).tolist()
    print("Done!")
    universe_ids = _clean_list(universe_ids)
    print(f"[Preflight] SKUs ({len(universe_ids)}): {universe_ids[:20]}{' ...' if len(universe_ids)>20 else ''}")
    print(f"[Preflight] Stores ({len(stores_union)}; all_stores={len(stores_union)==0})")

    if not universe_ids:
        raise ValueError("[SkipCampaign] SKU universe empty after expansion/clean.")

    # ---------- Cohort×week TOTALS (global lookbacks, cohort-scoped aggregation) ----------
    print("Begin get_transaction_data_by_scope_fast2_cohort_total ...", end=" ", flush=True)
    df = get_transaction_data_by_scope_fast2_cohort_total(
        client=client,
        base_table_fq=tx_table_fq,
        sku_list=universe_ids,        # pass promo (± brand-subcat) SKUs
        store_list=stores_union,      # [] => all stores (cohort = whole fleet)
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
        cols = set(tx_slice.columns)
        renames = {}

        def pick(candidates, canonical):
            for c in candidates:
                if c in cols:
                    renames[c] = canonical
                    return

        pick(["product_id", "product_number", "article_id", "Article", "ProductNumber"], "product_id")
        pick(["brand", "BrandDescription", "brand_description"], "brand")
        pick(["category", "CategoryDescription", "category_description"], "category")
        pick(["subcategory", "SubCategoryDescription", "Sub_CategoryDescription", "subcategory_description"], "subcategory")

        tx_norm = tx_slice.rename(columns=renames).copy()

        # Ensure all needed columns exist (fill missing with NA so downstream code can decide how to handle)
        for need in ["product_id", "brand", "category", "subcategory"]:
            if need not in tx_norm.columns:
                tx_norm[need] = pd.NA

        product_dim = tx_norm[["product_id", "brand", "category", "subcategory"]].drop_duplicates()
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

# safe tqdm
try:
    from tqdm.auto import tqdm as _tqdm
    HAS_TQDM = True
except Exception:
    _tqdm = None
    HAS_TQDM = False

# default EXOG (kept if user didn't define EXOG elsewhere)
DEFAULT_EXOG = [
    "discountpercent",
    "max_internal_competitor_discount_percent",
    "n_competitors", "n_any_cheaper", "n_shelf_cheaper",
    "n_promo_cheaper_no_hurdle", "n_promo_cheaper_hurdle",
    "avg_cheaper_gap", "worst_gap", "p90_gap",
    "brochure_Not on brochure", "multibuy_Not on Multibuy",
]

def run_pipeline2(
    *,
    media: pd.DataFrame,
    tx_table: str | None,
    tx_parquet: str | None,
    project: str | None,
    outdir: str,
    metrics: List[str],
    grains: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Per-campaign orchestration with:
      • Preflight diagnostics (SKUs / Stores / Brands) and brand guard (exactly 1).
      • Cohort-level totals (no store_id in forecasting path).
      • Parquet path preserved (lazy import).
      • Grains normalized ('brand' -> 'brand_subcategory'; only sku/brand_subcategory modeled).
      • NEW: if calendar is empty (ALL STORES), synthesize campaign-week keys from media dates.
    """
    # Resolve EXOG list
    try:
        EXOG  # may be defined elsewhere
    except NameError:
        EXOG = DEFAULT_EXOG

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "artifacts"), exist_ok=True)

    bq_client = None
    tx_master_df = None
    product_dim_df = None

    print("run_pipeline2 setup...", end=" ", flush=True)
    if tx_table:
        bq_client = bigquery.Client(project=project)
    else:
        if not tx_parquet:
            raise ValueError("Provide either tx_table (BigQuery) or tx_parquet path.")
        tx_master_df = (pd.read_parquet(tx_parquet) if tx_parquet.endswith(".parquet")
                        else pd.read_csv(tx_parquet, parse_dates=["week_start"]))
        product_dim_df = tx_master_df[[
            "product_number","article_id",
            "BrandDescription","CategoryDescription","SubCategoryDescription"
        ]].drop_duplicates()
    print("Done.")

    # ---- sanitize grains
    cleaned_grains: List[str] = []
    for g in (grains or []):
        g = g.strip().lower()
        if g == "brand":
            g = "brand_subcategory"
        if g in {"sku","brand_subcategory"} and g not in cleaned_grains:
            cleaned_grains.append(g)
    if not cleaned_grains:
        cleaned_grains = ["sku","brand_subcategory"]

    bookings = sorted(media["booking_number"].astype(str).unique().tolist())
    iterator = _tqdm(bookings, total=len(bookings), dynamic_ncols=True) if HAS_TQDM else bookings

    from src.ri.get_data.media.parse_media_table import to_list_flexible
    from src.ri.get_data.media.build_weekly_store_calendar import (
        build_weekly_store_calendar_for_campaign,
        weeks_to_store_sets,
    )
    from src.ri.model.orchestration.build_grids_v3 import _make_registry
    from src.ri.model.forecasting.cohort_forecasting import forecast_per_week_dynamic_cohorts

    t0_all = time.time()
    per_stage: Dict[str, List[float]] = defaultdict(list)
    rows_prog: List[Dict[str, object]] = []

    for i, cid in enumerate(iterator, 1):
        print(f"\nbooking_number {cid}")
        t0 = time.time()

        mezzo = media[media["booking_number"].astype(str) == cid].copy()
        if mezzo.empty:
            continue

        # ---- promo SKUs from media
        promo_skus = sorted({s for row in mezzo["sorted_sku_list"].dropna() for s in to_list_flexible(row)})
        if not promo_skus:
            print(f"[Skip] booking {cid}: no promo SKUs found")
            continue

        # ---- dynamic cohorts (empty list => ALL STORES)
        t_cal = time.time()
        raw = mezzo["sorted_store_list"].explode().dropna().tolist() if "sorted_store_list" in mezzo.columns else []
        fleet_input = [str(x).strip() for x in raw if str(x).strip()]
        cal = build_weekly_store_calendar_for_campaign(mezzo, fleet_input)
        print(f"{cid} Calendar: ")
        print(cal)
        wk_to_stores = weeks_to_store_sets(cal)
        print("wk_to_stores")
        print(wk_to_stores)
        per_stage["calendar"].append(time.time() - t_cal)

        # ---- media window (weekly aligned)
        min_start = pd.to_datetime(mezzo["media_start_date"]).min()
        max_end   = pd.to_datetime(mezzo["media_end_date"]).max()
        # campaign weeks (inclusive)
        print(f"Campaign Start: {min_start}")
        print(f"Campaign End: {max_end}")
        start_w = pd.Period(min_start, freq="W-WED").start_time
        end_w   = pd.Period(max_end,   freq="W-WED").start_time
        campaign_weeks = list(pd.period_range(start=start_w, end=end_w, freq="W-WED").start_time)
        print(f"Campaign Weeks: {campaign_weeks}")
        # NEW: if no per-week store sets (ALL STORES), synthesize a week→None mapping
        if not wk_to_stores:
            wk_to_stores = {wk: None for wk in campaign_weeks}
            stores_union = []  # all stores in SQL
            print(f"[Calendar] ALL STORES; synthesized {len(campaign_weeks)} campaign weeks from media dates.")
        else:
            stores_union = sorted({sid for s in wk_to_stores.values() for sid in (s or [])})

        # ---- 13m lookback + 2w pad (data pull window)
        min_d = (min_start - pd.DateOffset(months=13)).to_period("W-WED").start_time
        max_d = (max_end   + pd.Timedelta(weeks=2)).to_period("W-WED").start_time

        # ---- PREFLIGHT (prints + brand guard)
        if bq_client is not None:
            skus_pf, stores_pf, brands_pf = preflight_scope(
                client=bq_client,
                base_table_fq=tx_table,
                sku_list=promo_skus,
                store_list=stores_union,   # [] => ALL STORES
                start_date=min_d,
                end_date=max_d,
                print_full=True,
            )
            n_brands = len(brands_pf)
            if n_brands != 1:
                print(f"[Skip] booking {cid}: promo SKUs span {n_brands} brands: {brands_pf}")
                continue

        # ---------- Fetch campaign slice (COHORT TOTALS on BQ; legacy parquet preserved) ----------
        t_fetch = time.time()
        if bq_client is not None:
            from src.ri.get_data.transaction_pipeline import get_transaction_data_by_scope_fast2_cohort_total
            tx_slice = get_transaction_data_by_scope_fast2_cohort_total(
                client=bq_client,
                base_table_fq=tx_table,
                sku_list=promo_skus,
                store_list=stores_union,  # [] => all stores (cohort = whole fleet)
                start_date=min_d, end_date=max_d,
            )
            tx_slice_snippet = tx_slice.head(10)
            print("Transactions Snippet")
            print(tx_slice_snippet)
        else:
            try:
                from src.ri.get_data.transaction_pipeline import fetch_campaign_tx_slice_parquet as _fetch_parquet
            except Exception as e:
                raise ImportError(
                    "Parquet path requested but 'fetch_campaign_tx_slice_parquet' is not available in "
                    "src.ri.get_data.transaction_pipeline. Provide tx_table (BQ) or implement the helper."
                ) from e
            tx_slice = _fetch_parquet(
                tx_master_df=tx_master_df,
                product_dim_df=product_dim_df,
                promo_skus_article_ids=promo_skus,
                stores_union=stores_union if stores_union else list(tx_master_df["store_id"].astype(str).unique()),
                min_date=min_d, max_date=max_d
            )
        per_stage["fetch_slice"].append(time.time() - t_fetch)
        if tx_slice.empty:
            print(f"[Skip] booking {cid}: tx_slice empty")
            continue

        # ---------- Canonicalize columns ----------
        cols = set(tx_slice.columns)
        renames = {}
        def pick(cands, canonical):
            for c in cands:
                if c in cols:
                    renames[c] = canonical; return
        pick(["product_id","product_number","article_id","Article","ProductNumber"], "product_id")
        pick(["brand","BrandDescription","brand_description"], "brand")
        pick(["category","CategoryDescription","category_description"], "category")
        pick(["subcategory","SubCategoryDescription","Sub_CategoryDescription","subcategory_description"], "subcategory")
        tx_norm = tx_slice.rename(columns=renames).copy()
        tx_norm_snippet = tx_norm.head(10)
        print("Transactions Normalised")
        print(tx_norm_snippet)
        for need in ["product_id","brand","category","subcategory"]:
            if need not in tx_norm.columns:
                tx_norm[need] = pd.NA

        # ---------- Registry (synthesize product_dim if cohort removed product_id) ----------
        if tx_norm["product_id"].isna().all():
            b = tx_norm["brand"].dropna().iloc[0] if tx_norm["brand"].notna().any() else None
            c = tx_norm["category"].dropna().iloc[0] if tx_norm["category"].notna().any() else None
            s = tx_norm["subcategory"].dropna().iloc[0] if tx_norm["subcategory"].notna().any() else None
            product_dim = pd.DataFrame({
                "product_id": promo_skus,
                "brand": [b]*len(promo_skus),
                "category": [c]*len(promo_skus),
                "subcategory": [s]*len(promo_skus),
            })
            print(product_dim.head(5))
        else:
            product_dim = tx_norm[["product_id","brand","category","subcategory"]].drop_duplicates()

        registry = _make_registry(promo_skus, product_dim)
        print("Registry: ")
        print(registry)
        # ---------- Training cutoff ----------
        train_end = (pd.to_datetime(mezzo["media_start_date"]).min() - pd.Timedelta(days=1))
        print("Train End: ")
        print(train_end)
        # ---------- Targets ----------
        full_metrics = [
            "sales","shoppers",
            "new_to_sku_sales","new_to_sku_shoppers",
            "new_to_brand_sales","new_to_brand_shoppers",
        ]
        targets = (metrics if metrics != ["all"] else full_metrics)
        print("Targets: ")
        print(targets)
        # ---------- Forecast (use tx_norm!) ----------
        for tgt in targets:
            for grain in cleaned_grains:
                print(f"Target: {tgt}, Grain: {grain}")
                nodes = registry.get(grain, [])
                if not nodes:
                    print("!!!! No nodes!")
                    continue
                t_fc = time.time()
                cmp_df = forecast_per_week_dynamic_cohorts(
                    tx_master_df=tx_norm,
                    week_to_storelist=wk_to_stores,  # now guaranteed to have campaign week keys
                    nodes=nodes,
                    train_end_date=train_end,
                    target=tgt,
                    exog_regressors=EXOG
                )
                print("cmp_df: ")
                print(cmp_df.head(5))
                out_dir = os.path.join(outdir, "artifacts", cid)
                os.makedirs(out_dir, exist_ok=True)
                cmp_df.to_parquet(os.path.join(out_dir, f"compare_{tgt}_{grain}.parquet"), index=False)
                per_stage[f"fc_{tgt}@{grain}"].append(time.time() - t_fc)

        dur = time.time() - t0
        rows_prog.append({"campaign_id": cid, "seconds_total": round(dur,2)})
        if HAS_TQDM:
            avg = (time.time() - t0_all) / max(1,i)
            remain = (len(bookings) - i) * avg
            iterator.set_postfix_str(f"{dur/60:.1f}m | avg {avg/60:.1f}m/c | ~rem {remain/60:.1f}m")

    # ---------- Progress / bottlenecks ----------
    progress_df = pd.DataFrame(rows_prog)
    pd.DataFrame(rows_prog).to_csv(os.path.join(outdir, "progress_campaigns.csv"), index=False)

    stage_rows = [{"stage":k, "avg_seconds": (sum(v)/len(v)), "n": len(v)} for k,v in per_stage.items() if v]
    stages_df = pd.DataFrame(stage_rows).sort_values("avg_seconds", ascending=False) if stage_rows else pd.DataFrame(columns=["stage","avg_seconds","n"])
    stages_df.to_csv(os.path.join(outdir, "progress_stages.csv"), index=False)

    if HAS_TQDM and not stages_df.empty:
        print("Top bottlenecks (avg seconds):")
        print(stages_df.head(10).to_string(index=False))

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
