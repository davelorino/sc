from __future__ import annotations
from typing import List, Optional
import pandas as pd
from google.cloud import bigquery

PROJECT = "cart-dai-sandbox-nonprod-c3e7"
DATASET = "dlorino"

def _digits_only_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def create_weekly_transactions_master_table(client: Optional[bigquery.Client] = None) -> pd.DataFrame:
    print("Test 1")
    if client is None:
        client = bigquery.Client(project=PROJECT)

    sql = f"""
    -- Lean weekly base (no customer-level / new_to_* fields)
    CREATE OR REPLACE TABLE `cart-dai-sandbox-nonprod-c3e7.dlorino.sku_store_week_sales_base`
    PARTITION BY week_start
    CLUSTER BY product_number, store_id AS
    WITH
    cust_base AS (
      SELECT
        DATE_TRUNC(BusinessDate, WEEK(Wednesday)) AS week_start,
        SiteNumber                                  AS store_id,
        PriceFamilyCode,
        ProductNumber                               AS product_number,
        Article                                     AS article_id,
        BrandDescription,
        CategoryDescription,
        SubCategoryDescription,
        TotalAmountIncludingGST                     AS sales_amount
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v`
      WHERE SalesOrganisation = '1005'
        AND TotalAmountIncludingGST > 0
        AND LOWER(Channel) = 'in store'
        AND BusinessDate >= DATE_SUB(CURRENT_DATE("Australia/Sydney"), INTERVAL 3 YEAR)
    ),

    step_one AS (
      SELECT
        week_start, store_id, PriceFamilyCode, product_number, article_id,
        BrandDescription, CategoryDescription, SubCategoryDescription,
        SUM(sales_amount) AS sales
      FROM cust_base
      GROUP BY 1,2,3,4,5,6,7,8
    ),

    -- competitor metrics (unchanged)
    comp_base AS (
      SELECT
        ProdNbr                                        AS product_number,
        DATE_TRUNC(FWEndDate, WEEK(Wednesday))         AS week_start,
        Competitor,
        CompStdPrice                                   AS shelf_unit,
        SAFE_DIVIDE(CompOffer1Price, NULLIF(CompOffer1Qty,0)) AS promo1_unit,
        CASE WHEN CompOffer1Qty IS NULL OR CompOffer1Qty=1 THEN 1 ELSE CompOffer1Qty END AS promo1_qty,
        SAFE_DIVIDE(CompOffer2Price, NULLIF(CompOffer2Qty,0)) AS promo2_unit,
        CASE WHEN CompOffer2Qty IS NULL OR CompOffer2Qty=1 THEN 1 ELSE CompOffer2Qty END AS promo2_qty
      FROM `gcp-wow-wiq-price-cpi-prod.bb_cpi_smkt_user_view.Comp_Price_Index_V`
      WHERE FWEndDate >= DATE_SUB(CURRENT_DATE("Australia/Sydney"), INTERVAL 3 YEAR)
    ),
    comp_best AS (
      SELECT
        product_number, week_start, Competitor,
        ARRAY_AGG(STRUCT(unit,qty) ORDER BY unit)[OFFSET(0)].unit AS best_unit_price,
        ARRAY_AGG(STRUCT(unit,qty) ORDER BY unit)[OFFSET(0)].qty  AS hurdle_qty,
        MIN(shelf_unit)                                           AS shelf_unit
      FROM (
        SELECT product_number, week_start, Competitor, shelf_unit, unit, qty
        FROM comp_base,
        UNNEST([
          STRUCT(shelf_unit AS unit, 1 AS qty),
          STRUCT(promo1_unit AS unit, promo1_qty AS qty),
          STRUCT(promo2_unit AS unit, promo2_qty AS qty)
        ]) opt
        WHERE unit IS NOT NULL
      )
      GROUP BY product_number, week_start, Competitor
    ),
    our_price AS (
      SELECT
        ProdNbr                                        AS product_number,
        DATE_TRUNC(FWEndDate, WEEK(Wednesday))         AS week_start,
        WOWStdPrice                                    AS our_shelf_price
      FROM `gcp-wow-wiq-price-cpi-prod.bb_cpi_smkt_user_view.Comp_Price_Index_V`
      WHERE FWEndDate >= DATE_SUB(CURRENT_DATE("Australia/Sydney"), INTERVAL 3 YEAR)
      GROUP BY product_number, week_start, our_shelf_price
    ),
    competitor_metrics AS (
      SELECT
        c.product_number, c.week_start,
        COUNT(*)                                                 AS n_competitors,
        COUNTIF(c.best_unit_price < o.our_shelf_price)           AS n_any_cheaper,
        COUNTIF(c.best_unit_price < o.our_shelf_price AND c.hurdle_qty = 1 AND c.best_unit_price = c.shelf_unit)
                                                                AS n_shelf_cheaper,
        COUNTIF(c.best_unit_price < o.our_shelf_price AND c.hurdle_qty = 1 AND c.best_unit_price <> c.shelf_unit)
                                                                AS n_promo_cheaper_no_hurdle,
        COUNTIF(c.best_unit_price < o.our_shelf_price AND c.hurdle_qty > 1)
                                                                AS n_promo_cheaper_hurdle,
        AVG(IF(c.best_unit_price < o.our_shelf_price, c.best_unit_price - o.our_shelf_price, NULL))
                                                                AS avg_cheaper_gap,
        MIN(IF(c.best_unit_price < o.our_shelf_price, c.best_unit_price - o.our_shelf_price, NULL))
                                                                AS worst_gap,
        APPROX_QUANTILES(
          IF(c.best_unit_price < o.our_shelf_price, c.best_unit_price - o.our_shelf_price, NULL), 11
        )[OFFSET(1)]                                             AS p90_gap
      FROM comp_best c
      JOIN our_price o USING (product_number, week_start)
      GROUP BY c.product_number, c.week_start
    ),

    promo AS (
      SELECT
        pwstartdate      AS week_start,
        petpricefamily   AS PriceFamilyCode,
        discountpercent,
        brochure,
        MBTrigger        AS multibuy
      FROM `gcp-wow-ent-im-tbl-prod.adp_dm_quantium_inbound_view_smkt.qtm_smkt_pet_incrementality_v3_v`
      WHERE SalesDistrict = 'National'
        AND FutureFlag    = 'PAST'
    )

    SELECT
      s.week_start, s.store_id, s.PriceFamilyCode,
      s.BrandDescription, s.CategoryDescription, s.SubCategoryDescription,
      s.product_number, s.article_id,
      COALESCE(p.discountpercent, 0)              AS discountpercent,
      COALESCE(p.multibuy, 'Not on Multibuy')     AS multibuy,
      COALESCE(p.brochure, 'Not on brochure')     AS brochure,
      s.sales,
      COALESCE(cm.n_competitors, 0)                    AS n_competitors,
      COALESCE(cm.n_any_cheaper, 0)                    AS n_any_cheaper,
      COALESCE(cm.n_shelf_cheaper, 0)                  AS n_shelf_cheaper,
      COALESCE(cm.n_promo_cheaper_no_hurdle, 0)        AS n_promo_cheaper_no_hurdle,
      COALESCE(cm.n_promo_cheaper_hurdle, 0)           AS n_promo_cheaper_hurdle,
      COALESCE(cm.avg_cheaper_gap, 0)                  AS avg_cheaper_gap,
      COALESCE(cm.worst_gap, 0)                        AS worst_gap,
      COALESCE(cm.p90_gap, 0)                          AS p90_gap
    FROM step_one s
    LEFT JOIN promo p
      ON p.week_start = s.week_start AND p.PriceFamilyCode = s.PriceFamilyCode
    LEFT JOIN competitor_metrics cm
      ON cm.product_number = s.product_number AND cm.week_start = s.week_start
    ;
    """
    job = client.query(sql)
    _ = job.result()
    return pd.DataFrame({"status": ["ok"]})

def get_transaction_data_by_scope(
    client: Optional[bigquery.Client],
    base_table_fq: str,           # e.g. project.dataset.sku_store_week_sales_base (or old _pn2)
    sku_list: List[str],          # ARTICLE IDs (strings)
    store_list: List[str],        # store_ids (strings)
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    SSOT: Weekly SKU×store with sales, exogenous, CRN-safe shoppers, and all new_to_* metrics.
    Rewritten to avoid correlated subqueries by using JOINs to scope key tables.
    """
    if client is None:
        client = bigquery.Client()

    from datetime import date
    min_d = pd.to_datetime(start_date).date() if start_date else date(1900,1,1)
    max_d = pd.to_datetime(end_date).date()   if end_date   else date(2100,1,1)

    params = [
        bigquery.ArrayQueryParameter("sku_ids",   "STRING", [str(x) for x in sku_list]),
        bigquery.ArrayQueryParameter("store_ids", "STRING", [str(x) for x in store_list]),
        bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
        bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
    ]
    print("Getting transactions...", end=" ", flush=True)
    sql = f"""
    DECLARE lookback_13m_start DATE DEFAULT DATE_SUB(@min_d, INTERVAL 400 DAY);

    WITH
    skus   AS (SELECT id FROM UNNEST(@sku_ids)   AS id),
    stores AS (SELECT id FROM UNNEST(@store_ids) AS id),

    -- 1) Weekly base slice (cheap, pre-aggregated)
    base AS (
      SELECT
        t.week_start,
        CAST(t.store_id AS STRING)           AS store_id,
        t.PriceFamilyCode,
        t.product_number,
        CAST(t.article_id AS STRING)         AS article_id,
        t.BrandDescription, t.CategoryDescription, t.SubCategoryDescription,
        t.discountpercent, t.multibuy, t.brochure,
        t.sales,
        t.n_competitors, t.n_any_cheaper, t.n_shelf_cheaper,
        t.n_promo_cheaper_no_hurdle, t.n_promo_cheaper_hurdle,
        t.avg_cheaper_gap, t.worst_gap, t.p90_gap
      FROM `{base_table_fq}` t
      JOIN skus   s  ON CAST(t.article_id AS STRING) = s.id
      JOIN stores st ON CAST(t.store_id  AS STRING)  = st.id
      WHERE t.week_start BETWEEN @min_d AND @max_d
    ),

    -- keys for internal competitor computation (no CB scan)
    pf_by_week_store_brandcat AS (
      SELECT DISTINCT week_start, store_id, BrandDescription, CategoryDescription, PriceFamilyCode
      FROM base
    ),
    promo_pf AS (
      SELECT
        pwstartdate AS week_start,
        petpricefamily AS PriceFamilyCode,
        discountpercent,
        brochure,
        MBTrigger AS multibuy
      FROM `gcp-wow-ent-im-tbl-prod.adp_dm_quantium_inbound_view_smkt.qtm_smkt_pet_incrementality_v3_v`
      WHERE SalesDistrict='National'
        AND FutureFlag='PAST'
        AND pwstartdate BETWEEN @min_d AND @max_d
    ),
    mcd_agg AS (
      SELECT
        b.week_start, b.store_id, b.BrandDescription, b.CategoryDescription,
        MAX(CASE WHEN LOWER(p.brochure) LIKE '%not%' OR p.brochure IS NULL THEN 0 ELSE 1 END)
          AS internal_competitor_brochure,
        MAX(CASE WHEN LOWER(p.multibuy)  LIKE '%not%' OR p.multibuy  IS NULL THEN 0 ELSE 1 END)
          AS internal_competitor_multibuy,
        MAX(IFNULL(p.discountpercent,0.0)) AS max_internal_competitor_discount_percent
      FROM pf_by_week_store_brandcat b
      LEFT JOIN promo_pf p
        ON p.week_start = b.week_start AND p.PriceFamilyCode = b.PriceFamilyCode
      GROUP BY 1,2,3,4
    ),

    -- 2) Raw transactions in scope (+lookback), with CRN sanitization
    target_txn AS (
      SELECT
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday))  AS week_start,
        DATE(cb.BusinessDate)                         AS BusinessDate,
        CAST(cb.SiteNumber AS STRING)                 AS store_id,
        cb.PriceFamilyCode,
        cb.ProductNumber                              AS product_number,
        CAST(cb.Article AS STRING)                    AS article_id,
        cb.BrandDescription, cb.CategoryDescription, cb.SubCategoryDescription,
        CASE
          WHEN cb.CRN IS NOT NULL
          AND REGEXP_CONTAINS(TRIM(CAST(cb.CRN AS STRING)), '^[0-9]{{3,}}$')
          THEN TRIM(CAST(cb.CRN AS STRING))
          ELSE NULL
        END                                           AS crn_valid,
        cb.TotalAmountIncludingGST                    AS sales_amount
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN skus   s  ON CAST(cb.Article AS STRING)   = s.id
      JOIN stores st ON CAST(cb.SiteNumber AS STRING)= st.id
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.BusinessDate BETWEEN lookback_13m_start AND @max_d
    ),

    -- 3) Keys for 13m lookback (valid CRNs only; brand & sku only)
    keys_sku   AS (SELECT DISTINCT crn_valid, product_number   FROM target_txn WHERE crn_valid IS NOT NULL),
    keys_brand AS (SELECT DISTINCT crn_valid, BrandDescription FROM target_txn WHERE crn_valid IS NOT NULL),

    -- 4) 13m history: prior dates per (crn, dim)
    hist_sku AS (
      SELECT DATE(cb.BusinessDate) AS BusinessDate,
            TRIM(CAST(cb.CRN AS STRING)) AS crn_valid,
            cb.ProductNumber AS product_number
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN keys_sku k
        ON TRIM(CAST(cb.CRN AS STRING)) = k.crn_valid
      AND cb.ProductNumber = k.product_number
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.BusinessDate BETWEEN lookback_13m_start AND @max_d
    ),
    hist_brand AS (
      SELECT DATE(cb.BusinessDate) AS BusinessDate,
            TRIM(CAST(cb.CRN AS STRING)) AS crn_valid,
            cb.BrandDescription AS BrandDescription
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN keys_brand k
        ON TRIM(CAST(cb.CRN AS STRING)) = k.crn_valid
      AND cb.BrandDescription = k.BrandDescription
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.BusinessDate BETWEEN lookback_13m_start AND @max_d
    ),
    prev_sku AS (
      SELECT *,
            LAG(BusinessDate) OVER (PARTITION BY crn_valid, product_number ORDER BY BusinessDate) AS prev_dt
      FROM hist_sku
    ),
    prev_brand AS (
      SELECT *,
            LAG(BusinessDate) OVER (PARTITION BY crn_valid, BrandDescription ORDER BY BusinessDate) AS prev_dt
      FROM hist_brand
    ),

    -- 5) Per-transaction flags (13m only; invalid CRNs never flagged)
    flags AS (
      SELECT
        t.week_start, t.BusinessDate, t.store_id, t.PriceFamilyCode, t.product_number, t.article_id,
        t.BrandDescription, t.CategoryDescription, t.SubCategoryDescription,
        t.crn_valid, t.sales_amount,

        -- 13m new-to-SKU / new-to-Brand
        IF(t.crn_valid IS NULL, FALSE,
          (ps.prev_dt IS NULL OR DATE_DIFF(t.BusinessDate, ps.prev_dt, MONTH) >= 13)) AS is_new_to_sku_13m,
        IF(t.crn_valid IS NULL, FALSE,
          (pb.prev_dt IS NULL OR DATE_DIFF(t.BusinessDate, pb.prev_dt, MONTH) >= 13)) AS is_new_to_brand_13m
      FROM target_txn t
      LEFT JOIN prev_sku   ps ON ps.crn_valid = t.crn_valid AND ps.product_number   = t.product_number   AND ps.BusinessDate = t.BusinessDate
      LEFT JOIN prev_brand pb ON pb.crn_valid = t.crn_valid AND pb.BrandDescription = t.BrandDescription AND pb.BusinessDate = t.BusinessDate
      WHERE t.BusinessDate BETWEEN @min_d AND @max_d
    ),

    -- 6) Weekly aggregates (valid CRNs only for counts)
    customer_weekly AS (
      SELECT
        week_start, store_id, PriceFamilyCode, product_number, article_id,
        BrandDescription, CategoryDescription, SubCategoryDescription,

        COUNT(DISTINCT crn_valid)                                      AS shoppers,

        COUNT(DISTINCT IF(is_new_to_sku_13m,   crn_valid, NULL))       AS new_to_sku_shoppers_13m,
        SUM(           IF(is_new_to_sku_13m,   sales_amount, 0.0))     AS new_to_sku_sales_13m,

        COUNT(DISTINCT IF(is_new_to_brand_13m, crn_valid, NULL))       AS new_to_brand_shoppers_13m,
        SUM(           IF(is_new_to_brand_13m, sales_amount, 0.0))     AS new_to_brand_sales_13m
      FROM flags
      GROUP BY 1,2,3,4,5,6,7,8
    )

    -- 7) Final join back to base & internal competitor (from base+promo)
    SELECT
      b.week_start, b.store_id, b.PriceFamilyCode,
      b.product_number, b.article_id,
      b.BrandDescription, b.CategoryDescription, b.SubCategoryDescription,
      b.discountpercent, b.multibuy, b.brochure,
      b.sales,

      COALESCE(cw.shoppers, 0)                    AS shoppers,

      COALESCE(cw.new_to_sku_shoppers_13m, 0)     AS new_to_sku_shoppers_13m,
      COALESCE(cw.new_to_sku_sales_13m,   0.0)    AS new_to_sku_sales_13m,

      COALESCE(cw.new_to_brand_shoppers_13m, 0)   AS new_to_brand_shoppers_13m,
      COALESCE(cw.new_to_brand_sales_13m,   0.0)  AS new_to_brand_sales_13m,

      b.n_competitors, b.n_any_cheaper, b.n_shelf_cheaper,
      b.n_promo_cheaper_no_hurdle, b.n_promo_cheaper_hurdle,
      b.avg_cheaper_gap, b.worst_gap, b.p90_gap,

      COALESCE(mi.internal_competitor_brochure, 0)               AS internal_competitor_brochure,
      COALESCE(mi.internal_competitor_multibuy, 0)               AS internal_competitor_multibuy,
      COALESCE(mi.max_internal_competitor_discount_percent, 0.0) AS max_internal_competitor_discount_percent

    FROM base b
    LEFT JOIN customer_weekly cw
      ON cw.week_start=b.week_start AND cw.store_id=b.store_id
    AND cw.product_number=b.product_number AND cw.article_id=b.article_id
    AND cw.PriceFamilyCode=b.PriceFamilyCode
    AND cw.BrandDescription=b.BrandDescription
    AND cw.CategoryDescription=b.CategoryDescription
    AND cw.SubCategoryDescription=b.SubCategoryDescription
    LEFT JOIN mcd_agg mi
      ON mi.week_start=b.week_start AND mi.store_id=b.store_id
    AND mi.BrandDescription=b.BrandDescription AND mi.CategoryDescription=b.CategoryDescription
    #ORDER BY b.week_start, b.store_id, b.product_number
    """

    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    tx_df = job.result().to_dataframe()
    print("Done!")
    return tx_df

def get_transaction_data_by_scope_fast(
    client: Optional[bigquery.Client],
    base_table_fq: str,           # e.g. project.dataset.sku_store_week_sales_base
    sku_list: List[str],          # ARTICLE IDs (strings) -- keep as strings if Article is STRING
    store_list: List[str],        # store_ids (strings)  -- change to ints if store_id is INT64
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if client is None:
        client = bigquery.Client()

    from datetime import date

    # --- minimal preflight: sanitize lists (dedupe, drop blanks/None/'nan') ---
    def _clean(vals):
        out, seen = [], set()
        for v in (vals or []):
            s = str(v).strip() if v is not None else ""
            if s and s.lower() != "nan" and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    sku_ids_clean   = _clean(sku_list)
    store_ids_clean = _clean(store_list)

    if not sku_ids_clean:
        raise ValueError("Sanitized SKU list is empty. Check your media SKU inputs.")
    if not store_ids_clean:
        raise ValueError("Sanitized store list is empty. Check your media store inputs.")

    min_d = pd.to_datetime(start_date).date() if start_date else date(1900,1,1)
    max_d = pd.to_datetime(end_date).date()   if end_date   else date(2100,1,1)

    # --- preflight: derive Brand list in-scope from base, then print SKU + Brand lists ---
    pre_sql = f"""
    WITH
      skus AS (
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
    WHERE week_start BETWEEN @min_d AND @max_d
      AND CAST(article_id AS STRING) IN (SELECT id FROM skus)
      AND CAST(store_id  AS STRING)  IN (SELECT id FROM stores)
      AND BrandDescription IS NOT NULL AND TRIM(BrandDescription) <> ''
    ORDER BY BrandDescription
    """
    pre_params = [
        bigquery.ArrayQueryParameter("sku_ids",   "STRING", sku_ids_clean),
        bigquery.ArrayQueryParameter("store_ids", "STRING", store_ids_clean),
        bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
        bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
    ]
    brands_in_scope = (
        client.query(pre_sql, job_config=bigquery.QueryJobConfig(query_parameters=pre_params))
        .result().to_dataframe()["BrandDescription"].astype(str).tolist()
    )

    # Print FULL lists as requested
    print(f"SKUs ({len(sku_ids_clean)}): {sku_ids_clean}")
    print(f"Brands ({len(brands_in_scope)}): {brands_in_scope}")

    if not brands_in_scope:
        raise ValueError("Derived Brand list is empty for this scope (stores × skus × window).")

    # --- original param build, but using sanitized lists ---
    # Prefer INT64 for stores if the column is INT64 in BQ
    store_ids_native = []
    try:
        store_ids_native = [int(x) for x in store_ids_clean]
        store_param_type = "INT64"
    except Exception:
        store_ids_native = [str(x) for x in store_ids_clean]
        store_param_type = "STRING"

    params = [
        bigquery.ArrayQueryParameter("sku_ids",   "STRING", [str(x) for x in sku_ids_clean]),
        bigquery.ArrayQueryParameter("store_ids", store_param_type, store_ids_native),
        bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
        bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
    ]

    # --- same SQL as your version, with only three small changes:
    #     (1) skus/stores CTEs now DISTINCT + trim non-blanks
    #     (2) brands CTE excludes null/blank
    #     (3) everything else left as-is
    sql = f"""
    DECLARE lookback_13m_start DATE DEFAULT DATE_SUB(@min_d, INTERVAL 400 DAY);

    -- Small in-memory lists (sanitized & deduped)
    WITH
    skus   AS (
      SELECT DISTINCT id
      FROM UNNEST(@sku_ids) AS id
      WHERE id IS NOT NULL AND TRIM(id) <> ''
    ),
    stores AS (
      SELECT DISTINCT CAST(id AS STRING) AS id
      FROM UNNEST(@store_ids) AS id
      WHERE id IS NOT NULL AND TRIM(CAST(id AS STRING)) <> ''
    ),

    /* 1) Slice of your pre-aggregated weekly base */
    base AS (
      SELECT
        t.week_start,
        t.store_id,
        t.PriceFamilyCode,
        t.product_number,
        t.article_id,
        t.BrandDescription, t.CategoryDescription, t.SubCategoryDescription,
        t.discountpercent, t.multibuy, t.brochure,
        t.sales,
        t.n_competitors, t.n_any_cheaper, t.n_shelf_cheaper,
        t.n_promo_cheaper_no_hurdle, t.n_promo_cheaper_hurdle,
        t.avg_cheaper_gap, t.worst_gap, t.p90_gap
      FROM `{base_table_fq}` t
      JOIN skus   s  ON CAST(t.article_id AS STRING) = s.id
      JOIN stores st ON CAST(t.store_id  AS STRING)  = CAST(st.id AS STRING)  -- cast once to match type
      WHERE t.week_start BETWEEN @min_d AND @max_d
    ),

    /* 2) Internal-competitor features keyed by (week, store, brand, category) */
    pf_by_w_s_bcat AS (
      SELECT DISTINCT week_start, store_id, BrandDescription, CategoryDescription, PriceFamilyCode
      FROM base
    ),
    promo_pf AS (
      SELECT
        pwstartdate    AS week_start,
        petpricefamily AS PriceFamilyCode,
        discountpercent,
        brochure,
        MBTrigger      AS multibuy
      FROM `gcp-wow-ent-im-tbl-prod.adp_dm_quantium_inbound_view_smkt.qtm_smkt_pet_incrementality_v3_v`
      WHERE SalesDistrict='National'
        AND FutureFlag='PAST'
        AND pwstartdate BETWEEN @min_d AND @max_d
    ),
    mcd_agg AS (
      SELECT
        b.week_start, b.store_id, b.BrandDescription, b.CategoryDescription,
        MAX(CASE WHEN LOWER(p.brochure) LIKE '%not%' OR p.brochure IS NULL THEN 0 ELSE 1 END)
          AS internal_competitor_brochure,
        MAX(CASE WHEN LOWER(p.multibuy)  LIKE '%not%' OR p.multibuy  IS NULL THEN 0 ELSE 1 END)
          AS internal_competitor_multibuy,
        MAX(IFNULL(p.discountpercent,0.0)) AS max_internal_competitor_discount_percent
      FROM pf_by_w_s_bcat b
      LEFT JOIN promo_pf p
        ON p.week_start = b.week_start AND p.PriceFamilyCode = b.PriceFamilyCode
      GROUP BY 1,2,3,4
    ),

    /* 3) For shoppers counts (exact), we only need transactions in (stores × skus) for [min_d, max_d] */
    txn_scope AS (
      SELECT
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday))  AS week_start,
        CAST(cb.SiteNumber AS STRING)                 AS store_id,
        cb.ProductNumber                              AS product_number,
        CAST(cb.Article AS STRING)                    AS article_id,
        cb.BrandDescription, cb.CategoryDescription, cb.SubCategoryDescription,
        CASE
          WHEN cb.CRN IS NOT NULL
           AND REGEXP_CONTAINS(TRIM(CAST(cb.CRN AS STRING)), r'^[0-9]{{3,}}$')
          THEN TRIM(CAST(cb.CRN AS STRING))
        END AS crn_valid,
        cb.TotalAmountIncludingGST AS sales_amount,
        DATE(cb.BusinessDate)      AS d
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN skus   s  ON CAST(cb.Article AS STRING) = s.id
      JOIN stores st ON CAST(cb.SiteNumber AS STRING) = CAST(st.id AS STRING)
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.BusinessDate BETWEEN @min_d AND @max_d
    ),
    shoppers_weekly AS (
      SELECT
        week_start, store_id, product_number, article_id,
        BrandDescription, CategoryDescription, SubCategoryDescription,
        COUNT(DISTINCT crn_valid) AS shoppers
      FROM txn_scope
      WHERE crn_valid IS NOT NULL
      GROUP BY 1,2,3,4,5,6,7
    ),

    /* 4) First purchase within extended window, across ALL stores, but restricted to relevant SKUs/brands */
    -- SKU path
    tx_sku_window AS (
      SELECT
        DATE(cb.BusinessDate)                          AS d,
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday))   AS week_start,
        CAST(cb.SiteNumber AS STRING)                  AS store_id,
        cb.ProductNumber                               AS product_number,
        CAST(cb.Article AS STRING)                     AS article_id,
        cb.BrandDescription, cb.CategoryDescription, cb.SubCategoryDescription,
        CASE
          WHEN cb.CRN IS NOT NULL
           AND REGEXP_CONTAINS(TRIM(CAST(cb.CRN AS STRING)), r'^[0-9]{{3,}}$')
          THEN TRIM(CAST(cb.CRN AS STRING))
        END AS crn_valid,
        cb.TotalAmountIncludingGST AS sales_amount
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN skus s ON CAST(cb.Article AS STRING) = s.id      -- restrict to SKUs of interest, but no store filter here
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.BusinessDate BETWEEN lookback_13m_start AND @max_d
    ),
    first_date_sku AS (
      SELECT crn_valid, product_number, MIN(d) AS first_date
      FROM tx_sku_window
      WHERE crn_valid IS NOT NULL
      GROUP BY 1,2
    ),
    first_txn_sku AS (
      -- keep ALL first-day rows (handles multi-store same-day purchases)
      SELECT t.*
      FROM tx_sku_window t
      JOIN first_date_sku f
        ON f.crn_valid = t.crn_valid
       AND f.product_number = t.product_number
       AND f.first_date = t.d
    ),
    new_to_sku_weekly AS (
      SELECT
        DATE_TRUNC(d, WEEK(Wednesday)) AS week_start,
        store_id, product_number, article_id,
        BrandDescription, CategoryDescription, SubCategoryDescription,
        COUNT(DISTINCT crn_valid) AS new_to_sku_shoppers_13m,
        SUM(sales_amount)         AS new_to_sku_sales_13m
      FROM first_txn_sku
      JOIN stores st ON CAST(first_txn_sku.store_id AS STRING) = CAST(st.id AS STRING)
      WHERE d BETWEEN @min_d AND @max_d
      GROUP BY 1,2,3,4,5,6,7
    ),

    -- BRAND path (only brands in our base slice; exclude blanks)
    brands AS (
      SELECT DISTINCT BrandDescription
      FROM base
      WHERE BrandDescription IS NOT NULL AND TRIM(BrandDescription) <> ''
    ),
    tx_brand_window AS (
      SELECT
        DATE(cb.BusinessDate)                        AS d,
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday)) AS week_start,
        CAST(cb.SiteNumber AS STRING)                AS store_id,
        cb.ProductNumber                             AS product_number,
        CAST(cb.Article AS STRING)                   AS article_id,
        cb.BrandDescription, cb.CategoryDescription, cb.SubCategoryDescription,
        CASE
          WHEN cb.CRN IS NOT NULL
           AND REGEXP_CONTAINS(TRIM(CAST(cb.CRN AS STRING)), r'^[0-9]{{3,}}$')
          THEN TRIM(CAST(cb.CRN AS STRING))
        END AS crn_valid,
        cb.TotalAmountIncludingGST AS sales_amount
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN brands b ON cb.BrandDescription = b.BrandDescription
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.BusinessDate BETWEEN lookback_13m_start AND @max_d
    ),
    first_date_brand AS (
      SELECT crn_valid, BrandDescription, MIN(d) AS first_date
      FROM tx_brand_window
      WHERE crn_valid IS NOT NULL
      GROUP BY 1,2
    ),
    first_txn_brand AS (
      SELECT t.*
      FROM tx_brand_window t
      JOIN first_date_brand f
        ON f.crn_valid = t.crn_valid
       AND f.BrandDescription = t.BrandDescription
       AND f.first_date = t.d
    ),
    new_to_brand_weekly AS (
      SELECT
        DATE_TRUNC(d, WEEK(Wednesday)) AS week_start,
        store_id,
        product_number, article_id,     -- keep SKU so join keys match base grain
        BrandDescription, CategoryDescription, SubCategoryDescription,
        COUNT(DISTINCT crn_valid) AS new_to_brand_shoppers_13m,
        SUM(sales_amount)         AS new_to_brand_sales_13m
      FROM first_txn_brand
      JOIN stores st ON CAST(first_txn_brand.store_id AS STRING) = CAST(st.id AS STRING)
      WHERE d BETWEEN @min_d AND @max_d
      GROUP BY 1,2,3,4,5,6,7
    )

    /* 5) Final join */
    SELECT
      b.week_start, b.store_id, b.PriceFamilyCode,
      b.product_number, b.article_id,
      b.BrandDescription, b.CategoryDescription, b.SubCategoryDescription,
      b.discountpercent, b.multibuy, b.brochure,
      b.sales,

      COALESCE(sw.shoppers, 0)                  AS shoppers,

      COALESCE(ns.new_to_sku_shoppers_13m, 0)   AS new_to_sku_shoppers_13m,
      COALESCE(ns.new_to_sku_sales_13m,   0.0)  AS new_to_sku_sales_13m,

      COALESCE(nb.new_to_brand_shoppers_13m, 0) AS new_to_brand_shoppers_13m,
      COALESCE(nb.new_to_brand_sales_13m,   0.0)AS new_to_brand_sales_13m,

      b.n_competitors, b.n_any_cheaper, b.n_shelf_cheaper,
      b.n_promo_cheaper_no_hurdle, b.n_promo_cheaper_hurdle,
      b.avg_cheaper_gap, b.worst_gap, b.p90_gap,

      COALESCE(mi.internal_competitor_brochure, 0)               AS internal_competitor_brochure,
      COALESCE(mi.internal_competitor_multibuy, 0)               AS internal_competitor_multibuy,
      COALESCE(mi.max_internal_competitor_discount_percent, 0.0) AS max_internal_competitor_discount_percent

    FROM base b
    LEFT JOIN shoppers_weekly sw
      ON sw.week_start=b.week_start AND sw.store_id=CAST(b.store_id AS STRING)
     AND sw.product_number=b.product_number AND sw.article_id=b.article_id
     AND sw.BrandDescription=b.BrandDescription
     AND sw.CategoryDescription=b.CategoryDescription
     AND sw.SubCategoryDescription=b.SubCategoryDescription

    LEFT JOIN new_to_sku_weekly ns
      ON ns.week_start=b.week_start AND ns.store_id=CAST(b.store_id AS STRING)
     AND ns.product_number=b.product_number AND ns.article_id=b.article_id
     AND ns.BrandDescription=b.BrandDescription
     AND ns.CategoryDescription=b.CategoryDescription
     AND ns.SubCategoryDescription=b.SubCategoryDescription

    LEFT JOIN new_to_brand_weekly nb
      ON nb.week_start=b.week_start AND nb.store_id=CAST(b.store_id AS STRING)
     AND nb.product_number=b.product_number AND nb.article_id=b.article_id
     AND nb.BrandDescription=b.BrandDescription
     AND nb.CategoryDescription=b.CategoryDescription
     AND nb.SubCategoryDescription=b.SubCategoryDescription

    LEFT JOIN mcd_agg mi
      ON mi.week_start=b.week_start AND mi.store_id=b.store_id
     AND mi.BrandDescription=b.BrandDescription AND mi.CategoryDescription=b.CategoryDescription
    ;
    """

    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    df = job.result().to_dataframe(create_bqstorage_client=True)
    return df


def get_transaction_data_by_scope_fast2(
    client: Optional[bigquery.Client],
    base_table_fq: str,           # e.g. project.dataset.sku_store_week_sales_base
    sku_list: List[str],          # ARTICLE IDs (strings)
    store_list: List[str],        # store_ids (strings or ints)
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Weekly slice with window-function new_to_* logic.

    Semantics:
      • Weekly aggregation (sales/shoppers) is scoped to the campaign's stores if provided.
      • new_to_* lookbacks are GLOBAL across ALL stores (not limited by store_list).
      • new_to_sku_*     : first (crn, product_number) in last 13 months via LAG
      • new_to_brand_*   : first (crn, BrandDescription, SubCategoryDescription) in last 13 months via LAG
                           (i.e., "brand" means brand-subcategory throughout)
      • No category/subcategory totals are computed.
    """
    if client is None:
        client = bigquery.Client()

    from datetime import date

    def _clean(vals):
        out, seen = [], set()
        for v in (vals or []):
            s = str(v).strip() if v is not None else ""
            if s and s.lower() != "nan" and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    sku_ids_clean   = _clean(sku_list)
    store_ids_clean = _clean(store_list)
    all_stores = (len(store_ids_clean) == 0)

    if not sku_ids_clean:
        raise ValueError("Sanitized SKU list is empty. Check your media SKU inputs.")

    min_d = pd.to_datetime(start_date).date() if start_date else date(1900,1,1)
    max_d = pd.to_datetime(end_date).date()   if end_date   else date(2100,1,1)

    # choose native type for stores, but allow empty [] with either type
    try:
        store_ids_native = [int(x) for x in store_ids_clean]
        store_param_type = "INT64"
    except Exception:
        store_ids_native = [str(x) for x in store_ids_clean]
        store_param_type = "STRING"

    params = [
        bigquery.ArrayQueryParameter("sku_ids",   "STRING", [str(x) for x in sku_ids_clean]),
        bigquery.ArrayQueryParameter("store_ids", store_param_type, store_ids_native),
        bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
        bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
        bigquery.ScalarQueryParameter("all_stores", "BOOL", all_stores),
    ]

    sql = f"""
    DECLARE lookback_13m_start DATE DEFAULT DATE_SUB(@min_d, INTERVAL 400 DAY);
    DECLARE all_stores BOOL DEFAULT @all_stores;

    /* Parameter tables */
    WITH
    skus AS (
      SELECT DISTINCT id
      FROM UNNEST(@sku_ids) AS id
      WHERE id IS NOT NULL AND TRIM(id) <> ''
    ),
    stores AS (
      SELECT DISTINCT CAST(id AS STRING) AS id
      FROM UNNEST(@store_ids) AS id
      WHERE id IS NOT NULL AND TRIM(CAST(id AS STRING)) <> ''
    ),

    /* A) Weekly base (scoped to campaign stores if provided) */
    base AS (
      SELECT
        t.week_start,
        t.store_id,
        t.PriceFamilyCode,
        t.product_number,
        t.article_id,
        t.BrandDescription, t.CategoryDescription, t.SubCategoryDescription,
        t.discountpercent, t.multibuy, t.brochure,
        t.sales,
        t.n_competitors, t.n_any_cheaper, t.n_shelf_cheaper,
        t.n_promo_cheaper_no_hurdle, t.n_promo_cheaper_hurdle,
        t.avg_cheaper_gap, t.worst_gap, t.p90_gap
      FROM `{base_table_fq}` t
      JOIN skus s ON CAST(t.article_id AS STRING) = s.id
      WHERE t.week_start BETWEEN @min_d AND @max_d
        AND ( all_stores OR CAST(t.store_id AS STRING) IN (SELECT id FROM stores) )
    ),

    /* Internal-competitor features (unchanged) */
    pf_by_w_s_bcat AS (
      SELECT DISTINCT week_start, store_id, BrandDescription, CategoryDescription, PriceFamilyCode
      FROM base
    ),
    promo_pf AS (
      SELECT
        pwstartdate    AS week_start,
        petpricefamily AS PriceFamilyCode,
        discountpercent,
        brochure,
        MBTrigger      AS multibuy
      FROM `gcp-wow-ent-im-tbl-prod.adp_dm_quantium_inbound_view_smkt.qtm_smkt_pet_incrementality_v3_v`
      WHERE SalesDistrict='National'
        AND FutureFlag='PAST'
        AND pwstartdate BETWEEN @min_d AND @max_d
    ),
    mcd_agg AS (
      SELECT
        b.week_start, b.store_id, b.BrandDescription, b.CategoryDescription,
        MAX(CASE WHEN LOWER(p.brochure) LIKE '%not%' OR p.brochure IS NULL THEN 0 ELSE 1 END)
          AS internal_competitor_brochure,
        MAX(CASE WHEN LOWER(p.multibuy)  LIKE '%not%' OR p.multibuy  IS NULL THEN 0 ELSE 1 END)
          AS internal_competitor_multibuy,
        MAX(IFNULL(p.discountpercent,0.0)) AS max_internal_competitor_discount_percent
      FROM pf_by_w_s_bcat b
      LEFT JOIN promo_pf p
        ON p.week_start = b.week_start AND p.PriceFamilyCode = b.PriceFamilyCode
      GROUP BY 1,2,3,4
    ),

    /* B) In-window, in-scope transactions (scoped to campaign stores) for weekly shoppers */
    txn_scope AS (
      SELECT
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday))  AS week_start,
        CAST(cb.SiteNumber AS STRING)                 AS store_id,
        cb.ProductNumber                              AS product_number,
        CAST(cb.Article AS STRING)                    AS article_id,
        cb.BrandDescription, cb.CategoryDescription, cb.SubCategoryDescription,
        CASE
          WHEN cb.CRN IS NOT NULL
           AND REGEXP_CONTAINS(TRIM(CAST(cb.CRN AS STRING)), r'^[0-9]{{3,}}$')
          THEN TRIM(CAST(cb.CRN AS STRING))
        END AS crn_valid,
        cb.TotalAmountIncludingGST AS sales_amount,
        DATE(cb.BusinessDate)      AS d
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN skus s ON CAST(cb.Article AS STRING) = s.id
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.BusinessDate BETWEEN @min_d AND @max_d
        AND ( all_stores OR CAST(cb.SiteNumber AS STRING) IN (SELECT id FROM stores) )
    ),

    shoppers_weekly AS (
      SELECT
        week_start, store_id, product_number, article_id,
        BrandDescription, CategoryDescription, SubCategoryDescription,
        COUNT(DISTINCT crn_valid) AS shoppers
      FROM txn_scope
      WHERE crn_valid IS NOT NULL
      GROUP BY 1,2,3,4,5,6,7
    ),

    /* C) GLOBAL lookback planes (across ALL stores), using window LAG */
    -- SKU lookback: (crn, product_number)
    tx_sku_all AS (
      SELECT
        DATE(cb.BusinessDate)                        AS d,
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday)) AS week_start,
        CAST(cb.SiteNumber AS STRING)                AS store_id,
        cb.ProductNumber                             AS product_number,
        CAST(cb.Article AS STRING)                   AS article_id,
        cb.BrandDescription, cb.CategoryDescription, cb.SubCategoryDescription,
        CASE
          WHEN cb.CRN IS NOT NULL
           AND REGEXP_CONTAINS(TRIM(CAST(cb.CRN AS STRING)), r'^[0-9]{{3,}}$')
          THEN TRIM(CAST(cb.CRN AS STRING))
        END AS crn_valid,
        cb.TotalAmountIncludingGST AS sales_amount
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN skus s ON CAST(cb.Article AS STRING) = s.id
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.BusinessDate BETWEEN lookback_13m_start AND @max_d
    ),
    prev_sku AS (
      SELECT
        *,
        LAG(d) OVER (PARTITION BY crn_valid, product_number ORDER BY d) AS prev_dt
      FROM tx_sku_all
      WHERE crn_valid IS NOT NULL
    ),
    flags_sku AS (
      SELECT
        week_start, store_id, product_number, article_id,
        BrandDescription, CategoryDescription, SubCategoryDescription,
        crn_valid, sales_amount, d,
        -- "new to sku 13m" if no prior purchase or last prior >= 13 months ago
        IF(prev_dt IS NULL OR DATE_DIFF(d, prev_dt, MONTH) >= 13, TRUE, FALSE) AS is_new_to_sku_13m
      FROM prev_sku
      WHERE d BETWEEN @min_d AND @max_d
    ),
    new_to_sku_weekly AS (
      SELECT
        DATE_TRUNC(d, WEEK(Wednesday)) AS week_start,
        store_id, product_number, article_id,
        BrandDescription, CategoryDescription, SubCategoryDescription,
        COUNT(DISTINCT IF(is_new_to_sku_13m, crn_valid, NULL)) AS new_to_sku_shoppers_13m,
        SUM(           IF(is_new_to_sku_13m, sales_amount, 0.0)) AS new_to_sku_sales_13m
      FROM flags_sku
      WHERE ( all_stores OR store_id IN (SELECT id FROM stores) )
      GROUP BY 1,2,3,4,5,6,7
    ),

    -- BRAND-SUBCATEGORY lookback: (crn, BrandDescription, SubCategoryDescription)
    brand_sub_pairs AS (
      -- only the brand-subcategory pairs that exist in the analysis slice
      SELECT DISTINCT BrandDescription, SubCategoryDescription
      FROM base
      WHERE BrandDescription IS NOT NULL AND TRIM(BrandDescription) <> ''
        AND SubCategoryDescription IS NOT NULL AND TRIM(SubCategoryDescription) <> ''
    ),
    tx_bs_all AS (
      SELECT
        DATE(cb.BusinessDate)                        AS d,
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday)) AS week_start,
        CAST(cb.SiteNumber AS STRING)                AS store_id,
        cb.ProductNumber                             AS product_number,
        CAST(cb.Article AS STRING)                   AS article_id,
        cb.BrandDescription, cb.CategoryDescription, cb.SubCategoryDescription,
        CASE
          WHEN cb.CRN IS NOT NULL
           AND REGEXP_CONTAINS(TRIM(CAST(cb.CRN AS STRING)), r'^[0-9]{{3,}}$')
          THEN TRIM(CAST(cb.CRN AS STRING))
        END AS crn_valid,
        cb.TotalAmountIncludingGST AS sales_amount
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN brand_sub_pairs p
        ON cb.BrandDescription       = p.BrandDescription
       AND cb.SubCategoryDescription = p.SubCategoryDescription
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.BusinessDate BETWEEN lookback_13m_start AND @max_d
    ),
    prev_bs AS (
      SELECT
        *,
        LAG(d) OVER (PARTITION BY crn_valid, BrandDescription, SubCategoryDescription ORDER BY d) AS prev_dt
      FROM tx_bs_all
      WHERE crn_valid IS NOT NULL
    ),
    flags_bs AS (
      SELECT
        week_start, store_id,
        BrandDescription, CategoryDescription, SubCategoryDescription,
        product_number, article_id,
        crn_valid, sales_amount, d,
        IF(prev_dt IS NULL OR DATE_DIFF(d, prev_dt, MONTH) >= 13, TRUE, FALSE) AS is_new_to_brand_sub_13m
      FROM prev_bs
      WHERE d BETWEEN @min_d AND @max_d
    ),
    new_to_brand_sub_weekly AS (
      SELECT
        DATE_TRUNC(d, WEEK(Wednesday)) AS week_start,
        store_id,
        BrandDescription, SubCategoryDescription,
        COUNT(DISTINCT IF(is_new_to_brand_sub_13m, crn_valid, NULL)) AS new_to_brand_shoppers_13m,
        SUM(           IF(is_new_to_brand_sub_13m, sales_amount, 0.0)) AS new_to_brand_sales_13m
      FROM flags_bs
      WHERE ( all_stores OR store_id IN (SELECT id FROM stores) )
      GROUP BY 1,2,3,4
    ),

    /* Optional: brand-subcategory shoppers (unique CRNs per week/store at that grain) */
    brand_subcategory_shoppers_weekly AS (
      SELECT
        week_start, store_id, BrandDescription, SubCategoryDescription,
        COUNT(DISTINCT crn_valid) AS brand_subcategory_shoppers
      FROM txn_scope
      WHERE crn_valid IS NOT NULL
      GROUP BY 1,2,3,4
    )

    /* Final join */
    SELECT
      b.week_start, b.store_id, b.PriceFamilyCode,
      b.product_number, b.article_id,
      b.BrandDescription, b.CategoryDescription, b.SubCategoryDescription,
      b.discountpercent, b.multibuy, b.brochure,
      b.sales,

      COALESCE(sw.shoppers, 0)                  AS shoppers,

      COALESCE(ns.new_to_sku_shoppers_13m, 0)   AS new_to_sku_shoppers_13m,
      COALESCE(ns.new_to_sku_sales_13m,   0.0)  AS new_to_sku_sales_13m,

      COALESCE(nbs.new_to_brand_shoppers_13m, 0)   AS new_to_brand_shoppers_13m,
      COALESCE(nbs.new_to_brand_sales_13m,   0.0)  AS new_to_brand_sales_13m,

      b.n_competitors, b.n_any_cheaper, b.n_shelf_cheaper,
      b.n_promo_cheaper_no_hurdle, b.n_promo_cheaper_hurdle,
      b.avg_cheaper_gap, b.worst_gap, b.p90_gap,

      COALESCE(mi.internal_competitor_brochure, 0)               AS internal_competitor_brochure,
      COALESCE(mi.internal_competitor_multibuy, 0)               AS internal_competitor_multibuy,
      COALESCE(mi.max_internal_competitor_discount_percent, 0.0) AS max_internal_competitor_discount_percent,

      COALESCE(bssw.brand_subcategory_shoppers, 0) AS brand_subcategory_shoppers

    FROM base b
    LEFT JOIN shoppers_weekly sw
      ON sw.week_start=b.week_start AND sw.store_id=CAST(b.store_id AS STRING)
     AND sw.product_number=b.product_number AND sw.article_id=b.article_id
     AND sw.BrandDescription=b.BrandDescription
     AND sw.CategoryDescription=b.CategoryDescription
     AND sw.SubCategoryDescription=b.SubCategoryDescription

    LEFT JOIN new_to_sku_weekly ns
      ON ns.week_start=b.week_start AND ns.store_id=CAST(b.store_id AS STRING)
     AND ns.product_number=b.product_number AND ns.article_id=b.article_id
     AND ns.BrandDescription=b.BrandDescription
     AND ns.CategoryDescription=b.CategoryDescription
     AND ns.SubCategoryDescription=b.SubCategoryDescription

    LEFT JOIN new_to_brand_sub_weekly nbs
      ON nbs.week_start=b.week_start AND nbs.store_id=CAST(b.store_id AS STRING)
     AND nbs.BrandDescription=b.BrandDescription
     AND nbs.SubCategoryDescription=b.SubCategoryDescription

    LEFT JOIN mcd_agg mi
      ON mi.week_start=b.week_start AND mi.store_id=b.store_id
     AND mi.BrandDescription=b.BrandDescription AND mi.CategoryDescription=b.CategoryDescription

    LEFT JOIN brand_subcategory_shoppers_weekly bssw
      ON bssw.week_start=b.week_start AND bssw.store_id=CAST(b.store_id AS STRING)
     AND bssw.BrandDescription=b.BrandDescription AND bssw.SubCategoryDescription=b.SubCategoryDescription
    ;
    """

    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    df = job.result().to_dataframe(create_bqstorage_client=True)
    return df

def get_transaction_data_by_scope_fast2_cohort_total(
    client: Optional[bigquery.Client],
    base_table_fq: str,           # e.g. project.dataset.sku_store_week_sales_base
    sku_list: List[str],          # ARTICLE IDs (strings)
    store_list: List[str],        # store_ids (strings or ints) — defines the cohort; empty => all stores
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Cohort × week totals with CRN×WEEK-level new_to_* logic and global lookbacks.

    Targets produced (one row per week_start):
      - sales
      - shoppers
      - new_to_sku_sales_13m, new_to_sku_shoppers_13m
      - new_to_brand_sales_13m, new_to_brand_shoppers_13m   (brand = brand-subcategory)
    EXOG aggregated to cohort-week (sales-weighted where sensible):
      - discountpercent
      - max_internal_competitor_discount_percent
      - n_competitors, n_any_cheaper, n_shelf_cheaper,
        n_promo_cheaper_no_hurdle, n_promo_cheaper_hurdle,
        avg_cheaper_gap, worst_gap, p90_gap
      - brochure_Not on brochure, multibuy_Not on Multibuy (sales-weighted shares)
    """
    if client is None:
        client = bigquery.Client()

    from datetime import date

    def _clean(vals):
        out, seen = [], set()
        for v in (vals or []):
            s = str(v).strip() if v is not None else ""
            if s and s.lower() != "nan" and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    sku_ids_clean   = _clean(sku_list)
    store_ids_clean = _clean(store_list)
    all_stores = (len(store_ids_clean) == 0)

    if not sku_ids_clean:
        raise ValueError("Sanitized SKU list is empty.")

    min_d = pd.to_datetime(start_date).date() if start_date else date(1900,1,1)
    max_d = pd.to_datetime(end_date).date()   if end_date   else date(2100,1,1)

    try:
        store_ids_native = [int(x) for x in store_ids_clean]
        store_param_type = "INT64"
    except Exception:
        store_ids_native = [str(x) for x in store_ids_clean]
        store_param_type = "STRING"

    params = [
        bigquery.ArrayQueryParameter("sku_ids",   "STRING", [str(x) for x in sku_ids_clean]),
        bigquery.ArrayQueryParameter("store_ids", store_param_type, store_ids_native),
        bigquery.ScalarQueryParameter("min_d", "DATE", min_d),
        bigquery.ScalarQueryParameter("max_d", "DATE", max_d),
        bigquery.ScalarQueryParameter("all_stores", "BOOL", all_stores),
    ]

    sql = f"""
    DECLARE lookback_13m_start DATE DEFAULT DATE_SUB(@min_d, INTERVAL 400 DAY);
    DECLARE all_stores BOOL DEFAULT @all_stores;

    /* Params */
    WITH
    skus AS (
      SELECT DISTINCT id
      FROM UNNEST(@sku_ids) AS id
      WHERE id IS NOT NULL AND TRIM(id) <> ''
    ),
    stores AS (
      SELECT DISTINCT CAST(id AS STRING) AS id
      FROM UNNEST(@store_ids) AS id
      WHERE id IS NOT NULL AND TRIM(CAST(id AS STRING)) <> ''
    ),

    /* A) Cohort-week product base (sum sales over cohort stores) */
    base_cohort AS (
      SELECT
        t.week_start,
        t.product_number,
        t.article_id,
        ANY_VALUE(t.BrandDescription)  AS BrandDescription,
        ANY_VALUE(t.CategoryDescription) AS CategoryDescription,
        ANY_VALUE(t.SubCategoryDescription) AS SubCategoryDescription,

        -- exogs per product-week; keep as-is here
        ANY_VALUE(t.discountpercent)                       AS discountpercent,
        ANY_VALUE(t.multibuy)                              AS multibuy,
        ANY_VALUE(t.brochure)                              AS brochure,
        ANY_VALUE(t.n_competitors)                         AS n_competitors,
        ANY_VALUE(t.n_any_cheaper)                         AS n_any_cheaper,
        ANY_VALUE(t.n_shelf_cheaper)                       AS n_shelf_cheaper,
        ANY_VALUE(t.n_promo_cheaper_no_hurdle)             AS n_promo_cheaper_no_hurdle,
        ANY_VALUE(t.n_promo_cheaper_hurdle)                AS n_promo_cheaper_hurdle,
        ANY_VALUE(t.avg_cheaper_gap)                       AS avg_cheaper_gap,
        ANY_VALUE(t.worst_gap)                             AS worst_gap,
        ANY_VALUE(t.p90_gap)                               AS p90_gap,

        SUM(t.sales) AS sales
      FROM `{base_table_fq}` t
      JOIN skus s ON CAST(t.article_id AS STRING) = s.id
      WHERE t.week_start BETWEEN @min_d AND @max_d
        AND ( all_stores OR CAST(t.store_id AS STRING) IN (SELECT id FROM stores) )
      GROUP BY 1,2,3
    ),

    /* Internal competitor discount (from promo table) aggregated to week */
    pf_keys AS (
      SELECT DISTINCT week_start, BrandDescription, CategoryDescription, PriceFamilyCode
      FROM (
        SELECT
          t.week_start,
          ANY_VALUE(t.BrandDescription) AS BrandDescription,
          ANY_VALUE(t.CategoryDescription) AS CategoryDescription,
          ANY_VALUE(t.PriceFamilyCode) AS PriceFamilyCode
        FROM `{base_table_fq}` t
        JOIN skus s ON CAST(t.article_id AS STRING)=s.id
        WHERE t.week_start BETWEEN @min_d AND @max_d
          AND ( all_stores OR CAST(t.store_id AS STRING) IN (SELECT id FROM stores) )
        GROUP BY t.week_start
      )
    ),
    promo_pf AS (
      SELECT
        pwstartdate    AS week_start,
        petpricefamily AS PriceFamilyCode,
        discountpercent AS pf_discount,
        brochure,
        MBTrigger      AS multibuy
      FROM `gcp-wow-ent-im-tbl-prod.adp_dm_quantium_inbound_view_smkt.qtm_smkt_pet_incrementality_v3_v`
      WHERE SalesDistrict='National'
        AND FutureFlag='PAST'
    ),
    mcd_store AS (
      SELECT
        k.week_start, k.PriceFamilyCode,
        MAX(COALESCE(pf_discount,0.0)) AS max_internal_competitor_discount_percent
      FROM pf_keys k
      LEFT JOIN promo_pf p
        ON p.week_start = k.week_start
       AND p.PriceFamilyCode = k.PriceFamilyCode
      GROUP BY 1,2
    ),
    mcd_week AS (
      SELECT week_start,
             MAX(max_internal_competitor_discount_percent) AS max_internal_competitor_discount_percent
      FROM mcd_store
      GROUP BY 1
    ),

    /* B) Cohort CRN×WEEK streams (store-scoped) */
    crn_sku_cohort_wk AS (
      SELECT
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday))  AS week_start,
        cb.ProductNumber                              AS product_number,
        TRIM(CAST(cb.CRN AS STRING))                  AS crn_valid,
        SUM(cb.TotalAmountIncludingGST)               AS sales_amount_wk
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN skus s ON CAST(cb.Article AS STRING) = s.id
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.CRN IS NOT NULL
        AND SAFE_CAST(cb.CRN AS INT64) IS NOT NULL
        AND LENGTH(CAST(cb.CRN AS STRING)) >= 3
        AND cb.BusinessDate BETWEEN @min_d AND @max_d
        AND ( all_stores OR CAST(cb.SiteNumber AS STRING) IN (SELECT id FROM stores) )
      GROUP BY 1,2,3
    ),
    cohort_shoppers AS (
      SELECT week_start, COUNT(DISTINCT crn_valid) AS shoppers
      FROM crn_sku_cohort_wk
      GROUP BY 1
    ),

    /* C) GLOBAL lookbacks (all stores) at weekly grain via LAG */
    tx_all_sku_wk AS (
      SELECT
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday)) AS week_start,
        cb.ProductNumber                             AS product_number,
        TRIM(CAST(cb.CRN AS STRING))                 AS crn_valid
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN skus s ON CAST(cb.Article AS STRING) = s.id
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.CRN IS NOT NULL
        AND SAFE_CAST(cb.CRN AS INT64) IS NOT NULL
        AND LENGTH(CAST(cb.CRN AS STRING)) >= 3
        AND cb.BusinessDate BETWEEN lookback_13m_start AND @max_d
      GROUP BY 1,2,3
    ),
    prev_sku_wk AS (
      SELECT *,
             LAG(week_start) OVER (PARTITION BY crn_valid, product_number ORDER BY week_start) AS prev_wk
      FROM tx_all_sku_wk
    ),
    flags_sku_wk AS (
      SELECT week_start, product_number, crn_valid,
             IF(prev_wk IS NULL OR DATE_DIFF(week_start, prev_wk, MONTH) >= 13, TRUE, FALSE) AS is_new_to_sku_13m
      FROM prev_sku_wk
      WHERE week_start BETWEEN @min_d AND @max_d
    ),

    bs_pairs AS (
      SELECT DISTINCT BrandDescription, SubCategoryDescription
      FROM base_cohort
      WHERE BrandDescription IS NOT NULL AND TRIM(BrandDescription) <> ''
        AND SubCategoryDescription IS NOT NULL AND TRIM(SubCategoryDescription) <> ''
    ),
    tx_all_bs_wk AS (
      SELECT
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday)) AS week_start,
        cb.BrandDescription, cb.SubCategoryDescription,
        TRIM(CAST(cb.CRN AS STRING))                 AS crn_valid
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN bs_pairs p
        ON cb.BrandDescription       = p.BrandDescription
       AND cb.SubCategoryDescription = p.SubCategoryDescription
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.CRN IS NOT NULL
        AND SAFE_CAST(cb.CRN AS INT64) IS NOT NULL
        AND LENGTH(CAST(cb.CRN AS STRING)) >= 3
        AND cb.BusinessDate BETWEEN lookback_13m_start AND @max_d
      GROUP BY 1,2,3,4
    ),
    prev_bs_wk AS (
      SELECT *,
             LAG(week_start) OVER (PARTITION BY crn_valid, BrandDescription, SubCategoryDescription ORDER BY week_start) AS prev_wk
      FROM tx_all_bs_wk
    ),
    flags_bs_wk AS (
      SELECT week_start, BrandDescription, SubCategoryDescription, crn_valid,
             IF(prev_wk IS NULL OR DATE_DIFF(week_start, prev_wk, MONTH) >= 13, TRUE, FALSE) AS is_new_to_brand_sub_13m
      FROM prev_bs_wk
      WHERE week_start BETWEEN @min_d AND @max_d
    ),

    /* D) Join flags to cohort CRN×WEEK planes and aggregate to cohort-week totals */
    sku_flagged AS (
      SELECT
        c.week_start, c.crn_valid, c.sales_amount_wk,
        f.is_new_to_sku_13m
      FROM crn_sku_cohort_wk c
      LEFT JOIN flags_sku_wk f
        ON f.week_start = c.week_start
       AND f.product_number = c.product_number
       AND f.crn_valid = c.crn_valid
    ),
    new_to_sku_cohort AS (
      SELECT
        week_start,
        COUNT(DISTINCT IF(is_new_to_sku_13m, crn_valid, NULL)) AS new_to_sku_shoppers_13m,
        SUM(           IF(is_new_to_sku_13m, sales_amount_wk, 0.0)) AS new_to_sku_sales_13m
      FROM sku_flagged
      GROUP BY 1
    ),

    bs_cohort_wk AS (
      SELECT
        DATE_TRUNC(cb.BusinessDate, WEEK(Wednesday)) AS week_start,
        TRIM(CAST(cb.CRN AS STRING))                 AS crn_valid,
        SUM(cb.TotalAmountIncludingGST)              AS sales_amount_wk_bs
      FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` cb
      JOIN bs_pairs p
        ON cb.BrandDescription       = p.BrandDescription
       AND cb.SubCategoryDescription = p.SubCategoryDescription
      WHERE cb.SalesOrganisation='1005'
        AND LOWER(cb.Channel)='in store'
        AND cb.TotalAmountIncludingGST > 0
        AND cb.CRN IS NOT NULL
        AND SAFE_CAST(cb.CRN AS INT64) IS NOT NULL
        AND LENGTH(CAST(cb.CRN AS STRING)) >= 3
        AND cb.BusinessDate BETWEEN @min_d AND @max_d
        AND ( all_stores OR CAST(cb.SiteNumber AS STRING) IN (SELECT id FROM stores) )
      GROUP BY 1,2
    ),
    bs_flagged AS (
      SELECT
        b.week_start, b.crn_valid, b.sales_amount_wk_bs,
        f.is_new_to_brand_sub_13m
      FROM bs_cohort_wk b
      LEFT JOIN flags_bs_wk f
        ON f.week_start = b.week_start
       AND f.crn_valid  = b.crn_valid
    ),
    new_to_brand_sub_cohort AS (
      SELECT
        week_start,
        COUNT(DISTINCT IF(is_new_to_brand_sub_13m, crn_valid, NULL)) AS new_to_brand_shoppers_13m,
        SUM(           IF(is_new_to_brand_sub_13m, sales_amount_wk_bs, 0.0)) AS new_to_brand_sales_13m
      FROM bs_flagged
      GROUP BY 1
    ),

    /* E) Aggregate EXOG to cohort-week (sales-weighted) + total sales */
    exog_agg AS (
      SELECT
        week_start,
        SUM(sales) AS sales,

        SAFE_DIVIDE(SUM(COALESCE(discountpercent,0) * sales), NULLIF(SUM(sales),0))       AS discountpercent,

        SAFE_DIVIDE(SUM(COALESCE(n_competitors,0) * sales), NULLIF(SUM(sales),0))         AS n_competitors,
        SAFE_DIVIDE(SUM(COALESCE(n_any_cheaper,0) * sales), NULLIF(SUM(sales),0))         AS n_any_cheaper,
        SAFE_DIVIDE(SUM(COALESCE(n_shelf_cheaper,0) * sales), NULLIF(SUM(sales),0))       AS n_shelf_cheaper,
        SAFE_DIVIDE(SUM(COALESCE(n_promo_cheaper_no_hurdle,0) * sales), NULLIF(SUM(sales),0)) AS n_promo_cheaper_no_hurdle,
        SAFE_DIVIDE(SUM(COALESCE(n_promo_cheaper_hurdle,0) * sales), NULLIF(SUM(sales),0))    AS n_promo_cheaper_hurdle,

        SAFE_DIVIDE(SUM(COALESCE(avg_cheaper_gap,0) * sales), NULLIF(SUM(sales),0))       AS avg_cheaper_gap,
        SAFE_DIVIDE(SUM(COALESCE(worst_gap,0) * sales), NULLIF(SUM(sales),0))             AS worst_gap,
        SAFE_DIVIDE(SUM(COALESCE(p90_gap,0) * sales), NULLIF(SUM(sales),0))               AS p90_gap,

        SAFE_DIVIDE(SUM(IF(brochure = 'Not on brochure', sales, 0)), NULLIF(SUM(sales),0)) AS `brochure_Not on brochure`,
        SAFE_DIVIDE(SUM(IF(multibuy  = 'Not on Multibuy', sales, 0)), NULLIF(SUM(sales),0)) AS `multibuy_Not on Multibuy`
      FROM base_cohort
      GROUP BY 1
    )

    /* Final: cohort-week totals only */
    SELECT
      e.week_start,

      -- EXOG
      e.discountpercent,
      COALESCE(m.max_internal_competitor_discount_percent, 0.0) AS max_internal_competitor_discount_percent,
      e.n_competitors, e.n_any_cheaper, e.n_shelf_cheaper,
      e.n_promo_cheaper_no_hurdle, e.n_promo_cheaper_hurdle,
      e.avg_cheaper_gap, e.worst_gap, e.p90_gap,
      e.`brochure_Not on brochure`, e.`multibuy_Not on Multibuy`,

      -- Targets
      e.sales,
      COALESCE(cs.shoppers, 0)                                  AS shoppers,
      COALESCE(ns.new_to_sku_sales_13m,   0.0)                  AS new_to_sku_sales_13m,
      COALESCE(ns.new_to_sku_shoppers_13m,0)                    AS new_to_sku_shoppers_13m,
      COALESCE(nb.new_to_brand_sales_13m,   0.0)                AS new_to_brand_sales_13m,
      COALESCE(nb.new_to_brand_shoppers_13m,0)                  AS new_to_brand_shoppers_13m

    FROM exog_agg e
    LEFT JOIN mcd_week m ON m.week_start = e.week_start
    LEFT JOIN cohort_shoppers cs ON cs.week_start = e.week_start
    LEFT JOIN new_to_sku_cohort ns ON ns.week_start = e.week_start
    LEFT JOIN new_to_brand_sub_cohort nb ON nb.week_start = e.week_start
    ORDER BY e.week_start;
    """

    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    return job.result().to_dataframe(create_bqstorage_client=True)