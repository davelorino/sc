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
    ORDER BY b.week_start, b.store_id, b.product_number
    """

    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    tx_df = job.result().to_dataframe()
    print("Done!")
    return tx_df