-- x_precompute/sql/media_master.sql
-- Materializes a per-(booking, media_start/end) asset summary with:
--   - arrays of media/media_location/media_type
--   - comma-separated strings for sorted_store_list and sorted_sku_list
--   - campaign_week / campaign_week_split / study_id
-- Notes:
--   - This table is a "once-and-refresh" precompute; the runtime pipeline reads it.
--   - Downstream code converts the comma strings to python lists via to_list_flexible().

DECLARE MIN_CAMPAIGN_START DATE DEFAULT DATE '2023-01-01';

CREATE OR REPLACE TABLE `{{PROJECT}}.{{DATASET}}.{{TABLE}}` AS
WITH campaigns AS (
  SELECT DISTINCT
    booking_number,
    quote_status,
    campaign_start_date,
    campaign_end_date,
    media_start_date,
    media_end_date,
    media,
    media_location,
    media_type,
    CASE
      WHEN LOWER(media_type) LIKE "%aisle fin%" THEN "aisle fin"
      WHEN LOWER(media_type) = "digital screen network" THEN "digital screens supers"
      WHEN LOWER(media_type) = "floor" THEN "floor media"
      ELSE LOWER(media_type)
    END AS cleaned_media_type,
    opportunity_name,
    -- normalize inputs used for splitting
    REGEXP_REPLACE(IFNULL(quoteline_sku, ""), r"\s+", "") AS quoteline_sku,
    REGEXP_REPLACE(IFNULL(store_list, ""), r"\s+", "")     AS store_list
  FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns`
  WHERE media_location NOT IN ("Online","None")
    AND campaign_start_date >= MIN_CAMPAIGN_START
    AND campaign_end_date   <  CURRENT_DATE("Australia/Sydney")
    AND quote_status NOT IN ('Not Approved', 'Draft', 'For Review','Sent')
    AND LOWER(media_type) <> "off network"
    AND LOWER(media_location) NOT LIKE "%big w%"
    AND LOWER(media_type)     NOT LIKE "%big w%"
    AND LOWER(booking_number) LIKE "%wow%"
    -- store_list is empty or numeric-only (no alpha chars)
    AND (NOT REGEXP_CONTAINS(IFNULL(store_list,""), r'[a-zA-Z]') OR store_list IS NULL OR store_list = "")
    AND LOWER(media_location) NOT LIKE "wooliesx"
    AND LOWER(media_location) NOT LIKE "emma reach"
    AND LOWER(media_type)     NOT LIKE "woolworths rewards"
    AND LOWER(media)          NOT LIKE "%digital audience%"
),
campaigns_with_assets_we_care_about AS (
  SELECT DISTINCT booking_number
  FROM campaigns
  WHERE LOWER(cleaned_media_type) IN (
    "bus stop", "aisle fin", "digital screens supers",
    "chilled bus stop & decal package", "floor media",
    "door decal", "hba screen", "chilled fin package",
    "freezer package", "pelmet"
  )
),
sorted_skus AS (
  SELECT * EXCEPT(quoteline_sku, store_list, sku),
         ARRAY_AGG(DISTINCT sku ORDER BY sku) AS sorted_sku_list
  FROM campaigns, UNNEST(SPLIT(quoteline_sku, ",")) AS sku
  GROUP BY booking_number, quote_status, campaign_start_date, campaign_end_date,
           media_start_date, media_end_date, media, media_location, media_type,
           cleaned_media_type, opportunity_name
),
sorted_stores AS (
  SELECT * EXCEPT(quoteline_sku, store_list, store),
         ARRAY_AGG(DISTINCT store ORDER BY store) AS sorted_store_list
  FROM campaigns, UNNEST(SPLIT(store_list, ",")) AS store
  GROUP BY booking_number, quote_status, campaign_start_date, campaign_end_date,
           media_start_date, media_end_date, media, media_location, media_type,
           cleaned_media_type, opportunity_name
),
sorted_skus_and_stores AS (
  SELECT
    c.booking_number,
    c.quote_status,
    c.campaign_start_date,
    c.campaign_end_date,
    c.media_start_date,
    c.media_end_date,
    c.media,
    c.media_location,
    c.media_type,
    c.cleaned_media_type,
    c.opportunity_name,
    -- Keep as comma strings; downstream parser handles both arrays and strings
    ARRAY_TO_STRING(ss.sorted_store_list, ",") AS sorted_store_list,
    ARRAY_TO_STRING(sk.sorted_sku_list,   ",") AS sorted_sku_list
  FROM campaigns c
  LEFT JOIN sorted_skus   sk
    ON  c.booking_number  = sk.booking_number
   AND c.media_start_date = sk.media_start_date
   AND c.media            = sk.media
   AND c.media_type       = sk.media_type
  LEFT JOIN sorted_stores ss
    ON  c.booking_number  = ss.booking_number
   AND c.media_start_date = ss.media_start_date
   AND c.media            = ss.media
   AND c.media_type       = ss.media_type
),
medias_aggregated AS (
  SELECT
    s.booking_number,
    s.campaign_start_date,
    s.campaign_end_date,
    s.media_start_date,
    s.media_end_date,
    s.opportunity_name,
    ARRAY_AGG(DISTINCT s.media          IGNORE NULLS ORDER BY s.media)          AS media_array,
    ARRAY_AGG(DISTINCT s.media_location IGNORE NULLS ORDER BY s.media_location) AS media_location_array,
    ARRAY_AGG(DISTINCT s.cleaned_media_type IGNORE NULLS ORDER BY s.cleaned_media_type) AS media_type_array,
    s.sorted_store_list,
    s.sorted_sku_list
  FROM sorted_skus_and_stores s
  INNER JOIN campaigns_with_assets_we_care_about w
    ON w.booking_number = s.booking_number
  GROUP BY s.booking_number, s.campaign_start_date, s.campaign_end_date,
           s.media_start_date, s.media_end_date, s.opportunity_name,
           s.sorted_store_list, s.sorted_sku_list
  ORDER BY booking_number, media_start_date, media_end_date
),
week_splits AS (
  SELECT
    booking_number,
    opportunity_name,
    campaign_start_date,
    campaign_end_date,
    media_start_date,
    media_end_date,
    media_array,
    media_location_array,
    media_type_array,
    sorted_store_list,
    sorted_sku_list,
    DENSE_RANK() OVER (PARTITION BY booking_number ORDER BY media_start_date) AS campaign_week,
    ROW_NUMBER() OVER (PARTITION BY booking_number, media_start_date ORDER BY media_start_date, media_end_date) AS campaign_week_split
  FROM medias_aggregated
)
SELECT
  booking_number,
  CONCAT(booking_number, "_Week_", campaign_week, "_Split_", campaign_week_split) AS study_id,
  DATE_DIFF(media_end_date, media_start_date, WEEK) AS study_duration_weeks,
  opportunity_name,
  campaign_start_date,
  campaign_end_date,
  media_start_date,
  media_end_date,
  media_array,
  media_location_array,
  media_type_array,
  sorted_store_list,     -- comma-separated string (e.g. "1001,1002,...")
  sorted_sku_list,       -- comma-separated string
  campaign_week,
  campaign_week_split
FROM week_splits
ORDER BY booking_number, media_start_date, media_end_date, campaign_week, campaign_week_split;
