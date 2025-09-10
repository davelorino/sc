from __future__ import annotations
import pandas as pd
from typing import Dict, List, Tuple
from src.ri.insights.exposure_features import exposure_by_grain
from src.ri.insights.discount_depth import compute_discount_depth
from src.ri.model.structures import parse_group_key
from src.ri.insights.modeling.weekly_within_campaign import fit_weekly_within_campaign_model
from src.ri.insights.modeling.campaign_level import fit_campaign_level_model

def _parse_keys_from_group(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["group"].apply(parse_group_key).apply(pd.Series)
    return pd.concat([df, parsed], axis=1) 

def build_weekly_table_for_metric_grain(
    *,
    cmp_by_campaign_metric_band: Dict[str, pd.DataFrame],  # band -> compare df with ds,y,yhat,group,level,uplift
    media_master_df: pd.DataFrame,
    tx_master_df: pd.DataFrame,
    product_dim: pd.DataFrame,
    promo_skus_by_campaign: Dict[str, List[str]],
    grain: str,
    media_type_weights: Dict[str,float] | None = None
) -> pd.DataFrame:
    """
    Returns a stacked weekly table across campaigns at the requested grain with columns:
      campaign_id, week_start, keys..., uplift, n_assets_cov, n_types, dose_type::*, sov, esov,
      is_leader, coverage_frac, disc_depth_*, running_carto_media
    """
    rows = []
    fleet_size = tx_master_df["store_id"].nunique()

    for cid, bands in cmp_by_campaign_metric_band.items():
        # choose band → which compare df to use for this grain
        band = {
            "sku": "promoted_skus",
            "brand": "brand_skus",
            "brand_category":"brand_category",
            "brand_subcategory":"brand_subcat",
            "category":"category",
            "subcategory":"subcategory",
        }[grain]
        cmp_df = bands.get(band)
        if cmp_df is None or cmp_df.empty:
            continue

        # parse keys
        cmp_df = cmp_df.rename(columns={"ds":"week_start"})
        cmp_df = _parse_keys_from_group(cmp_df)

        # build exposure features for this grain
        media_one = media_master_df[media_master_df["booking_number"]==cid].copy()
        exp = exposure_by_grain(
            media_one_campaign=media_one,
            tx_master_df=tx_master_df,
            product_dim=product_dim,
            fleet_size=fleet_size,
            grain=grain,
            promo_skus=promo_skus_by_campaign[cid],
            train_end_date=cmp_df["week_start"].min() - pd.Timedelta(days=1),
            media_type_weights=media_type_weights
        )

        # discount depth per grain
        if grain in ("brand_category","brand_subcategory"):
            by = "brand_category" if grain=="brand_category" else "brand_subcategory"
        else:
            by = grain
        disc = compute_discount_depth(tx=tx_master_df, stores=tx_master_df["store_id"].astype(str).unique().tolist(), by=by)

        # join keys per grain
        # choose join keys
        if grain == "sku":
            k = ["week_start","product_id"]; cmp_df["product_id"] = cmp_df["product_id"].fillna("ALL")  # group contains sku when level==sku
            # lift exp features from brand-category: need product_dim
            pdim = product_dim.rename(columns={"product_id":"product_id"})
            cmp_df = cmp_df.merge(pdim[["product_id","brand","category"]], on="product_id", how="left")
            exp_join = exp.rename(columns={"sku":"product_id"})
            join_keys = ["week_start","product_id"]
            x = cmp_df.merge(exp_join, on=["week_start","product_id"], how="left")
        elif grain == "brand":
            x = cmp_df.merge(exp, on=["week_start","brand"], how="left")
        elif grain == "brand_category":
            x = cmp_df.merge(exp, on=["week_start","brand","category"], how="left")
        elif grain == "brand_subcategory":
            x = cmp_df.merge(exp.rename(columns={"subcat":"subcategory"}), on=["week_start","brand","subcategory"], how="left")
        elif grain == "category":
            x = cmp_df.merge(exp, on=["week_start","category"], how="left")
        elif grain == "subcategory":
            x = cmp_df.merge(exp, on=["week_start","subcategory"], how="left")
        else:
            continue

        # merge discount depth (aligned to the same keys)
        if by == "brand_category":
            x = x.merge(disc, on=["week_start","brand","category"], how="left")
            x = x.rename(columns={"disc_depth_bc":"disc_depth"})
        elif by == "brand_subcategory":
            x = x.merge(disc, on=["week_start","brand","subcategory"], how="left")
            x = x.rename(columns={"disc_depth_bsc":"disc_depth"})
        elif by == "brand":
            x = x.merge(disc, on=["week_start","brand"], how="left")
            x = x.rename(columns={"disc_depth_brand":"disc_depth"})
        elif by == "category":
            x = x.merge(disc, on=["week_start","category"], how="left")
            x = x.rename(columns={"disc_depth_category":"disc_depth"})
        elif by == "subcategory":
            x = x.merge(disc, on=["week_start","subcategory"], how="left")
            x = x.rename(columns={"disc_depth_subcategory":"disc_depth"})
        elif by == "sku":
            x = x.merge(disc, on=["week_start","product_id"], how="left")
            x = x.rename(columns={"disc_depth_sku":"disc_depth"})

        x["disc_depth"] = x["disc_depth"].fillna(0.0)
        x["campaign_id"] = cid
        rows.append(x)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def fit_models_for_metric_grain(
    weekly_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, object, object, pd.DataFrame]:
    """
    Fits weekly within-campaign (Ridge) and campaign-level (OLS).
    Returns coef_weekly, coef_campaign, weekly_model, campaign_model, campaign_agg
    """
    if weekly_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame()

    # Harmonize feature columns
    dose_cols = [c for c in weekly_df.columns if c.startswith("dose_type::")]
    base_cols = ["n_assets_cov","n_types","disc_depth","sov","esov","is_leader",
                 "running_carto_media"]
    # seasonality placeholders (0); you can decorate with woy_sin/cos if you prefer
    weekly_df["woy_sin"] = 0.0; weekly_df["woy_cos"] = 0.0

    from src.ri.insights.modeling.weekly_within_campaign import fit_weekly_within_campaign_model
    from src.ri.insights.modeling.campaign_level import fit_campaign_level_model

    pipe, coef_weekly = fit_weekly_within_campaign_model(weekly_df, alpha=10.0)
    model, coef_campaign, camp_agg = fit_campaign_level_model(weekly_df, alpha=None)
    return coef_weekly, coef_campaign, pipe, model, camp_agg
