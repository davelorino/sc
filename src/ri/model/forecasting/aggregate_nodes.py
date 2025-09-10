# src/ri/model/forecasting/aggregate_nodes.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from src.ri.model.structures import NodeSpec, make_node_key

# ---------------------------------------------------------------------
# Exogenous regressors to carry into the forecasting series (weekly avg)
# Per your directive: discount/brochure/multibuy are EXOGENOUS and kept.
# running_carto_media is NOT included here (treatment flag).
# ---------------------------------------------------------------------
_NUMERIC_REG = [
    "discountpercent",
    "max_internal_competitor_discount_percent",
    "n_competitors", "n_any_cheaper", "n_shelf_cheaper",
    "n_promo_cheaper_no_hurdle", "n_promo_cheaper_hurdle",
    "avg_cheaper_gap", "worst_gap", "p90_gap",
]
_CATEGORICAL_REG = ["brochure", "multibuy"]

# ---------------------------------------------------------------------
# Target map: what column to sum for each metric key
# These names must match the master table columns you build in SQL.
# (We default to 13-month definitions for "new-to" metrics.)
# ---------------------------------------------------------------------
_TARGET_MAP = {
    # base
    "sales": "sales",
    "shoppers": "shoppers",

    # new-to SKU
    "new_to_sku_sales": "new_to_sku_sales_13m",
    "new_to_sku_shoppers": "new_to_sku_shoppers_13m",

    # new-to Brand
    "new_to_brand_sales": "new_to_brand_sales_13m",
    "new_to_brand_shoppers": "new_to_brand_shoppers_13m",

    # new-to Category
    "new_to_category_sales": "new_to_category_sales_13m",
    "new_to_category_shoppers": "new_to_category_shoppers_13m",

    # new-to Subcategory
    "new_to_subcategory_sales": "new_to_subcategory_sales_13m",
    "new_to_subcategory_shoppers": "new_to_subcategory_shoppers_13m",
}

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _mask_node(df: pd.DataFrame, node: NodeSpec) -> pd.Series:
    """Boolean mask for rows belonging to a node."""
    if node.level == "sku":
        return df["product_id"] == node.key[0]
    if node.level == "brand":
        return df["brand"] == node.key[0]
    if node.level == "category":
        return df["category"] == node.key[0]
    if node.level == "subcategory":
        return df["subcategory"] == node.key[0]
    if node.level == "brand_category":
        return (df["brand"] == node.key[0]) & (df["category"] == node.key[1])
    if node.level == "brand_subcategory":
        return (df["brand"] == node.key[0]) & (df["subcategory"] == node.key[1])
    raise ValueError(f"Unknown node.level: {node.level}")

def _resolve_target_column(df: pd.DataFrame, target_key: str) -> str:
    """
    Return the column name to sum for this metric key.
    Raise a clear error if the column doesn't exist.
    """
    col = _TARGET_MAP.get(target_key)
    if col is None:
        raise ValueError(f"Unknown target '{target_key}'. Valid keys: {sorted(_TARGET_MAP.keys())}")
    if col not in df.columns:
        raise KeyError(
            f"Target column '{col}' expected by metric '{target_key}' "
            f"is missing from the transactions dataframe."
        )
    return col

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def build_weekly_series_for_stores(
    tx: pd.DataFrame,
    store_list: List[str],
    nodes: List[NodeSpec],
    target: str,  # one of _TARGET_MAP keys
) -> pd.DataFrame:
    """
    Build a stacked weekly panel for a FIXED store cohort (e.g., a campaign-week subset),
    aggregating to the specified node list and target metric.

    Returns columns:
      - week_start (Timestamp, W-WED)
      - group        (string key like "brand=..|cat=.." or "sku=..")
      - product_id   ("ALL" placeholder; node identity is in 'group')
      - level        (node level)
      - y            (sum of the target for that node/week)
      - <exogenous weekly regressors> (averaged numerics, mode-dummied categoricals)

    Notes:
      • This function NEVER includes 'running_carto_media' (treatment).
      • 'brochure' and 'multibuy' are one-hot encoded to ensure stable columns.
      • We accept either 'product_number' or 'product_id' in tx; will normalize to 'product_id'.
    """
    if tx.empty or not nodes:
        return pd.DataFrame(columns=["week_start", "group", "product_id", "level", "y"])

    df = tx.copy()
    # Normalize product id column name
    if "product_id" not in df.columns and "product_number" in df.columns:
        df = df.rename(columns={"product_number": "product_id"})

    # Restrict to cohort stores
    df["store_id"] = df["store_id"].astype(str)
    store_set = set(map(str, store_list))
    df = df[df["store_id"].isin(store_set)].copy()

    # Ensure week_start exists & is W-WED
    if "week_start" not in df.columns:
        if "date" not in df.columns:
            raise KeyError("Expect 'week_start' or 'date' column to construct weekly series.")
        df["week_start"] = pd.to_datetime(df["date"]).dt.to_period("W-WED").dt.start_time
    df["week_start"] = pd.to_datetime(df["week_start"])

    # Choose target column
    target_col = _resolve_target_column(df, target)

    # Regressors (exogenous)
    reg_num = [c for c in _NUMERIC_REG if c in df.columns]
    reg_cat = [c for c in _CATEGORICAL_REG if c in df.columns]

    # Default fill for categoricals
    if "brochure" in df.columns:
        df["brochure"] = df["brochure"].fillna("Not on brochure")
    if "multibuy" in df.columns:
        df["multibuy"] = df["multibuy"].fillna("Not on Multibuy")

    # Weekly reducer for regressors
    reg_week = df[["week_start"]].drop_duplicates()
    if reg_num:
        reg_week = reg_week.merge(
            df[["week_start"] + reg_num].groupby("week_start", as_index=False).mean(),
            on="week_start", how="left"
        )
    for c in reg_cat:
        mode_c = (
            df[["week_start", c]]
            .groupby("week_start")[c]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else ("Not on brochure" if c == "brochure" else "Not on Multibuy"))
        )
        reg_week = reg_week.merge(mode_c.reset_index(), on="week_start", how="left")

    # One-hot encode categoricals at the weekly grain to stabilize columns across nodes
    if "brochure" in reg_week.columns:
        reg_week = pd.get_dummies(reg_week, columns=["brochure"], prefix="brochure")
    if "multibuy" in reg_week.columns:
        reg_week = pd.get_dummies(reg_week, columns=["multibuy"], prefix="multibuy")

    # Aggregate per node
    out_frames: List[pd.DataFrame] = []
    base = df.rename(columns={"product_number": "product_id"})  # in case either is present

    for node in nodes:
        mask = _mask_node(base, node)
        if not mask.any():
            continue
        agg = (
            base.loc[mask, ["week_start", target_col]]
            .groupby("week_start", as_index=False)
            .sum()
            .rename(columns={target_col: "y"})
        )
        if agg.empty:
            continue
        agg["group"] = make_node_key(node.level, node.key)
        agg["product_id"] = "ALL"
        agg["level"] = node.level

        # Attach regressors
        agg = agg.merge(reg_week, on="week_start", how="left")

        # Fill numeric NaNs
        num_cols = agg.select_dtypes(include=[np.number]).columns
        agg[num_cols] = agg[num_cols].fillna(0.0)

        out_frames.append(agg)

    if not out_frames:
        return pd.DataFrame(columns=["week_start", "group", "product_id", "level", "y"])

    out = pd.concat(out_frames, ignore_index=True)
    return out.sort_values(["group", "week_start"]).reset_index(drop=True)
