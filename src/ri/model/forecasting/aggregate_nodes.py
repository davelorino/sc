# src/ri/model/forecasting/aggregate_nodes.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from src.ri.model.structures import NodeSpec, make_node_key

# Exogenous regressors (weekly averages/shares)
_NUMERIC_REG = [
    "discountpercent",
    "max_internal_competitor_discount_percent",
    "n_competitors", "n_any_cheaper", "n_shelf_cheaper",
    "n_promo_cheaper_no_hurdle", "n_promo_cheaper_hurdle",
    "avg_cheaper_gap", "worst_gap", "p90_gap",
    # cohort-level shares (numeric)
    "brochure_Not on brochure", "multibuy_Not on Multibuy",
]
_CATEGORICAL_REG = ["brochure", "multibuy"]  # legacy path; often absent in cohort totals

# Targets: canonical column names
_TARGET_MAP = {
    "sales": "sales",
    "shoppers": "shoppers",
    "new_to_sku_sales": "new_to_sku_sales_13m",
    "new_to_sku_shoppers": "new_to_sku_shoppers_13m",
    "new_to_brand_sales": "new_to_brand_sales_13m",
    "new_to_brand_shoppers": "new_to_brand_shoppers_13m",
    "new_to_category_sales": "new_to_category_sales_13m",
    "new_to_category_shoppers": "new_to_category_shoppers_13m",
    "new_to_subcategory_sales": "new_to_subcategory_sales_13m",
    "new_to_subcategory_shoppers": "new_to_subcategory_shoppers_13m",
}

def _resolve_target_column(df: pd.DataFrame, target_key: str) -> str:
    col = _TARGET_MAP.get(target_key)
    if col is None:
        raise ValueError(f"Unknown target '{target_key}'. Valid keys: {sorted(_TARGET_MAP.keys())}")
    if col not in df.columns:
        raise KeyError(
            f"Target column '{col}' expected by metric '{target_key}' "
            f"is missing from the transactions dataframe."
        )
    return col

def _mask_node(df: pd.DataFrame, node: NodeSpec) -> pd.Series:
    """
    Boolean mask for rows belonging to a node.

    IMPORTANT: For cohort-summed frames (no per-SKU identity), we treat SKU nodes
    as applying to the entire frame. That means:
      • If 'product_id' is missing OR entirely NaN -> mask == True for SKU nodes.
    """
    if node.level == "sku":
        if ("product_id" not in df.columns) or df["product_id"].isna().all():
            # Cohort-sum mode (no per-SKU identity in this frame)
            return pd.Series(True, index=df.index)
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

def build_weekly_series_for_stores(
    tx: pd.DataFrame,
    store_list: List[str],
    nodes: List[NodeSpec],
    target: str,  # one of _TARGET_MAP keys
) -> pd.DataFrame:
    """
    Build a stacked weekly panel for the specified nodes/target.

    Works with:
      • legacy per-store frames (filters by store_list when 'store_id' exists)
      • cohort-summed frames (no 'store_id' -> no filtering; SKU nodes apply to whole frame)
    """
    if tx.empty or not nodes:
        return pd.DataFrame(columns=["week_start", "group", "product_id", "level", "y"])

    df = tx.copy()

    # Normalize product id column name (if present)
    if "product_id" not in df.columns and "product_number" in df.columns:
        df = df.rename(columns={"product_number": "product_id"})

    # Legacy store filtering (skip for cohort totals)
    if "store_id" in df.columns and store_list:
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

    # Fill legacy categoricals if present
    if "brochure" in df.columns:
        df["brochure"] = df["brochure"].fillna("Not on brochure")
    if "multibuy" in df.columns:
        df["multibuy"] = df["multibuy"].fillna("Not on Multibuy")

    # Weekly reducer for regressors
    reg_week = df[["week_start"]].drop_duplicates()
    if reg_num:
        # If some numeric columns came in as strings (e.g., "0E-9"), coerce safely
        for c in reg_num:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        reg_week = reg_week.merge(
            df[["week_start"] + reg_num].groupby("week_start", as_index=False).mean(numeric_only=True),
            on="week_start", how="left"
        )
    for c in reg_cat:
        mode_c = (
            df[["week_start", c]]
            .groupby("week_start")[c]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else ("Not on brochure" if c == "brochure" else "Not on Multibuy"))
        )
        reg_week = reg_week.merge(mode_c.reset_index(), on="week_start", how="left")

    # One-hot encode legacy categoricals (if present)
    if "brochure" in reg_week.columns:
        reg_week = pd.get_dummies(reg_week, columns=["brochure"], prefix="brochure")
    if "multibuy" in reg_week.columns:
        reg_week = pd.get_dummies(reg_week, columns=["multibuy"], prefix="multibuy")

    # If product_id missing OR all NaN, collapse any SKU nodes to a single "ALL" node
    sku_nodes_exist = any(n.level == "sku" for n in nodes)
    no_sku_identity = ("product_id" not in df.columns) or df["product_id"].isna().all()
    if sku_nodes_exist and no_sku_identity:
        nodes = [n for n in nodes if n.level != "sku"] + [NodeSpec(level="sku", key=("ALL",))]

    out_frames: List[pd.DataFrame] = []
    base = df.rename(columns={"product_number": "product_id"})  # tolerate either

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
        agg["product_id"] = "ALL"  # identity is captured in 'group'
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
