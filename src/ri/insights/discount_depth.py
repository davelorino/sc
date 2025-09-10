# src/ri/insights/discount_depth.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable

def _normalize_discount(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "discountpercent" in d.columns and d["discountpercent"].max() > 1.5:
        d["discountpercent"] = d["discountpercent"] / 100.0
    return d

def compute_discount_depth(
    tx: pd.DataFrame,
    stores: Iterable[str],
    by: str = "brand_category",  # 'sku'|'brand'|'brand_category'|'brand_subcategory'|'category'|'subcategory'
) -> pd.DataFrame:
    """
    Returns weekly sales-weighted discount depth at the requested grain.
    Output column name is 'disc_depth' with a suffix per grain to ease merges.
    """
    df = tx.copy()
    df["store_id"] = df["store_id"].astype(str)
    df = df[df["store_id"].isin(set(map(str, stores)))]
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = _normalize_discount(df)

    if by == "sku":
        key = ["week_start","product_number"]
        outcol = "disc_depth_sku"
    elif by == "brand":
        key = ["week_start","BrandDescription"]
        outcol = "disc_depth_brand"
    elif by == "brand_category":
        key = ["week_start","BrandDescription","CategoryDescription"]
        outcol = "disc_depth_bc"
    elif by == "brand_subcategory":
        key = ["week_start","BrandDescription","SubCategoryDescription"]
        outcol = "disc_depth_bsc"
    elif by == "category":
        key = ["week_start","CategoryDescription"]
        outcol = "disc_depth_category"
    elif by == "subcategory":
        key = ["week_start","SubCategoryDescription"]
        outcol = "disc_depth_subcategory"
    else:
        raise ValueError(by)

    def _wavg(g):
        w = (g["sales"].abs() + 1e-9)
        return float(np.average(g["discountpercent"], weights=w))

    out = (df.groupby(key, as_index=False)
             .apply(lambda g: pd.Series({outcol: _wavg(g)}))
             .reset_index())
    # Rename keys to harmonized names used elsewhere
    ren = {
        "BrandDescription": "brand",
        "CategoryDescription": "category",
        "SubCategoryDescription": "subcategory",
        "product_number": "product_id"
    }
    out = out.rename(columns=ren)
    return out
