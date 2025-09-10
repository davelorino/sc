from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable

def compute_preperiod_market_share(
    *,
    tx_master_df: pd.DataFrame,
    stores: Iterable[str],
    train_end_date,          # pd.Timestamp
    lookback_weeks: int = 13
) -> pd.DataFrame:
    """
    Compute brand×category *pre-period* market share on the specified stores.

    Returns: ['brand','category','sales_brand','sales_cat','market_share','is_leader']
    """
    df = tx_master_df.copy()
    df["store_id"] = df["store_id"].astype(str)
    df = df[df["store_id"].isin(set(map(str, stores)))]
    df["week_start"] = pd.to_datetime(df["week_start"])

    start = pd.to_datetime(train_end_date) - pd.Timedelta(weeks=lookback_weeks)
    pre = df[(df["week_start"] >= start) & (df["week_start"] < pd.to_datetime(train_end_date))]

    brand_cat = (pre.groupby(["BrandDescription","CategoryDescription"], as_index=False)["sales"].sum()
                   .rename(columns={"BrandDescription":"brand","CategoryDescription":"category","sales":"sales_brand"}))
    cat_total = (pre.groupby(["CategoryDescription"], as_index=False)["sales"].sum()
                   .rename(columns={"CategoryDescription":"category","sales":"sales_cat"}))
    out = brand_cat.merge(cat_total, on="category", how="left")
    out["market_share"] = np.where(out["sales_cat"] > 0, out["sales_brand"]/out["sales_cat"], 0.0)
    out["rank_in_cat"] = out.groupby("category")["market_share"].rank(method="min", ascending=False)
    out["is_leader"]   = (out["rank_in_cat"] == 1).astype(int)
    return out[["brand","category","sales_brand","sales_cat","market_share","is_leader"]]
