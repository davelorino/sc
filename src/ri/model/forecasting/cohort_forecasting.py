from __future__ import annotations
import pandas as pd
from typing import Dict, List, Tuple
from src.ri.model.structures import NodeSpec
from src.ri.model.forecasting.aggregate_nodes import build_weekly_series_for_stores
from src.ri.model.forecasting.forecasters import fit_predict_series

def forecast_per_week_dynamic_cohorts(
    *,
    tx_master_df: pd.DataFrame,
    week_to_storelist: Dict[pd.Timestamp, List[str]],
    nodes: List[NodeSpec],
    train_end_date: pd.Timestamp,
    target: str = "sales",
    exog_regressors: List[str] = None
) -> pd.DataFrame:
    """
    For each campaign week, build a series over *that week's treated store subset*
    (across all historical weeks), train on pre-period, and predict that week.
    Returns stacked compare df: ['ds','group','level','y','yhat']
    """
    if exog_regressors is None: exog_regressors = []
    rows = []
    # Pre-compute actuals on the exact treated subset for each week
    for wk, stores in sorted(week_to_storelist.items()):
        series = build_weekly_series_for_stores(
            tx=tx_master_df.rename(columns={"product_number":"product_id"}),
            store_list=stores,
            nodes=nodes,
            target=target
        )
        if series.empty:
            continue
        series = series.rename(columns={"week_start":"ds"})
        # train end date = day before first campaign week (provided by caller)
        hist = series[series["ds"] < pd.to_datetime(train_end_date)].copy()
        this_week = series[series["ds"] == pd.to_datetime(wk)].copy()

        # per node forecast
        for g, gdf in hist.groupby("group"):
            reg_cols = [c for c in gdf.columns if c not in ("ds","y","group","product_id","level")]
            fcst = fit_predict_series(
                hist_df=gdf[["ds","y"]+reg_cols],
                forecast_ds=[pd.to_datetime(wk)],
                reg_cols=reg_cols
            )
            # actual y for that node/week
            ya = this_week[this_week["group"]==g][["ds","y","group","product_id","level"]]
            merged = ya.merge(fcst, on="ds", how="left")
            rows.append(merged)
    if not rows:
        return pd.DataFrame(columns=["ds","group","product_id","level","y","yhat"])
    out = pd.concat(rows, ignore_index=True)
    return out
