from __future__ import annotations
from typing import List, Dict, Set
import pandas as pd
from src.ri.utils.dates import week_range_wed
from .parse_media_table import to_list_flexible

def get_fleet_stores(tx_master_df: pd.DataFrame) -> List[str]:
    return sorted(tx_master_df["store_id"].astype(str).unique().tolist())

def build_weekly_store_calendar_for_campaign(
    media_rows_one_campaign: pd.DataFrame,
    fleet_stores: List[str]
) -> pd.DataFrame:
    """
    Returns rows: ['week_start','store_id','is_covered'] for that campaign.
    An empty `sorted_store_list` ⇒ full fleet coverage for the asset's active weeks.
    Union across assets per week.
    """
    if media_rows_one_campaign.empty:
        return pd.DataFrame(columns=["week_start","store_id","is_covered"])

    fleet_set: Set[str] = set(map(str, fleet_stores))
    cover_map: Dict[pd.Timestamp, Set[str]] = {}

    for _, r in media_rows_one_campaign.iterrows():
        wks = week_range_wed(r["media_start_date"], r["media_end_date"])
        stores = set(map(str, to_list_flexible(r.get("sorted_store_list"))))
        is_fleet = (len(stores) == 0)
        for wk in wks:
            if wk not in cover_map:
                cover_map[wk] = set()
            if is_fleet:
                cover_map[wk] = set(fleet_set)
            else:
                if len(cover_map[wk]) < len(fleet_set):
                    cover_map[wk].update(stores)

    rows = []
    for wk, stores in cover_map.items():
        for sid in stores:
            rows.append({"week_start": wk, "store_id": str(sid), "is_covered": 1})
    cal = pd.DataFrame(rows).sort_values(["week_start","store_id"]).reset_index(drop=True)
    return cal

def weeks_to_store_sets(calendar_df: pd.DataFrame) -> Dict[pd.Timestamp, List[str]]:
    """Map each week to its covered store list (strings)."""
    out: Dict[pd.Timestamp, List[str]] = {}
    for wk, g in calendar_df.groupby("week_start"):
        out[wk] = g["store_id"].astype(str).tolist()
    return out
