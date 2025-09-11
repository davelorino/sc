from __future__ import annotations
from typing import List, Dict, Set
import pandas as pd
from src.ri.utils.dates import week_range_wed
from .parse_media_table import to_list_flexible

def get_fleet_stores(tx_master_df: pd.DataFrame) -> List[str]:
    return sorted(tx_master_df["store_id"].astype(str).unique().tolist())

def _normalize_store_list(stores: Optional[Iterable]) -> List[str]:
    if not stores:
        return []
    out, seen = [], set()
    for s in stores:
        if s is None:
            continue
        v = str(s).strip()
        if v and v.lower() != "nan" and v not in seen:
            seen.add(v)
            out.append(v)
    return out

def build_weekly_store_calendar_for_campaign(
    media_rows_one_campaign: pd.DataFrame,
    fleet_stores: List[str],
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      week_start (Timestamp, W-WED), store_id (str), is_covered (int 0/1)

    Semantics:
      • If fleet_stores is empty  -> return an EMPTY DF with the correct columns.
        (Runner interprets this as “ALL STORES” downstream.)
      • Otherwise, emit every (week, store) with is_covered=1.
    """
    cols = ["week_start", "store_id", "is_covered"]

    if media_rows_one_campaign is None or media_rows_one_campaign.empty:
        return pd.DataFrame(columns=cols)

    if "media_start_date" not in media_rows_one_campaign.columns or "media_end_date" not in media_rows_one_campaign.columns:
        return pd.DataFrame(columns=cols)

    start = pd.to_datetime(media_rows_one_campaign["media_start_date"]).min()
    end   = pd.to_datetime(media_rows_one_campaign["media_end_date"]).max()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=cols)

    start_w = pd.Period(start, freq="W-WED").start_time
    end_w   = pd.Period(end,   freq="W-WED").start_time
    week_starts = pd.period_range(start=start_w, end=end_w, freq="W-WED").start_time

    fleet = _normalize_store_list(fleet_stores)

    # IMPORTANT: empty fleet => return EMPTY CALENDAR => “ALL STORES”
    if not fleet or len(week_starts) == 0:
        return pd.DataFrame(columns=cols)

    rows = [{"week_start": wk, "store_id": sid, "is_covered": 1}
            for wk in week_starts for sid in fleet]

    cal = pd.DataFrame.from_records(rows, columns=cols)
    if not cal.empty:
        cal = cal.sort_values(["week_start", "store_id"]).reset_index(drop=True)
    return cal

def weeks_to_store_sets(calendar_df: pd.DataFrame) -> Dict[pd.Timestamp, List[str]]:
    """
    { week_start : [store_id, ...] }
    Empty or malformed input -> {}
    """
    need = {"week_start", "store_id"}
    if calendar_df is None or calendar_df.empty or not need.issubset(set(calendar_df.columns)):
        return {}
    df = calendar_df[["week_start", "store_id"]].copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["store_id"] = df["store_id"].astype(str)
    grouped = df.groupby("week_start")["store_id"].apply(lambda s: sorted(set(s)))
    return dict(grouped.items())