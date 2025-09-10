from __future__ import annotations
import pandas as pd
from typing import List

def week_wed(dt) -> pd.Timestamp:
    return pd.Timestamp(dt).to_period("W-WED").start_time

def week_range_wed(start_dt, end_dt) -> List[pd.Timestamp]:
    start = week_wed(start_dt); end = week_wed(end_dt)
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += pd.Timedelta(days=7)
    return out
