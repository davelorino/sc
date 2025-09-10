from __future__ import annotations
import re
import pandas as pd
from typing import List

def to_list_flexible(x) -> List[str]:
    """Parse array-like strings in your media table robustly."""
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    s = str(x).strip()
    if s == "": return []
    if s.startswith("[") and s.endswith("]"):
        tokens = re.findall(r"'([^']+)'|\"([^\"]+)\"", s[1:-1])
        items = [a or b for a, b in tokens]
        if items:
            return [t.strip() for t in items if t.strip()]
        s = s[1:-1]
    parts = [p.strip().strip("'\"") for p in (s.split(",") if "," in s else s.split())]
    return [p for p in parts if p]
