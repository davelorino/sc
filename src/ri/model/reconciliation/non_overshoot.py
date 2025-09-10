from __future__ import annotations
import numpy as np
from typing import Dict

def _cap_child(child_val: float, parent_val: float | None) -> float:
    if parent_val is None or np.isnan(parent_val): return child_val
    return np.sign(child_val) * min(abs(child_val), abs(parent_val))

def enforce_non_overshoot_grid(totals: Dict[str, float]) -> Dict[str, float]:
    t = dict(totals)
    for k in ["brand_total","category_total","subcategory_total","brand_category_total","brand_subcategory_total","promo_sku_total"]:
        if k not in t: t[k] = 0.0

    bc_parent = t["brand_total"] if abs(t["brand_total"]) < abs(t["category_total"]) else t["category_total"]
    t["brand_category_total"] = _cap_child(t["brand_category_total"], bc_parent)

    bsc_parent = t["brand_total"] if abs(t["brand_total"]) < abs(t["subcategory_total"]) else t["subcategory_total"]
    t["brand_subcategory_total"] = _cap_child(t["brand_subcategory_total"], bsc_parent)

    ps_parent = t["brand_category_total"] if abs(t["brand_category_total"]) < abs(t["brand_subcategory_total"]) else t["brand_subcategory_total"]
    t["promo_sku_total"] = _cap_child(t["promo_sku_total"], ps_parent)
    return t
