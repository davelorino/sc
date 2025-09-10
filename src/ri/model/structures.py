from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

@dataclass(frozen=True)
class NodeSpec:
    level: str   # 'sku','brand','brand_category','brand_subcategory','category','subcategory'
    key:   Tuple

def make_node_key(level: str, key: Tuple) -> str:
    if level == "sku":                return f"sku={key[0]}"
    if level == "brand":              return f"brand={key[0]}"
    if level == "brand_category":     return f"brand={key[0]}|cat={key[1]}"
    if level == "brand_subcategory":  return f"brand={key[0]}|subcat={key[1]}"
    if level == "category":           return f"cat={key[0]}"
    if level == "subcategory":        return f"subcat={key[0]}"
    raise ValueError(level)

def parse_group_key(group: str) -> Dict[str, Optional[str]]:
    out = {"level": None, "product_id": None, "brand": None, "category": None, "subcategory": None}
    parts = group.split("|")
    for p in parts:
        if "=" not in p: continue
        k, v = p.split("=", 1)
        if k == "sku":
            out["level"] = "sku"; out["product_id"] = v
        elif k == "brand":
            out["brand"] = v; out["level"] = out["level"] or "brand"
        elif k == "cat":
            out["category"] = v; out["level"] = out["level"] or "category"
        elif k == "subcat":
            out["subcategory"] = v; out["level"] = out["level"] or "subcategory"
    if out["brand"] and out["category"]:
        out["level"] = "brand_category"
    if out["brand"] and out["subcategory"] and not out["category"]:
        out["level"] = "brand_subcategory"
    return out
