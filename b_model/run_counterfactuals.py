#!/usr/bin/env python
from __future__ import annotations
import argparse
import pandas as pd
from src.ri.model.orchestration.build_grids_v3 import build_grids_for_campaigns_v3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media_csv", required=True, help="Path to media campaign table CSV")
    ap.add_argument("--tx_parquet", required=True, help="Path to transactions master (parquet or CSV)")
    ap.add_argument("--bookings", nargs="+", required=True)
    ap.add_argument("--outdir", default="4_outputs")
    args = ap.parse_args()

    media = pd.read_csv(args.media_csv, parse_dates=["campaign_start_date","campaign_end_date","media_start_date","media_end_date"])
    tx = pd.read_parquet(args.tx_parquet) if args.tx_parquet.endswith(".parquet") else pd.read_csv(args.tx_parquet, parse_dates=["week_start"])
    grids, artifacts = build_grids_for_campaigns_v3(media_master_df=media, tx_master_df=tx, booking_numbers=args.bookings)

    os.makedirs(f"{args.outdir}/reports", exist_ok=True)
    os.makedirs(f"{args.outdir}/artifacts", exist_ok=True)
    grids.to_csv(f"{args.outdir}/reports/grids.csv", index=False)
    for cid, df in artifacts.items():
        df.to_parquet(f"{args.outdir}/artifacts/{cid}_compare_sales.parquet", index=False)
    print("Done.")

if __name__ == "__main__":
    import os
    main()
