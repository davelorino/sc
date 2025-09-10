#!/usr/bin/env python
from __future__ import annotations
import argparse
import pandas as pd
from src.ri.insights.weekly_join import prepare_weekly_uplift

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare_parquet", nargs="+", required=True)
    ap.add_argument("--out_csv", default="4_outputs/reports/weekly_uplift.csv")
    args = ap.parse_args()

    frames = []
    for p in args.compare_parquet:
        df = pd.read_parquet(p)
        frames.append(prepare_weekly_uplift(df))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)

if __name__ == "__main__":
    main()
