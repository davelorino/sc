#!/usr/bin/env python
from __future__ import annotations
from google.cloud import bigquery
from src.ri.get_data.transaction_pipeline import create_weekly_transactions_master_table

if __name__ == "__main__": 
    client = bigquery.Client()
    print(create_weekly_transactions_master_table(client))
