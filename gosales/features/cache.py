from __future__ import annotations

from pathlib import Path
import pandas as pd
import polars as pl

from gosales.utils.paths import OUTPUTS_DIR


def build_customer_month_aggregates(engine, cutoff: str) -> pl.DataFrame:
    df = pd.read_sql("SELECT customer_id, order_date, product_division, gross_profit FROM fact_transactions", engine)
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff)
    df = df[df['order_date'] <= cutoff_dt]
    df['ym'] = df['order_date'].values.astype('datetime64[M]')
    agg = df.groupby(['customer_id','ym']).agg(month_gp=('gross_profit','sum'), month_tx=('gross_profit','count')).reset_index()
    pdf = pl.from_pandas(agg)
    out = OUTPUTS_DIR / f"customer_month_{cutoff}.parquet"
    pdf.write_parquet(out)
    return pdf


