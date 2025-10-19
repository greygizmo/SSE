import pandas as pd
import polars as pl
import sqlalchemy as sa
from gosales.whitespace.als import build_als


def test_build_als_outputs_readable_ids(tmp_path):
    engine = sa.create_engine('sqlite://')
    data = pd.DataFrame({'customer_id': ['0001', '0001', '0002'], 'product_sku': ['A', 'B', 'A']})
    data.to_sql('fact_transactions', engine, if_exists='replace', index=False, dtype={'customer_id': sa.String()})
    output_path = tmp_path / 'als.csv'
    build_als(engine, output_path)
    result = pl.read_csv(output_path, schema_overrides={"customer_id": pl.Utf8})
    assert {'customer_id', 'product_name'} <= set(result.columns)
    assert set(result['product_name'].to_list()) <= {'A', 'B'}
    customer_ids = result['customer_id'].to_list()
    assert set(customer_ids) <= {'0001', '0002'}
    assert all(isinstance(cid, str) for cid in customer_ids)

