import polars as pl
import sqlalchemy as sa
from gosales.whitespace.als import build_als


def test_build_als_outputs_readable_ids(tmp_path):
    engine = sa.create_engine('sqlite://')
    data = pl.DataFrame({'customer_id': [1, 1, 2], 'product_name': ['A', 'B', 'A']})
    data.write_database('fact_orders', engine, if_table_exists='replace')
    output_path = tmp_path / 'als.csv'
    build_als(engine, output_path)
    result = pl.read_csv(output_path)
    assert {'customer_id', 'product_name'} <= set(result.columns)
    assert set(result['product_name'].to_list()) <= {'A', 'B'}
    assert set(result['customer_id'].to_list()) <= {1, 2}

