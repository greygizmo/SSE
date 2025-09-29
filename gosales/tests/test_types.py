import pandas as pd

from gosales.utils.types import DataTypeSchema, TypeEnforcer


def test_enforce_customer_id_handles_empty_pandas_df():
    df = pd.DataFrame(columns=["customer_id"])

    result = TypeEnforcer.enforce_customer_id(df)

    assert result is df


def test_enforce_customer_id_casts_pandas_values_to_string():
    df = pd.DataFrame({"customer_id": [1, 2], "value": [10.0, 20.0]})

    result = TypeEnforcer.enforce_customer_id(df)

    assert list(result["customer_id"]) == ["1", "2"]


def test_enforce_schema_handles_empty_pandas_df():
    df = pd.DataFrame(columns=["customer_id", "quantity"])
    schema = DataTypeSchema()

    result = TypeEnforcer.enforce_schema(df, schema)

    assert result is df


def test_enforce_schema_casts_pandas_columns():
    df = pd.DataFrame({"customer_id": [101, 102], "quantity": [1, 2]})
    schema = DataTypeSchema()

    result = TypeEnforcer.enforce_schema(df, schema)

    assert pd.api.types.is_string_dtype(result["customer_id"])
    assert str(result["quantity"].dtype) == "Int32"
