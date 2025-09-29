"""
Centralized type management for GoSales data pipeline.
Ensures consistent data types across all components.
"""
from dataclasses import dataclass
import pandas as pd
import polars as pl
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataTypeSchema:
    """Defines expected data types for key columns across the pipeline."""
    customer_id: str = "Utf8"
    order_date: str = "Date"
    gross_profit: str = "Float64"
    quantity: str = "Int32"
    product_division: str = "Utf8"
    product_sku: str = "Utf8"
    customer_name: str = "Utf8"


class TypeEnforcer:
    """Centralized type enforcement for DataFrames."""

    TYPE_MAPPING = {
        'polars': {
            'Utf8': pl.Utf8,
            'Int64': pl.Int64,
            'Int32': pl.Int32,
            'Float64': pl.Float64,
            'Float32': pl.Float32,
            'Date': pl.Date,
            'Datetime': pl.Datetime,
        },
        'pandas': {
            'Utf8': 'string',
            'Int64': 'Int64',
            'Int32': 'Int32',
            'Float64': 'float64',
            'Float32': 'float32',
            'Date': 'datetime64[ns]',
            'Datetime': 'datetime64[ns]',
        }
    }

    @classmethod
    def enforce_customer_id(cls, df: Union[pd.DataFrame, pl.DataFrame],
                          framework: str = 'auto') -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Ensure customer_id is consistently treated as string across all DataFrames.

        Args:
            df: DataFrame to process
            framework: 'polars', 'pandas', or 'auto'

        Returns:
            DataFrame with customer_id as string type
        """
        if cls._is_df_empty(df):
            return df

        # Auto-detect framework
        if framework == 'auto':
            framework = 'polars' if hasattr(df, 'with_columns') else 'pandas'

        try:
            if framework == 'polars':
                if "customer_id" in df.columns:
                    return df.with_columns(pl.col("customer_id").cast(pl.Utf8))
            else:  # pandas
                if "customer_id" in df.columns:
                    df = df.copy()
                    df["customer_id"] = df["customer_id"].astype(str)
                    return df
        except Exception as e:
            logger.warning(f"Failed to enforce customer_id type: {e}")

        return df

    @classmethod
    def enforce_schema(cls, df: Union[pd.DataFrame, pl.DataFrame],
                      schema: DataTypeSchema,
                      framework: str = 'auto') -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Enforce complete schema on DataFrame.

        Args:
            df: DataFrame to process
            schema: DataTypeSchema with expected types
            framework: 'polars', 'pandas', or 'auto'

        Returns:
            DataFrame with enforced schema
        """
        if cls._is_df_empty(df):
            return df

        if framework == 'auto':
            framework = 'polars' if hasattr(df, 'with_columns') else 'pandas'

        result = df
        for col, expected_type in schema.__dict__.items():
            if col in df.columns:
                try:
                    if framework == 'polars':
                        polars_type = cls.TYPE_MAPPING['polars'].get(expected_type)
                        if polars_type:
                            result = result.with_columns(pl.col(col).cast(polars_type))
                    else:
                        pandas_type = cls.TYPE_MAPPING['pandas'].get(expected_type)
                        if pandas_type:
                            result = result.copy()
                            result[col] = result[col].astype(pandas_type)
                except Exception as e:
                    logger.warning(f"Failed to enforce type for {col}: {e}")

        return result

    @staticmethod
    def _is_df_empty(df: Union[pd.DataFrame, pl.DataFrame, None]) -> bool:
        """Determine whether a dataframe-like object is empty."""
        if df is None:
            return True

        if hasattr(df, "is_empty"):
            is_empty_attr = df.is_empty
            if callable(is_empty_attr):
                return is_empty_attr()
            return bool(is_empty_attr)

        try:
            return len(df) == 0  # type: ignore[arg-type]
        except TypeError:
            return False

    @classmethod
    def validate_join_compatibility(cls, left_df: Union[pd.DataFrame, pl.DataFrame],
                                  right_df: Union[pd.DataFrame, pl.DataFrame],
                                  join_key: str) -> bool:
        """
        Validate that join keys have compatible types.

        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            join_key: Column name to check

        Returns:
            True if compatible, False otherwise
        """
        if join_key not in left_df.columns or join_key not in right_df.columns:
            return False

        # Extract dtypes
        if hasattr(left_df, 'with_columns'):  # polars
            left_type = str(left_df[join_key].dtype)
            right_type = str(right_df[join_key].dtype)
        else:  # pandas
            left_type = str(left_df[join_key].dtype)
            right_type = str(right_df[join_key].dtype)

        # Check compatibility (both should be string types or same numeric type)
        left_is_string = 'str' in left_type.lower() or 'utf' in left_type.lower()
        right_is_string = 'str' in right_type.lower() or 'utf' in right_type.lower()

        if left_is_string != right_is_string:
            logger.warning(f"Join key type mismatch: {left_type} vs {right_type}")
            return False

        return True
