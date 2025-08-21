"""
Data validation layer for GoSales pipeline.
Provides comprehensive data quality checks and schema validation.
"""
import pandas as pd
import polars as pl
from typing import List, Dict, Any, Union, Tuple
import logging
from gosales.utils.types import DataTypeSchema, TypeEnforcer

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def add_issue(self, column: str, issue_type: str, description: str):
        """Add a validation issue."""
        self.issues.append({
            'column': column,
            'type': issue_type,
            'description': description
        })

    def add_error(self, message: str):
        """Add an error."""
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add a warning."""
        self.warnings.append(message)


class DataValidator:
    """Comprehensive data validation for the GoSales pipeline."""

    def __init__(self):
        self.type_enforcer = TypeEnforcer()

    def validate_dataframe(self, df: Union[pd.DataFrame, pl.DataFrame],
                         name: str = "DataFrame") -> ValidationResult:
        """
        Comprehensive validation of a DataFrame.

        Args:
            df: DataFrame to validate
            name: Name for logging purposes

        Returns:
            ValidationResult with issues, warnings, and errors
        """
        result = ValidationResult()

        if df is None:
            result.add_error(f"{name} is None")
            return result

        # Check if empty
        if hasattr(df, 'is_empty'):
            is_empty = df.is_empty()
        else:
            is_empty = len(df) == 0

        if is_empty:
            result.add_warning(f"{name} is empty")
            return result

        # Basic structure validation
        self._validate_basic_structure(df, name, result)

        # Type validation
        self._validate_types(df, name, result)

        # Data quality validation
        self._validate_data_quality(df, name, result)

        return result

    def _validate_basic_structure(self, df: Union[pd.DataFrame, pl.DataFrame],
                                name: str, result: ValidationResult):
        """Validate basic DataFrame structure."""
        # Check for required columns
        required_columns = ['customer_id']
        for col in required_columns:
            if col not in df.columns:
                result.add_error(f"Required column '{col}' missing from {name}")

        # Check for null customer_ids
        if 'customer_id' in df.columns:
            try:
                if hasattr(df, 'with_columns'):  # polars
                    null_count = df.filter(pl.col('customer_id').is_null()).height
                else:  # pandas
                    null_count = df['customer_id'].isnull().sum()

                if null_count > 0:
                    result.add_issue('customer_id', 'null_values',
                                   f"Found {null_count} null customer_id values")
            except Exception as e:
                result.add_warning(f"Could not validate null customer_ids: {e}")

    def _validate_types(self, df: Union[pd.DataFrame, pl.DataFrame],
                      name: str, result: ValidationResult):
        """Validate data types."""
        schema = DataTypeSchema()

        for col, expected_type in schema.__dict__.items():
            if col in df.columns:
                try:
                    if hasattr(df, 'with_columns'):  # polars
                        actual_type = str(df[col].dtype)
                    else:  # pandas
                        actual_type = str(df[col].dtype)

                    # Check customer_id specifically
                    if col == 'customer_id':
                        if not self._is_string_type(actual_type):
                            result.add_issue(col, 'type_mismatch',
                                           f"Expected string type, got {actual_type}")

                except Exception as e:
                    result.add_warning(f"Could not validate type for {col}: {e}")

    def _validate_data_quality(self, df: Union[pd.DataFrame, pl.DataFrame],
                             name: str, result: ValidationResult):
        """Validate data quality aspects."""
        # Check for duplicate customer_ids if customer_id exists
        if 'customer_id' in df.columns:
            try:
                if hasattr(df, 'with_columns'):  # polars
                    unique_count = df.select(pl.col('customer_id').n_unique()).item()
                    total_count = df.height
                else:  # pandas
                    unique_count = df['customer_id'].nunique()
                    total_count = len(df)

                if unique_count < total_count:
                    duplicate_count = total_count - unique_count
                    result.add_issue('customer_id', 'duplicates',
                                   f"Found {duplicate_count} duplicate customer_id values")
            except Exception as e:
                result.add_warning(f"Could not validate customer_id uniqueness: {e}")

    def _is_string_type(self, type_str: str) -> bool:
        """Check if a type string represents a string type."""
        type_lower = type_str.lower()
        return ('str' in type_lower or
                'utf' in type_lower or
                'string' in type_lower or
                'object' in type_lower)

    def validate_join_compatibility(self, left_df: Union[pd.DataFrame, pl.DataFrame],
                                  right_df: Union[pd.DataFrame, pl.DataFrame],
                                  join_key: str,
                                  left_name: str = "left",
                                  right_name: str = "right") -> ValidationResult:
        """
        Validate compatibility of two DataFrames for joining.

        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            join_key: Column to join on
            left_name: Name of left DataFrame
            right_name: Name of right DataFrame

        Returns:
            ValidationResult indicating compatibility
        """
        result = ValidationResult()

        # Check if join key exists in both
        if join_key not in left_df.columns:
            result.add_error(f"Join key '{join_key}' not found in {left_name}")
        if join_key not in right_df.columns:
            result.add_error(f"Join key '{join_key}' not found in {right_name}")

        if result.errors:
            return result

        # Check type compatibility
        if not self.type_enforcer.validate_join_compatibility(left_df, right_df, join_key):
            if hasattr(left_df, 'with_columns'):  # polars
                left_type = str(left_df[join_key].dtype)
                right_type = str(right_df[join_key].dtype)
            else:  # pandas
                left_type = str(left_df[join_key].dtype)
                right_type = str(right_df[join_key].dtype)

            result.add_error(f"Join key type mismatch: {left_name} has {left_type}, {right_name} has {right_type}")

        return result

    def prepare_for_join(self, left_df: Union[pd.DataFrame, pl.DataFrame],
                        right_df: Union[pd.DataFrame, pl.DataFrame],
                        join_key: str) -> Tuple[Union[pd.DataFrame, pl.DataFrame],
                                              Union[pd.DataFrame, pl.DataFrame]]:
        """
        Prepare DataFrames for joining by ensuring type compatibility.

        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            join_key: Column to join on

        Returns:
            Tuple of (processed_left_df, processed_right_df) ready for joining
        """
        # Enforce customer_id type consistency
        processed_left = self.type_enforcer.enforce_customer_id(left_df)
        processed_right = self.type_enforcer.enforce_customer_id(right_df)

        return processed_left, processed_right
