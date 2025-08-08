from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pandas as pd


@dataclass
class ContractViolation:
    table_name: str
    column_name: str
    violation_type: str
    details: str


def check_required_columns(df: pd.DataFrame, table_name: str, required: Iterable[str]) -> List[ContractViolation]:
    required_set = list(required)
    missing = [c for c in required_set if c not in df.columns]
    if not missing:
        return []
    return [
        ContractViolation(
            table_name=table_name,
            column_name=col,
            violation_type="missing_column",
            details=f"Column '{col}' not found in {table_name}",
        )
        for col in missing
    ]


def check_primary_key_not_null(df: pd.DataFrame, table_name: str, pk_cols: Tuple[str, ...]) -> List[ContractViolation]:
    violations: List[ContractViolation] = []
    for col in pk_cols:
        if col not in df.columns:
            violations.append(
                ContractViolation(
                    table_name=table_name,
                    column_name=col,
                    violation_type="missing_pk_column",
                    details=f"Primary key column '{col}' not present",
                )
            )
            continue
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            violations.append(
                ContractViolation(
                    table_name=table_name,
                    column_name=col,
                    violation_type="null_in_pk",
                    details=f"{null_count} nulls found in PK column '{col}'",
                )
            )
    return violations


def check_no_duplicate_pk(df: pd.DataFrame, table_name: str, pk_cols: Tuple[str, ...]) -> List[ContractViolation]:
    if not pk_cols:
        return []
    if any(c not in df.columns for c in pk_cols):
        return []
    dup_mask = df.duplicated(subset=list(pk_cols), keep=False)
    if not dup_mask.any():
        return []
    dup_rows = int(dup_mask.sum())
    return [
        ContractViolation(
            table_name=table_name,
            column_name="|".join(pk_cols),
            violation_type="duplicate_pk",
            details=f"{dup_rows} duplicate rows by PK {pk_cols}",
        )
    ]


def violations_to_dataframe(violations: List[ContractViolation]) -> pd.DataFrame:
    if not violations:
        return pd.DataFrame(columns=["table_name", "column_name", "violation_type", "details"])
    return pd.DataFrame([v.__dict__ for v in violations])


def check_date_parse_and_bounds(
    df: pd.DataFrame,
    table_name: str,
    date_column: str,
    max_date: pd.Timestamp | None = None,
) -> List[ContractViolation]:
    violations: List[ContractViolation] = []
    if date_column not in df.columns:
        violations.append(
            ContractViolation(
                table_name=table_name,
                column_name=date_column,
                violation_type="missing_column",
                details=f"Date column '{date_column}' not found in {table_name}",
            )
        )
        return violations
    ser = df[date_column]
    parsed = pd.to_datetime(ser, errors="coerce")
    invalid_count = int(((parsed.isna()) & ser.notna()).sum())
    if invalid_count > 0:
        violations.append(
            ContractViolation(
                table_name=table_name,
                column_name=date_column,
                violation_type="invalid_date",
                details=f"{invalid_count} values could not be parsed as dates",
            )
        )
    if max_date is None:
        max_date = pd.Timestamp.today().normalize()
    over_count = int((parsed.dropna() > max_date).sum())
    if over_count > 0:
        violations.append(
            ContractViolation(
                table_name=table_name,
                column_name=date_column,
                violation_type="date_after_max",
                details=f"{over_count} dates are after allowed maximum {max_date.date()}",
            )
        )
    return violations





