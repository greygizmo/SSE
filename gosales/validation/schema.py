from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _coerce_int(series: pd.Series) -> Tuple[pd.Series, bool]:
    s = pd.to_numeric(series, errors="coerce")
    ok = s.notna().mean() >= 0.99
    return s.astype("Int64"), bool(ok)


def _coerce_float(series: pd.Series) -> Tuple[pd.Series, bool]:
    s = pd.to_numeric(series, errors="coerce")
    ok = s.notna().mean() >= 0.99
    return s.astype(float), bool(ok)


def _is_date_like(series: pd.Series) -> bool:
    try:
        _ = pd.to_datetime(series, errors="coerce")
        return _.notna().mean() >= 0.99
    except Exception:
        return False


def validate_icp_scores_schema(csv_path: Path) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    required = [
        "customer_id",
        "division_name",
        "icp_score",
        "cutoff_date",
        "prediction_window_months",
    ]
    optional = [
        "run_id",
        "model_version",
        "customer_name",
        "bought_in_division",
    ]
    present_cols = set(df.columns)
    missing = [c for c in required if c not in present_cols]

    type_issues: List[Dict[str, object]] = []
    # customer_id int-like
    if "customer_id" in df.columns:
        _, ok = _coerce_int(df["customer_id"])
        if not ok:
            type_issues.append({"column": "customer_id", "expected": "int", "ok": False})
    # division_name str-like (non-empty)
    if "division_name" in df.columns:
        non_empty = df["division_name"].astype(str).str.len().gt(0).mean() >= 0.99
        if not non_empty:
            type_issues.append({"column": "division_name", "expected": "non-empty str", "ok": False})
    # icp_score float-like in [0,1]
    if "icp_score" in df.columns:
        col, ok = _coerce_float(df["icp_score"]) 
        if not ok:
            type_issues.append({"column": "icp_score", "expected": "float", "ok": False})
        else:
            in_bounds = ((col >= 0.0) & (col <= 1.0)).mean() >= 0.95
            if not in_bounds:
                type_issues.append({"column": "icp_score", "expected": "0<=p<=1", "ok": False})
    # cutoff_date date-like
    if "cutoff_date" in df.columns:
        if not _is_date_like(df["cutoff_date"]):
            type_issues.append({"column": "cutoff_date", "expected": "YYYY-MM-DD", "ok": False})
    # prediction_window_months int-like small positive
    if "prediction_window_months" in df.columns:
        col, ok = _coerce_int(df["prediction_window_months"])
        if not ok:
            type_issues.append({"column": "prediction_window_months", "expected": "int", "ok": False})
        else:
            positive = (col.fillna(0) >= 1).mean() >= 0.99
            if not positive:
                type_issues.append({"column": "prediction_window_months", "expected": ">=1", "ok": False})

    report: Dict[str, object] = {
        "file": str(csv_path),
        "ok": len(missing) == 0 and len(type_issues) == 0,
        "missing_columns": missing,
        "type_issues": type_issues,
        "row_count": int(len(df)),
        "optional_present": [c for c in optional if c in present_cols],
    }
    return report


def validate_whitespace_schema(csv_path: Path) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    # Accept either 'division_name' or 'division' for the division key
    has_division_name = 'division_name' in df.columns
    has_division = 'division' in df.columns
    division_col = 'division_name' if has_division_name else ('division' if has_division else None)
    required_base = [
        "customer_id",
        "score",
        "p_icp",
        "p_icp_pct",
        "lift_norm",
        "als_norm",
        "EV_norm",
        "nba_reason",
        "run_id",
    ]
    present_cols = set(df.columns)
    missing = [c for c in required_base if c not in present_cols]
    if division_col is None:
        missing.append("division_name|division")

    type_issues: List[Dict[str, object]] = []
    # customer_id int-like
    if "customer_id" in df.columns:
        _, ok = _coerce_int(df["customer_id"])
        if not ok:
            type_issues.append({"column": "customer_id", "expected": "int", "ok": False})
    # division str-like
    if division_col is not None:
        non_empty = df[division_col].astype(str).str.len().gt(0).mean() >= 0.99
        if not non_empty:
            type_issues.append({"column": division_col, "expected": "non-empty str", "ok": False})
    # numeric columns
    for col_name in ["score", "p_icp", "p_icp_pct", "lift_norm", "als_norm", "EV_norm"]:
        if col_name in df.columns:
            _, ok = _coerce_float(df[col_name])
            if not ok:
                type_issues.append({"column": col_name, "expected": "float", "ok": False})
    # explanation length and forbidden tokens check summary
    explain_issues = 0
    if "nba_reason" in df.columns:
        txt = df["nba_reason"].astype(str)
        too_long = (txt.str.len() > 150).sum()
        forbidden = {"race", "gender", "religion", "ssn", "social security", "age", "ethnicity", "disability", "veteran", "pregnan"}
        contains_forbidden = txt.str.lower().apply(lambda s: any(t in s for t in forbidden)).sum()
        explain_issues = int(too_long + contains_forbidden)
        if explain_issues > 0:
            type_issues.append({"column": "nba_reason", "expected": "<=150 chars & no sensitive tokens", "ok": False, "violations": int(explain_issues)})

    report: Dict[str, object] = {
        "file": str(csv_path),
        "ok": len(missing) == 0 and len(type_issues) == 0,
        "missing_columns": missing,
        "type_issues": type_issues,
        "row_count": int(len(df)),
        "division_column": division_col,
    }
    return report


def write_schema_report(report: Dict[str, object], out_path: Path) -> None:
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


