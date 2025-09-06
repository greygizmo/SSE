from __future__ import annotations

"""Centralized SQL templates and helpers.

All dynamic identifiers are validated via `ensure_allowed_identifier` or
`validate_identifier` to mitigate injection risks.
"""

from typing import Optional

from gosales.utils.sql import ensure_allowed_identifier, validate_identifier


def select_all(view_or_table: str, *, allowlist: Optional[set[str]] = None) -> str:
    if allowlist:
        ident = ensure_allowed_identifier(view_or_table, allowlist)
    else:
        validate_identifier(view_or_table)
        ident = view_or_table
    return f"SELECT * FROM {ident}"


def moneyball_assets_select(view: str, *, allowlist: Optional[set[str]] = None) -> str:
    if allowlist:
        ident = ensure_allowed_identifier(view, allowlist)
    else:
        validate_identifier(view)
        ident = view
    return (
        "SELECT [Customer Name] AS customer_name, [Product] AS product, [Purchase Date] AS purchase_date, "
        "[Expiration Date] AS expiration_date, [QTY] AS qty, [internalid] AS internalid, "
        "[Department] AS department, [Category] AS category, [Sub Category A] AS sub_category_a, [Sub Category B] AS sub_category_b, "
        "[Audience] AS audience, [Expired] AS expired, [Sales Rep] AS sales_rep, [CAM Sales Rep] AS cam_sales_rep, [AM Sales Rep] AS am_sales_rep, [Simulation Sales Rep] AS sim_sales_rep "
        f"FROM {ident}"
    )


def items_category_limited_select(view: str, *, allowlist: Optional[set[str]] = None) -> str:
    if allowlist:
        ident = ensure_allowed_identifier(view, allowlist)
    else:
        validate_identifier(view)
        ident = view
    return (
        "SELECT itemid, internalid, Item_Rollup, name, department_name, Category, Sub_Category_A, Sub_Category_B, Audience "
        f"FROM {ident}"
    )


def top_n_preview(table: str, dialect: str, n: int = 1, *, allowlist: Optional[set[str]] = None) -> str:
    if allowlist:
        ident = ensure_allowed_identifier(table, allowlist)
    else:
        validate_identifier(table)
        ident = table
    dname = (dialect or "").lower()
    if dname.startswith("mssql") or dname == "pyodbc":
        return f"SELECT TOP {int(n)} * FROM {ident}"
    return f"SELECT * FROM {ident} LIMIT {int(n)}"

