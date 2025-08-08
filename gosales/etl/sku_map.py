from __future__ import annotations

from typing import Dict, Iterable, Tuple


def get_sku_mapping() -> Dict[str, Dict[str, str]]:
    """Return the canonical SKU→(qty_col, division) mapping used for unpivoting.

    The keys are GP columns in the raw `sales_log`, values include the paired
    quantity column and the canonical division name.
    """
    return {
        "SWX_Core": {"qty_col": "SWX_Core_Qty", "division": "Solidworks"},
        "SWX_Pro_Prem": {
            "qty_col": "SWX_Pro_Prem_Qty",
            "division": "Solidworks",
        },
        "Core_New_UAP": {
            "qty_col": "Core_New_UAP_Qty",
            "division": "Solidworks",
        },
        "Pro_Prem_New_UAP": {
            "qty_col": "Pro_Prem_New_UAP_Qty",
            "division": "Solidworks",
        },
        "PDM": {"qty_col": "PDM_Qty", "division": "Solidworks"},
        "Simulation": {"qty_col": "Simulation_Qty", "division": "Simulation"},
        "Services": {"qty_col": "Services_Qty", "division": "Services"},
        "Training": {"qty_col": "Training_Qty", "division": "Services"},
        "Success Plan GP": {
            "qty_col": "Success_Plan_Qty",
            "division": "Services",
        },
        # Assuming supplies = consumables
        "Supplies": {"qty_col": "Consumables_Qty", "division": "Hardware"},
    }


def iter_required_columns() -> Iterable[str]:
    """Yield the minimal set of raw columns required for star-build unpivoting.

    Includes identifying columns and all GP/Qty columns present in the mapping.
    """
    id_vars = ["CustomerId", "Rec Date", "Division"]
    mapping = get_sku_mapping()
    yield from id_vars
    for gp_col, meta in mapping.items():
        yield gp_col
        yield meta["qty_col"]


def get_required_columns() -> Tuple[str, ...]:
    """Return the required columns as an ordered tuple for contracts."""
    return tuple(dict.fromkeys(iter_required_columns()).keys())






