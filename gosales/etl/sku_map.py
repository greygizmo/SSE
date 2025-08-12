from __future__ import annotations

from typing import Dict, Iterable, Tuple
from pathlib import Path
import csv
from gosales.utils.paths import ROOT_DIR


def get_sku_mapping() -> Dict[str, Dict[str, str]]:
    """Return the canonical SKU→(qty_col, division) mapping used for unpivoting.

    The keys are GP columns in the raw `sales_log`, values include the paired
    quantity column and the canonical division name.
    """
    base = {
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

        # Simulation and related
        # Plastics: quantity-only; GP often captured elsewhere (HV_Simulation). Keep under Simulation for seat signals
        "SW_Plastics": {"qty_col": "SW_Plastics_Qty", "division": "Simulation"},

        # Additive Manufacturing
        # Post_Processing as separate subdivision
        "Post_Processing": {"qty_col": "Post_Processing_Qty", "division": "Post_Processing"},
        # AM Software and alias of 3DP software qty
        "AM_Software": {"qty_col": "AM_Software_Qty", "division": "AM_Software"},

        # FDM Printers: keep Fortus canonical; alias legacy printer SKUs to Fortus in ETL normalization
        "Fortus": {"qty_col": "Fortus_Qty", "division": "FDM"},

        # PLM / 3DEXPERIENCE (CPE)
        "HV_Simulation": {"qty_col": "HV_Simulation_Qty", "division": "CPE"},
        "CATIA": {"qty_col": "CATIA_Qty", "division": "CPE"},
        "Delmia_Apriso": {"qty_col": "Delmia_Apriso_Qty", "division": "CPE"},
    }
    # Apply optional overrides
    try:
        overrides_path = ROOT_DIR / "data" / "lookup" / "sku_map_overrides.csv"
        if overrides_path.exists():
            with open(overrides_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gp_col = (row.get("gp_col") or row.get("gp") or row.get("sku") or "").strip()
                    qty_col = (row.get("qty_col") or row.get("qty") or "").strip()
                    division = (row.get("division") or row.get("product_division") or "").strip()
                    if not gp_col or not qty_col or not division:
                        continue
                    base[gp_col] = {"qty_col": qty_col, "division": division}
    except Exception:
        # Non-fatal if overrides unreadable
        pass
    return base


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
    # Preserve order and uniqueness
    return tuple(dict.fromkeys(list(iter_required_columns())))


def division_set() -> Tuple[str, ...]:
    mapping = get_sku_mapping()
    return tuple(sorted({meta["division"] for meta in mapping.values()}))


def sku_to_division(sku: str) -> str:
    m = get_sku_mapping()
    if sku in m:
        return m[sku]["division"]
    return "UNKNOWN"






