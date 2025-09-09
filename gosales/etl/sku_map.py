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
        # SolidWorks CAD licenses (target: SWX seats)
        "SWX_Core": {
            "qty_col": "SWX_Core_Qty",
            "division": "Solidworks",
            "family": "SWX",
            "sale_type": "License",
        },
        "SWX_Pro_Prem": {
            "qty_col": "SWX_Pro_Prem_Qty",
            "division": "Solidworks",
            "family": "SWX",
            "sale_type": "License",
        },

        # SWX maintenance add-on (predictor; not a target for seats)
        "Core_New_UAP": {
            "qty_col": "Core_New_UAP_Qty",
            "division": "Maintenance",
            "family": "SWX",
            "sale_type": "Maintenance",
        },
        "Pro_Prem_New_UAP": {
            "qty_col": "Pro_Prem_New_UAP_Qty",
            "division": "Maintenance",
            "family": "SWX",
            "sale_type": "Maintenance",
        },

        # PDM seats (target: PDM seats)
        "PDM": {
            "qty_col": "PDM_Qty",
            "division": "PDM",
            "family": "PDM",
            "sale_type": "License",
        },
        # DB headers present in Azure view
        "EPDM_CAD_Editor_Seats": {
            "qty_col": "EPDM_CAD_Editor_Seats_Qty",
            "division": "PDM",
            "family": "PDM",
            "sale_type": "License",
        },

        # Simulation (target: Simulation)
        "Simulation": {
            "qty_col": "Simulation_Qty",
            "division": "Simulation",
            "family": "Simulation",
            "sale_type": "License",
        },

        # Services (target: Services)
        "Services": {
            "qty_col": "Services_Qty",
            "division": "Services",
            "family": "Services",
            "sale_type": "Service",
        },

        # Training (target: Training)
        "Training": {
            "qty_col": "Training_Qty",
            "division": "Training",
            "family": "Training",
            "sale_type": "Training",
        },

        # Success Plan (target: Success Plan)
        "Success_Plan": {
            "qty_col": "Success_Plan_Qty",
            "division": "Success Plan",
            "family": "SWX",
            "sale_type": "Support_Subscription",
        },

        # Hardware ecosystem signals (predictors for Printers)
        # Assuming supplies = consumables
        "Consumables": {
            "qty_col": "Consumables_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Consumable",
        },

        # Additive Support rollup (predictor): defaults to Hardware; route by DB Division when available
        # e.g., DB Division "Scanning" -> Scanning, DB Division "Stratasys" -> Hardware
        "AM_Support": {
            "qty_col": "AM_Support_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Support_Subscription",
            "db_division_routes": {"Scanning": "Scanning", "Stratasys": "Hardware"},
        },

        # Spare/repair T&M (predictor for Printers)
        "SpareParts_RepairParts_TimeAndMaterials": {
            "qty_col": "SpareParts_RepairParts_TimeAndMaterials_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "SpareRepair",
        },

        # Simulation and related
        # Plastics: quantity-only; GP often captured elsewhere (HV_Simulation)
        "SW_Plastics": {
            "qty_col": "SW_Plastics_Qty",
            "division": "Simulation",
            "family": "Simulation",
            "sale_type": "License",
        },

        # Additive Manufacturing ecosystem (predictors for Printers)
        "Post_Processing": {
            "qty_col": "Post_Processing_Qty",
            "division": "Post_Processing",
            "family": "Hardware",
            "sale_type": "Post_Processing",
        },
        "AM_Software": {
            "qty_col": "AM_Software_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "AM_Software",
        },
        # Treat _3DP_Software the same as AM_Software; do not create a new division
        "_3DP_Software": {
            "qty_col": "_3DP_Software_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "AM_Software",
        },

        # Printers (target: Printers) — include all series/brands provided
        # Generic series
        "FormLabs": {
            "qty_col": "FormLabs_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },
        "FDM": {
            "qty_col": "FDM_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },
        "SAF": {
            "qty_col": "SAF_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },
        "SLA": {
            "qty_col": "SLA_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },
        "P3": {
            "qty_col": "P3_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },

        # Additional printer types
        "Metals": {
            "qty_col": "Metals_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },
        "Polyjet": {
            "qty_col": "Polyjet_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },

        # Specific legacy/brand lines
        "Fortus": {
            "qty_col": "Fortus_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },
        "uPrint": {
            "qty_col": "uPrint_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },
        "_1200_Elite_Fortus250": {
            "qty_col": "_1200_Elite_Fortus250_Qty",
            "division": "Hardware",
            "family": "Hardware",
            "sale_type": "Printer",
        },

        # PLM / 3DEXPERIENCE (CPE)
        "HV_Simulation": {
            "qty_col": "HV_Simulation_Qty",
            "division": "CPE",
            "family": "CPE",
            "sale_type": "License",
        },
        "CATIA": {
            "qty_col": "CATIA_Qty",
            "division": "CPE",
            "family": "CPE",
            "sale_type": "License",
        },
        "Delmia_Apriso": {
            "qty_col": "Delmia_Apriso_Qty",
            "division": "CPE",
            "family": "CPE",
            "sale_type": "License",
        },
        "DELMIA": {
            "qty_col": "Delmia_Qty",
            "division": "CPE",
            "family": "CPE",
            "sale_type": "License",
        },

        # Scanning (targets)
        "Creaform": {
            "qty_col": "Creaform_Qty",
            "division": "Scanning",
            "family": "Scanning",
            "sale_type": "Scanner",
        },
        "Artec": {
            "qty_col": "Artec_Qty",
            "division": "Scanning",
            "family": "Scanning",
            "sale_type": "Scanner",
        },

        # CAMWorks (target)
        "CAMWorks_Seats": {
            "qty_col": "CAMWorks_Seats_Qty",
            "division": "CAMWorks",
            "family": "CAM",
            "sale_type": "License",
        },

        # Additional SolidWorks add-on targets
        "SW_Electrical": {
            "qty_col": "SW_Electrical_Qty",
            "division": "Solidworks",
            "family": "SWX",
            "sale_type": "Module",
        },
        "SW_Inspection": {
            "qty_col": "SW_Inspection_Qty",
            "division": "Solidworks",
            "family": "SWX",
            "sale_type": "Module",
        },
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


# ---- Modeling helpers -------------------------------------------------------

def get_model_targets(model: str) -> Tuple[str, ...]:
    """Return the GP column names that constitute positives for a given model.

    Models supported:
      - "SWX_Seats": SolidWorks new/additional seats (Core, Pro/Prem)
      - "PDM_Seats": PDM/EPDM seats
      - "Printers": Any printer sale across series/brands
      - "Services": Services
      - "Success_Plan": Success Plan
      - "Training": Training
      - "Simulation": Simulation (incl. SW Plastics)
    """
    m = get_sku_mapping()

    if model == "SWX_Seats":
        return tuple(x for x in ("SWX_Core", "SWX_Pro_Prem") if x in m)
    if model == "PDM_Seats":
        return tuple(x for x in ("PDM", "EPDM_CAD_Editor_Seats") if x in m)
    if model == "Printers":
        # Include series/brands used in current and legacy naming that still roll up in data.
        # Consumables and Post_Processing are predictors, not positives, so they are intentionally excluded.
        printer_keys = (
            "FormLabs",
            "FDM",
            "SAF",
            "SLA",
            "P3",
            "Metals",
            "Polyjet",
            "Fortus",
            "uPrint",
            "_1200_Elite_Fortus250",
        )
        return tuple(x for x in printer_keys if x in m)
    if model == "Services":
        return tuple(x for x in ("Services",) if x in m)
    if model == "Success_Plan":
        return tuple(x for x in ("Success_Plan",) if x in m)
    if model == "Training":
        return tuple(x for x in ("Training",) if x in m)
    if model == "Simulation":
        return tuple(x for x in ("Simulation", "SW_Plastics") if x in m)
    if model == "Scanning":
        return tuple(x for x in ("Creaform", "Artec") if x in m)
    if model == "CAMWorks":
        return tuple(x for x in ("CAMWorks_Seats",) if x in m)
    if model == "SW_Electrical":
        return tuple(x for x in ("SW_Electrical",) if x in m)
    if model == "SW_Inspection":
        return tuple(x for x in ("SW_Inspection",) if x in m)
    return tuple()


def is_printer_sku(sku: str) -> bool:
    """Return True if the sku maps to a printer sale_type."""
    meta = get_sku_mapping().get(sku)
    return bool(meta and meta.get("sale_type") == "Printer")


def get_supported_models() -> Tuple[str, ...]:
    """Return the list of logical target models supported by the mapping.

    These include SKU-based targets that may not correspond 1:1 with a single
    reporting division (e.g., "Printers" spans multiple printer SKUs).
    """
    candidates = (
        "SWX_Seats",
        "PDM_Seats",
        "Printers",
        "Services",
        "Success_Plan",
        "Training",
        "Simulation",
        "Scanning",
        "CAMWorks",
        "SW_Electrical",
        "SW_Inspection",
    )
    m = get_sku_mapping()
    # Keep only where at least one target resolves
    out = []
    for c in candidates:
        tgts = get_model_targets(c)
        if any(k in m for k in tgts):
            out.append(c)
    return tuple(out)


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


def effective_division(sku: str, db_division: str | None = None) -> str:
    """Return the effective division, optionally routing by the source DB Division.

    For SKUs like `AM_Support` that may span hardware and scanning, we use
    a small routing map when `db_division` is provided; otherwise, fall back
    to the static mapping division.
    """
    m = get_sku_mapping()
    meta = m.get(sku)
    if not meta:
        return "UNKNOWN"
    if db_division and isinstance(meta, dict) and "db_division_routes" in meta:
        route = meta.get("db_division_routes", {})
        # normalize incoming label
        key = db_division.strip()
        if key in route:
            return route[key]
    return meta.get("division", "UNKNOWN")






