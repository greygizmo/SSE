# Targets and SKU Taxonomy

This document explains how SKUs are mapped to reporting divisions and logical target models.

## Mapping Fields

- `qty_col`: paired quantity column from the raw sales log (keys in the map are GP columns).
- `division`: reporting division used for analytics and reporting.
- `family`: product family for cross-product signals (e.g., SWX, PDM, Hardware).
- `sale_type`: coarse type used by modeling (License, Printer, Consumable, Maintenance, Training, Support_Subscription, etc.).
- `db_division_routes` (optional): route division based on the source DB `Division` field (e.g., `AM_Support` → Scanning if `Division='Scanning'`).

See `gosales/etl/sku_map.py` for the full mapping.

## Reporting Divisions

- Solidworks (SWX licenses and modules)
- PDM (PDM/EPDM seats)
- Simulation
- Services
- Training
- Success Plan
- Hardware (printers and ecosystem spend)
- CPE (3DEXPERIENCE: CATIA/DELMIA/HV_Simulation)
- Scanning (Creaform, Artec)
- CAMWorks
- Maintenance (UAP, support add-ons)

## Logical Target Models (by SKU sets)

Use `gosales.etl.sku_map.get_model_targets(name)` to resolve SKUs.

- Printers: Formlabs, FDM, SAF, SLA, P3, Metals, PolyJet, Fortus, uPrint, _1200_Elite_Fortus250
- SWX_Seats: SWX_Core, SWX_Pro_Prem
- PDM_Seats: PDM, EPDM_CAD_Editor_Seats
- SW_Electrical: SW_Electrical
- SW_Inspection: SW_Inspection
- Services: Services
- Success_Plan: Success_Plan
- Training: Training
- Simulation: Simulation, SW_Plastics
- Scanning: Creaform, Artec
- CAMWorks: CAMWorks

Notes:
- UAP (Core_New_UAP, Pro_Prem_New_UAP) is mapped to `division=Maintenance` and acts as a predictor (not a target) for SWX_Seats.
- AM_Support is routed via `db_division_routes` (Scanning vs Hardware) and is treated as a predictor.
- Consumables (`Consumables`) and Post_Processing/AM_Software are predictors for the Printers model.

## Eventization

Invoice-level events are built by `gosales/etl/events.py` into `fact_events`:
- Events are `(invoice_id, customer_id, order_date)` groups with per-model labels `label_<Model>` and aggregates `qty_<Model>`, `gp_<Model>`.
- Use these to audit leakage and for advanced feature engineering that excludes same-invoice contributions.

## Training Targets in `score_all`

The orchestrator collects targets as:
- All reporting divisions except `Hardware` and `Maintenance`, and
- All supported models from `get_supported_models()`.

You can train a specific target directly via:

```
python -m gosales.models.train --division Printers --cutoffs "2024-06-30" --window-months 6
```

