# SolidWorks Prospect Scoring (Phase A)

## Overview
- Division: SolidWorks (CRE)
- Objective: rank NetSuite prospects by probability of first SolidWorks purchase within 6 months.
- Scope: Active prospects (non-customers) sourced from `dim_ns_customer`. Weekly scoring, monthly retraining.

## Data Sources
- `dim_ns_customer`: extended with NetSuite attributes, first_* dates, account type flags.
- `dim_customer_enriched`: curated view for existing customer workflows (unchanged by prospect scoring).
- No transactional facts used to avoid leakage.

## Modules Added
- `gosales/labels/prospects_solidworks.py` — builds time-based prospect labels.
- `gosales/features/prospects_solidworks.py` — derives cold-start features from NetSuite data.
- `gosales/models/train_prospects_solidworks.py` — trains LightGBM plus isotonic calibration, stores artifacts under `models/prospects/solidworks/`.
- `gosales/pipeline/score_prospects_solidworks.py` — weekly scorer writing to `scores_prospects_solidworks` table and parquet export.

## Key Features
- Account age, days since last CRM update.
- Contactability indicators (email/phone/url/weblead) and lead source info.
- Territory/region categorical encoding.
- Cross-division history flags and recency (CPE/HW/3DX first-purchase dates relative to cutoff).
- Named account and strategic flags, known competitor presence.

## Labels
- Snapshots: month-end cutoffs (default 24 months history).
- Positive if `ns_first_cre_date` falls within (cutoff, cutoff + 6 months].
- Excludes existing SolidWorks customers and inactive accounts.

## Training
- Time-based holdout (last 3 cutoffs for validation).
- LightGBM with class balancing followed by isotonic calibration.
- Metrics stored in `models/prospects/solidworks/metrics.json` and feature importance CSV.

## Scoring
- CLI: `python -m gosales.pipeline.score_prospects_solidworks --cutoff YYYY-MM-DD --top-k 100`.
- Outputs: curated table `scores_prospects_solidworks`, parquet snapshot under `outputs/scores/`.
- Includes global and territory rank plus core feature snippets for actionability.

## Next Steps
1. Validate model lift with sales operations (contact -> opportunity -> first PO).
2. Tune territory-level thresholds / top-K; consider capacity constraints.
3. Add monitoring (drift on territory mix, calibration) under `monitoring/`.
4. Expand to additional divisions (CPE, HW) using shared modules.
5. Integrate outputs into UI for territory planning.
