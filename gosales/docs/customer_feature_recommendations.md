Customer Feature Recommendations (Azure NetSuite Views)

Purpose

- Identify high‑value customer attributes from dbo.customer_customerOnly and dbo.customer_cleaned_headers that complement the curated star (fact_transactions, dim_customer, fact_assets) and are likely to improve prediction and segmentation.
- Clarify identifier alignment and propose a robust keying strategy to eliminate NULL customer_id in fact_assets.

Keying and Identifier Alignment

- Canonical key: Use NetSuite internalid as the authoritative customer identifier.
  - Validation: internalid in dbo.customer_customerOnly matches curated dim_customer.customer_id (examples checked: 10000524, 10000636, 10000741, 10001543, 10001654).
  - Action (ETL follow‑up): When building fact_assets, map customer_id by exact internalid join to dbo.customer_customerOnly; fall back to name normalization only if internalid is missing. This resolves NULLs where names diverge.
- Keep internalId_Int only as a numeric mirror of internalid; store IDs as strings in curated layers.

Recommendation Scope

- Source: dbo.customer_customerOnly (62 columns)
- Source: dbo.customer_cleaned_headers (406 columns)
- Exclusions: fields that duplicate curated features (e.g., Assets_* snapshots that are better derived as‑of cutoff from fact_assets), free‑form long text notes (comments, custnote), and transient operational artifacts unless clearly predictive.

Identity & Hierarchy

- internalid (both views): Primary join key → curated customer_id.
- entityid (both): Secondary stable text key; useful for de‑dup QA and joins to external systems.
- companyname (both): Display and fuzzy join fallback; can inform brand normalization.
- Parent_Customer, parent, Parent_Child_Account, custom_parent (cleaned_headers): Parent/child structure supports roll‑ups and group‑level features (e.g., group assets, cross‑sell at holding‑company level).

Geography & Territory

- ShippingCity, ShippingState, ShippingZip, ShippingCountry (both): Location for regional demand, distance features, regulatory/tax proxies. When the payoff justifies it we can geocode to latitude/longitude (e.g., via `geopy`) once we budget the additional API/runtime cost; until then, postal code + state should cover most segmentation needs.
- Territory_Name (customerOnly/cleaned_headers): Sales coverage segmentation; strong lift for product mix. Drop `Territory_Id` per guidance—`Territory_Name` is the human‑meaningful key we’ll carry forward.
- region (both): Coarser market segmentation; useful interaction with Territory.
- CAD_Territory, AM_Territory (cleaned_headers): Modality‑specific territories (CAD vs Additive) to align with division‑specific models.

Lifecycle & Status

- datecreated (both): Account age (days since create) is consistently predictive of adoption/renewal patterns.
- lastmodifieddate (both): Recency of CRM activity; proxy for engagement.
- entitystatus, entityStatus_Value (both): Account lifecycle status; include both raw and value label for stability across environments.
- stage, Stage_value (both): Distinguishes Customers vs. Prospects in NetSuite. Recommend deriving a boolean `is_prospect` + “prospect age” features (days since created without revenue) and using the raw labels only for QA—this can help focus outreach without leaking post‑purchase data.

Financial Health & Credit

- creditlimit (both): Customer purchasing capacity proxy.
- overduebalance, unbilledorders, balance/consolbalance (cleaned_headers): Liquidity/AR risk signals; use with guardrails to avoid post‑outcome leakage.
- taxable (both) and Terms_Value (both): Customer type and payment behavior proxy; correlate with maturity/size.

Contactability & Digital Presence

- email (both), phone (both): Presence flags; channel availability.
- url (both): Web presence; parse domain for sectoral cues; presence vs missing is predictive.
- weblead (both): Digital engagement flag; origin channel blending with leadsource.
- leadsource, leadsource_name (cleaned_headers): Acquisition channel; meaningful for upsell pathways.

Organization Size & Profile

- **Deferred (external feed)**: employees, companyrevenue/revenue, siccode — Tyler will source these from a higher‑quality provider, so we’ll omit the NetSuite versions and wire the external feed when it lands.

Sales Coverage & Roles

- salesrep_Name (both), am_sales_rep, cam_sales_rep, PDM_Sales_Rep, Sim_Sales_Rep, EDA_Sales_Rep, additive_rep, Hardware_CSM, Subscription_Service_Representative, Subscription_Service_Rep, customer_success_rep (both/cleaned_headers): Since coverage is universal, raw IDs add little. Instead capture light-weight signals such as “has_dedicated_success_rep” or cadence of reassignment if we can detect changes over time; otherwise treat these as QA references only.

Product Footprint & Tech Stack (Self‑Declared)

- Current_CAD_Software (customerOnly/cleaned_headers)
- Current_CAM_Software (both)
- Current_Simulation_Software (cleaned_headers)
- Current_PDM_Software (cleaned_headers)
- Current_ECAD_Software (cleaned_headers)
- Current_Tech_Pubs_Software (cleaned_headers)
- Current_3D_Printer, Current_3D_Print_Solution (cleaned_headers)
- Current_ERP_MRP_Software (cleaned_headers if present)
Reason: Tech stack signals whitespace/cross‑sell potential beyond what’s in transactions and assets. Treat as categorical (top‑k levels + "OTHER/UNKNOWN").

Subscription & Success Signals

- globalsubscriptionstatus (both): Global comms/subscription opt‑in; proxy for engagement/trust.
- Success_Plan_Level, Success_Plan_Date, Success_plan_used, Success_plan_total (cleaned_headers): Managed success intensity; leading indicator of retention/expansion. Time‑box to avoid leakage (as‑of cutoff only).

Industry & Strategic Flags

- Strategic_Account, Strategic_Account_Level (cleaned_headers), top_50 (customerOnly): Management prioritization; strong signal for adoption of higher‑tier offerings.
- Known_Competitor / Other_VAR / othervar (customerOnly, where populated): Competitive context; sparse but high‑value when present.
- CAD_Named_Account, SIM_Named_Account, AM_Named_Account, Electrical_Named_Account (cleaned_headers): Flagged strategic programs—treat as categorical yes/no features respecting cutoff dates.

Milestone & Success Tracking

- milestone_* columns (e.g., milestone_successplan, milestone_kickoffcall, milestone_supportused): Capture when Customer Success hits onboarding checkpoints. Use as dated events clipped at modeling cutoff to avoid forward‑looking leakage.

Data Quality & Completeness (Derived)

- Address completeness score (from Shipping* fields) and Contact completeness score (email/phone/url presence): Robust proxies for CRM hygiene; correlates with sales execution and model stability.

Columns to De‑prioritize or Handle with Care

- Assets_* columns in dbo.customer_cleaned_headers (e.g., Assets_SWx_All, Assets_PDM, Assets_SIM_All, Assets_3D_Printers, Assets_Stratasys, etc.): These are current snapshots and can leak post‑cutoff state. Prefer deriving all asset features from fact_assets with explicit cutoff logic. Use headers’ Assets_* only as cold‑start fallback when fact_assets is unavailable, with strict time‑boxing.
- Free‑form long text (comments, custnote, Hoovers_Comments, Financial_Note, mfg_notes): High preprocessing cost and potential PII; exclude unless we pursue a dedicated NLP feature track.
- Highly sparse legacy event checkbox fields (e.g., dozens of historic rollout/hand‑on flags): Consider aggregating into a single "any_event_attended" or "events_in_last_N_months" only if dates are available; otherwise de‑prioritize.
- Duplicated/synonymous columns (Other_VAR vs othervar; Stage vs Stage_value; entitystatus vs entityStatus_Value): Keep one canonical and retain the alternate as QA reference.

Feature Engineering Notes (when we integrate)

- Keying: Left join these views to dim_customer on internalid (cast to string), not on names.
- Encoding:
  - Categorical: target‑encode or one‑hot top‑k with rare bucket; ensure per‑division leakage guards.
  - Numeric: winsorize/clip and z‑score per division; log1p skewed amounts.
  - Dates: derive ages/recencies as of the modeling cutoff date.
- Governance: add contract checks (null rates, cardinality bounds) to catch schema/meaning drifts.

Proposed Minimal Column Set to Start

- Keys: internalid, entityid, companyname
- Hierarchy: Parent_Customer, parent, Parent_Child_Account
- Geo/Territory: ShippingState, ShippingZip, ShippingCountry, Territory_Name, region, CAD_Territory, AM_Territory
- Lifecycle/Status: datecreated, lastmodifieddate, entitystatus, entityStatus_Value, stage, Stage_value, derived `is_prospect`
- Financial: creditlimit, overduebalance, unbilledorders, taxable, Terms_Value
- Contact/Digital: email, phone, url, weblead, leadsource (± leadsource_name)
- Coverage: lightweight flags (e.g., has_success_rep) for salesrep assignments as needed
- Size: defer to external enrichment feed (employees, revenue, siccode)
- Tech Stack: Current_CAD_Software, Current_CAM_Software, Current_Simulation_Software, Current_PDM_Software, Current_ECAD_Software, Current_Tech_Pubs_Software, Current_3D_Printer, Current_3D_Print_Solution
- Subscription/Success: globalsubscriptionstatus, Success_Plan_Level, Success_Plan_Date, Success_plan_used, Success_plan_total
- Industry/Strategic: Strategic_Account, Strategic_Account_Level, top_50, Named_Account flags, Known_Competitor / Other_VAR / othervar
- Milestones: milestone_* columns (cutoff‑safe), Success_Plan_Level/Date/usage totals

Rationale Summary

- These columns broaden the signal beyond past transactions: they capture account structure, territory context, size, lifecycle, assigned roles, declared tech stack, and engagement—factors known to drive cross‑sell and retention. They are also relatively stable, widely populated, and low‑risk to collect.

Next Steps (design only — do not integrate yet)

- Build a thin curated dimension dim_ns_customer (internalid, entityid, companyname, territory, region, status, stage, size, key reps, tech stack, strategic flags). Join fact_assets by internalid to eliminate NULL customer_id and to enrich customer features.
- Add Pandera/Great Expectations‑style checks to enforce ID casting to string and date recency bounds for lifecycle fields.
- Pilot feature importance on a held‑out division to confirm lift from territory/lifecycle/tech‑stack groups before expanding breadth.

[Tyler's notes: we should add the milestone_* columns because they are marking the date when our Customer Success team takes a customer through a milestone (typically done in the year following a new purchase), however these must adhere to the cutoff date. Also, the "named account" columns could be helpful because they are signaling strategic accounts internally]
