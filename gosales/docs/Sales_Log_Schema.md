# Sales_Log.csv – Column Reference (v2025-08-05)

A “living” reference for the raw GoSales *Sales_Log.csv* export. This document has been updated based on the comprehensive 2023-2024 transaction history file.

---

| Group | Column(s) (exact header text) | Data Type* | Example | Meaning / Notes | **ML Relevance** |
|-------|------------------------------|------------|---------|-----------------|------------------|
| **A. Core Dates & Docs** | `Rec Date`, `Invoice_Date`, `PO_Date`, `Created_Date`, `Last_Update` | datetime | `2023-07-14` | Key transaction timestamps. `Rec Date` is the primary driver for time-series analysis. | **High** (for recency features) |
| | `Invoice`, `PO_Number` | text | `280146` | Document identifiers. | Low |
| **B. Organizational** | `Branch`, `Division` | text | `Missouri`, `Solidworks` | Selling branch and business unit. `Division` is key for segmenting models. | **High** (for filtering) |
| | `Rep`, `Manager`, `ProcessingRepresentative` | text | `Steve Vakulyuk` | Sales and operations personnel associated with the sale. | Medium (can be used for rep-level analysis later) |
| **C. Transaction Meta** | `OrderType`, `Status`, `Origination`, `SalesChannel` | text | `Term Renewal` | ERP/CRM classifications for the transaction type and its current state. | Medium (can be used as categorical features) |
| | `Notes`, `Description` | text | `Stratasys Supplies` | Free-form text fields. | Low (requires NLP to be useful) |
| **D. Deal IDs / Keys** | `Id`, `SalesOrderId`, `PurchaseOrderId`, `InvoiceId`, `CustomerId`, `OpportunityId` | int | `99811` | **Primary and foreign keys**. `CustomerId` is essential for linking all transactions to a single account. | **High** (essential for joins) |
| **E. Counterparty** | `Customer`, `_3DigitZip` | text | `497588 Crayola` | Customer name and partial location identifier. | **High** (`Customer` for identification) |
| | `Referral Rep` | text | name | Rep credited for a referral. | Low |
| **F. Core Financials** | `Revenue`, `COGS`, `GP`, `CGP*` | decimal | `2995.00` | Core financial metrics for the transaction. `CGP*` (Calculated Gross Profit) is the primary value metric. | **High** (core of monetary features) |
| | `Term GP`, `Referral_GP` | decimal | ... | Specialized GP calculations. | Medium |
| **G. SOLIDWORKS GP ($)** | `SWX_Core`, `SWX_Pro_Prem`, `Core_New_UAP`, `Pro_Prem_New_UAP` | decimal | `4000.00` | Gross Profit from core SOLIDWORKS products and their associated support (UAP) contracts. | **High** (direct inputs for 'Solidworks' model) |
| **H. SOLIDWORKS Qty** | `SWX_Core_Qty`, `SWX_Pro_Prem_Qty`, `Core_New_UAP_Qty`, `Pro_Prem_New_UAP_Qty` | int | `1` | Seat counts for core SOLIDWORKS products and UAP contracts. | **High** (essential for `seat_cagr` feature) |
| **I. Other Product GP ($)** | `Simulation`, `PDM`, `CAMWorks`, `Services`, `Training`, `Supplies`, `DraftSight`, `Post_Processing`, `AM_Software`, `HV_Simulation`, `CATIA`, `Delmia_Apriso` | decimal | `1497.50` | Gross Profit allocated to other specific products or services. `HV_Simulation`/`CATIA`/`Delmia_Apriso` belong to the new `CPE` division. | **High** |
| **J. Other Product Qty** | `Simulation_Qty`, `PDM_Qty`, `CAMWorks_Qty`, `Services_Qty`, `SW_Plastics_Qty`, `AM_Software_Qty`, `DraftSight_Qty`, `Post_Processing_Qty`, `HV_Simulation_Qty`, `CATIA_Qty`, `Delmia_Apriso_Qty` | int | `1` | Quantities for the corresponding products. `SW_Plastics_Qty` rolls into Simulation seat counts even when GP is zero. `_3DP_Software_Qty` is aliased to `AM_Software_Qty`. | **High** |
| **K. Quotas & Targets** | `GP Quota`, `SW Quota`, `Sim Quota`, `PDM Quota`, etc. | decimal | numeric | Rep's sales quotas for various products. | Low (Rep-specific, not customer-specific) |
| **L. Success Plan** | `Success Plan GP`, `Success_Plan_Qty`, `Success Plan Level`, `Success Plan Attached` | mixed | `1497.5 / Elite / TRUE` | Metrics related to the sale of an elevated customer support plan. | **High** (strong indicator of customer investment) |
| **M. Currency & FX** | `InvoiceCurrency`, `UsdCadConversionRate`, `GPCurrencyRateAdjustment` | text/dec | `USD / 1.00` | Fields for handling multi-currency transactions. | Medium (important if dealing with non-USD transactions) |
| **N. Boolean Flags** | `New`, `CurrencyMismatch`, `Success Plan Only`, `CPE_Duplicated_GP_Flag` | bool | `FALSE` | Flags indicating specific attributes of the transaction. | Medium (can be useful binary features) |

---

## Analysis for Machine Learning

Based on this comprehensive schema, here is an assessment of the most valuable columns and potential calculated features for our ICP models.

### Directly Relevant Columns for ML:

The following columns should be "unpivoted" into our `fact_transactions` table and will form the basis of our features:

*   **Identifiers:** `CustomerId`, `Rec Date`, `Division`
*   **Core Product Metrics (Solidworks):** `SWX_Core`, `SWX_Core_Qty`, `SWX_Pro_Prem`, `SWX_Pro_Prem_Qty`, `Core_New_UAP`, `Core_New_UAP_Qty`, `Pro_Prem_New_UAP`, `Pro_Prem_New_UAP_Qty`
*   **Ecosystem Metrics:** `Simulation`, `PDM`, `CAMWorks`, `Services`, `Training`, `Supplies`, `DraftSight`, `Post_Processing`, `AM_Software`, `HV_Simulation`, `CATIA`, `Delmia_Apriso`, `Success Plan GP`, `Success_Plan_Qty` and all other `_Qty` and `GP` columns. `_3DP_Software_Qty` → `AM_Software_Qty` (alias).
*   **Transaction Flags:** `New`, `Success Plan Attached`

### Proposed Calculated Features (Behavioral Metrics):

This new dataset allows for the creation of powerful behavioral features. I will add these to the feature engineering script (`gosales/features/engine.py`).

1.  **Recency & Frequency:**
    *   `days_since_last_order`: Days since the customer's most recent purchase in *any* division.
    *   `days_since_last_swx_order`: Days since the last SOLIDWORKS-specific purchase.
    *   `order_frequency__last_2y`: Total number of transactions in the last two years.

2.  **Monetary Value:**
    *   `total_gp_all_time`: Sum of all Gross Profit from this customer.
    *   `total_gp_last_4q`: Sum of Gross Profit in the last 4 quarters (a rolling year).
    *   `avg_transaction_gp`: The average Gross Profit per transaction for the customer.

3.  **Customer Growth & Scale:**
    *   `total_core_seats`: Total number of `SWX_Core_Qty` seats purchased all-time.
    *   `total_pro_prem_seats`: Total number of `SWX_Pro_Prem_Qty` seats purchased all-time.
    *   `seat_cagr_last_2y`: The compound annual growth rate of total SOLIDWORKS seats over the last 8 quarters. **(This is now possible!)**

4.  **Ecosystem Engagement:**
    *   `has_uap_support`: A binary flag (`1` or `0`) indicating if the customer has ever purchased `Core_New_UAP` or `Pro_Prem_New_UAP`.
    *   `has_success_plan`: A binary flag indicating if the customer has ever purchased a Success Plan.
    *   `product_diversity_score`: A count of the number of distinct product divisions the customer has purchased from (e.g., Solidworks, Simulation, Services).

This updated schema and feature plan provide a robust foundation for building a highly accurate ICP model for the 'Solidworks' division, and cleanly introduces new divisions such as `CPE` and `Post_Processing`, aligning ETL, features, and future models.

My next step will be to implement the ETL changes to create the `fact_transactions` table based on this new understanding.
