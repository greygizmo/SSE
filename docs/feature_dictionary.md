# Feature Dictionary

## Overview
- **Division:** Solidworks
- **Cutoff date:** 2024-06-30 (prediction window: 6 months)
- **Model source:** LightGBM inside a calibrated pipeline (`model.pkl`) with gain-based feature importances.
- **Importance metric:** LightGBM gain, summed across all 661 engineered features.

### Family importance snapshot
| Family | Features | Total gain | Gain % | Top drivers |
| --- | ---: | ---: | ---: | --- |
| Asset footprint | 113 | 14,327.41 | 38.34% | assets_on_subs_total (10.49%)<br>assets_tenure_days (8.55%)<br>assets_active_total (3.43%) |
| Lifecycle & cadence | 17 | 7,251.54 | 19.41% | last_gap_days (6.29%)<br>gp_monthly_slope_12m (3.62%)<br>tenure_days (3.45%) |
| Branch & rep coverage | 80 | 2,755.62 | 7.37% | rep_share_cindy_tubbs (0.78%)<br>branch_share_ca_santa_ana (0.58%)<br>branch_share_indiana (0.38%) |
| Industry interactions | 48 | 2,679.98 | 7.17% | is_aerospace_and_defense_x_avg_gp (0.66%)<br>is_services_x_avg_gp (0.52%)<br>is_aerospace_and_defense_x_growth (0.50%) |
| Core transaction history | 5 | 2,628.83 | 7.03% | avg_transaction_gp (5.05%)<br>total_gp_all_time (1.82%)<br>total_transactions_all_time (0.17%) |
| Industry segmentation | 52 | 2,272.61 | 6.08% | is_services (1.27%)<br>is_mold_tool_and_die (0.38%)<br>is_medical_devices_and_life_sciences (0.37%) |
| Division mix & share | 12 | 1,356.34 | 3.63% | success plan_gp_share_12m (0.93%)<br>solidworks_gp_share_12m (0.67%)<br>training_gp_share_12m (0.63%) |
| Recent window stats | 6 | 1,212.85 | 3.25% | gp_mean_last_24m (1.65%)<br>gp_sum_last_24m (1.37%)<br>margin__all__gp_pct__24m (0.12%) |
| Binary flags | 3 | 814.56 | 2.18% | ever_bought_solidworks (2.18%)<br>ever_acr (0.00%)<br>ever_new_customer (0.00%) |
| Adjacent division engagement | 5 | 810.89 | 2.17% | hardware_transaction_count (1.17%)<br>total_training_gp (0.51%)<br>total_services_gp (0.37%) |
| RFM derived metrics | 20 | 370.23 | 0.99% | rfm__div__gp_sum__24m (0.42%)<br>rfm__div__gp_mean__24m (0.25%)<br>rfm__div__tx_n__24m (0.15%) |
| Diversity & seasonality | 10 | 354.07 | 0.95% | q1_share_24m (0.33%)<br>q2_share_24m (0.31%)<br>product_diversity_score (0.15%) |
| Market basket & affinity | 2 | 330.62 | 0.88% | mb_lift_mean_lag60d (0.59%)<br>mb_lift_max_lag60d (0.30%) |
| Growth & momentum | 1 | 203.64 | 0.54% | growth_ratio_24_over_23 (0.54%) |
| Missingness indicators | 287 | 0.00 | 0.00% | total_transactions_all_time_missing (0.00%)<br>transactions_last_2y_missing (0.00%)<br>total_gp_all_time_missing (0.00%) |

## Detailed feature reference
### Asset footprint (113 features)
Moneyball asset rollups and subscription tenure metrics evaluated as of the cutoff date.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 1 | assets_on_subs_total | Total quantity of assets currently on subscription at the cutoff. | 3,919.29 | 10.49% |
| 2 | assets_tenure_days | Days since the first effective asset purchase at the cutoff. | 3,194.59 | 8.55% |
| 7 | assets_active_total | Total quantity of active assets at the cutoff. | 1,282.58 | 3.43% |
| 9 | assets_bad_purchase_share | Share of active assets with missing or invalid purchase dates. | 876.62 | 2.35% |
| 10 | assets_rollup_swx_pro_prem | Active asset quantity for the SWX PRO Prem rollup at the cutoff. | 835.05 | 2.23% |
| 12 | assets_rollup_swx_core | Active asset quantity for the SWX Core rollup at the cutoff. | 720.01 | 1.93% |
| 17 | assets_off_subs_total | Total quantity of assets that have churned off subscription before the cutoff. | 496.05 | 1.33% |
| 22 | assets_rollup_epdm_cad_editor_seats | Active asset quantity for the Epdm CAD Editor Seats rollup at the cutoff. | 329.50 | 0.88% |
| 23 | assets_rollup_unidentified | Active asset quantity for the Unidentified rollup at the cutoff. | 327.63 | 0.88% |
| 25 | assets_off_subs_swx_pro_prem | Quantity of churned/off-subscription assets for the SWX PRO Prem rollup at the cutoff. | 284.33 | 0.76% |
| 26 | assets_off_subs_3dx_revenue | Quantity of churned/off-subscription assets for the 3DX Revenue rollup at the cutoff. | 255.18 | 0.68% |
| 32 | assets_off_subs_swx_core | Quantity of churned/off-subscription assets for the SWX Core rollup at the cutoff. | 204.29 | 0.55% |
| 34 | assets_rollup_simulation | Active asset quantity for the Simulation rollup at the cutoff. | 197.46 | 0.53% |
| 37 | assets_off_subs_unidentified | Quantity of churned/off-subscription assets for the Unidentified rollup at the cutoff. | 189.06 | 0.51% |
| 40 | assets_rollup_draftsight | Active asset quantity for the Draftsight rollup at the cutoff. | 186.87 | 0.50% |
| 42 | assets_rollup_sw_electrical | Active asset quantity for the SW Electrical rollup at the cutoff. | 167.34 | 0.45% |
| 61 | assets_off_subs_simulation | Quantity of churned/off-subscription assets for the Simulation rollup at the cutoff. | 112.65 | 0.30% |
| 65 | assets_rollup_misc_seats | Active asset quantity for the Misc Seats rollup at the cutoff. | 106.94 | 0.29% |
| 74 | assets_rollup_camworks_seats | Active asset quantity for the Camworks Seats rollup at the cutoff. | 93.13 | 0.25% |
| 77 | assets_off_subs_draftsight | Quantity of churned/off-subscription assets for the Draftsight rollup at the cutoff. | 85.47 | 0.23% |
| 84 | assets_rollup_3dx_revenue | Active asset quantity for the 3DX Revenue rollup at the cutoff. | 79.14 | 0.21% |
| 89 | assets_off_subs_epdm_cad_editor_seats | Quantity of churned/off-subscription assets for the Epdm CAD Editor Seats rollup at the cutoff. | 67.70 | 0.18% |
| 91 | assets_off_subs_misc_seats | Quantity of churned/off-subscription assets for the Misc Seats rollup at the cutoff. | 62.08 | 0.17% |
| 115 | assets_rollup_creaform | Active asset quantity for the Creaform rollup at the cutoff. | 42.21 | 0.11% |
| 119 | assets_off_subs_sw_electrical | Quantity of churned/off-subscription assets for the SW Electrical rollup at the cutoff. | 39.93 | 0.11% |
| 130 | assets_rollup_none | Active asset quantity for the None rollup at the cutoff. | 30.87 | 0.08% |
| 154 | assets_off_subs_training | Quantity of churned/off-subscription assets for the Training rollup at the cutoff. | 22.65 | 0.06% |
| 157 | assets_off_subs_none | Quantity of churned/off-subscription assets for the None rollup at the cutoff. | 21.70 | 0.06% |
| 176 | assets_rollup_service | Active asset quantity for the Service rollup at the cutoff. | 15.46 | 0.04% |
| 177 | assets_off_subs_camworks_seats | Quantity of churned/off-subscription assets for the Camworks Seats rollup at the cutoff. | 15.36 | 0.04% |
| 180 | assets_rollup_sw_inspection | Active asset quantity for the SW Inspection rollup at the cutoff. | 14.75 | 0.04% |
| 203 | assets_off_subs_creaform | Quantity of churned/off-subscription assets for the Creaform rollup at the cutoff. | 9.08 | 0.02% |
| 212 | assets_rollup_fdm | Active asset quantity for the FDM rollup at the cutoff. | 6.83 | 0.02% |
| 214 | assets_off_subs_polyjet | Quantity of churned/off-subscription assets for the Polyjet rollup at the cutoff. | 6.57 | 0.02% |
| 215 | assets_off_subs_fdm | Quantity of churned/off-subscription assets for the FDM rollup at the cutoff. | 6.42 | 0.02% |
| 224 | assets_rollup_polyjet | Active asset quantity for the Polyjet rollup at the cutoff. | 4.50 | 0.01% |
| 225 | assets_off_subs_sw_inspection | Quantity of churned/off-subscription assets for the SW Inspection rollup at the cutoff. | 4.36 | 0.01% |
| 227 | assets_rollup_catia | Active asset quantity for the CATIA rollup at the cutoff. | 4.17 | 0.01% |
| 232 | assets_rollup_hv_simulation | Active asset quantity for the HV Simulation rollup at the cutoff. | 3.00 | 0.01% |
| 237 | assets_rollup_am_software | Active asset quantity for the AM Software rollup at the cutoff. | 2.46 | 0.01% |
| 238 | assets_off_subs_service | Quantity of churned/off-subscription assets for the Service rollup at the cutoff. | 2.37 | 0.01% |
| 244 | assets_off_subs_post_processing | Quantity of churned/off-subscription assets for the Post Processing rollup at the cutoff. | 1.32 | 0.00% |
| 249 | assets_rollup_training | Active asset quantity for the Training rollup at the cutoff. | 0.43 | 0.00% |
| 353 | assets_rollup_metals | Active asset quantity for the Metals rollup at the cutoff. | 0.00 | 0.00% |
| 354 | assets_rollup_geomagic | Active asset quantity for the Geomagic rollup at the cutoff. | 0.00 | 0.00% |
| 513 | assets_rollup_artec | Active asset quantity for the Artec rollup at the cutoff. | 0.00 | 0.00% |
| 514 | assets_rollup_altium_pcbworks | Active asset quantity for the Altium Pcbworks rollup at the cutoff. | 0.00 | 0.00% |
| 355 | assets_rollup_formlabs | Active asset quantity for the Formlabs rollup at the cutoff. | 0.00 | 0.00% |
| 511 | assets_rollup_delmia | Active asset quantity for the Delmia rollup at the cutoff. | 0.00 | 0.00% |
| 417 | assets_on_subs_catia | Quantity of assets on subscription for the CATIA rollup at the cutoff. | 0.00 | 0.00% |
| 418 | assets_on_subs_camworks_seats | Quantity of assets on subscription for the Camworks Seats rollup at the cutoff. | 0.00 | 0.00% |
| 419 | assets_on_subs_artec | Quantity of assets on subscription for the Artec rollup at the cutoff. | 0.00 | 0.00% |
| 404 | assets_on_subs_altium_pcbworks | Quantity of assets on subscription for the Altium Pcbworks rollup at the cutoff. | 0.00 | 0.00% |
| 405 | assets_on_subs_am_support | Quantity of assets on subscription for the AM Support rollup at the cutoff. | 0.00 | 0.00% |
| 406 | assets_on_subs_am_software | Quantity of assets on subscription for the AM Software rollup at the cutoff. | 0.00 | 0.00% |
| 407 | assets_on_subs_3dx_revenue | Quantity of assets on subscription for the 3DX Revenue rollup at the cutoff. | 0.00 | 0.00% |
| 556 | assets_rollup_yxc_renewal | Active asset quantity for the YXC Renewal rollup at the cutoff. | 0.00 | 0.00% |
| 508 | assets_rollup_saf | Active asset quantity for the SAF rollup at the cutoff. | 0.00 | 0.00% |
| 400 | assets_rollup_pro_prem_new_uap | Active asset quantity for the PRO Prem NEW UAP rollup at the cutoff. | 0.00 | 0.00% |
| 401 | assets_rollup_post_processing | Active asset quantity for the Post Processing rollup at the cutoff. | 0.00 | 0.00% |
| 389 | assets_rollup_p3 | Active asset quantity for the P3 rollup at the cutoff. | 0.00 | 0.00% |
| 352 | assets_rollup_other_misc | Active asset quantity for the Other Misc rollup at the cutoff. | 0.00 | 0.00% |
| 507 | assets_rollup_sla | Active asset quantity for the SLA rollup at the cutoff. | 0.00 | 0.00% |
| 506 | assets_rollup_sw_plastics | Active asset quantity for the SW Plastics rollup at the cutoff. | 0.00 | 0.00% |
| 544 | assets_rollup_swood | Active asset quantity for the Swood rollup at the cutoff. | 0.00 | 0.00% |
| 512 | assets_rollup_consumables | Active asset quantity for the Consumables rollup at the cutoff. | 0.00 | 0.00% |
| 515 | assets_rollup_am_support | Active asset quantity for the AM Support rollup at the cutoff. | 0.00 | 0.00% |
| 414 | assets_on_subs_delmia | Quantity of assets on subscription for the Delmia rollup at the cutoff. | 0.00 | 0.00% |
| 416 | assets_on_subs_consumables | Quantity of assets on subscription for the Consumables rollup at the cutoff. | 0.00 | 0.00% |
| 412 | assets_on_subs_epdm_cad_editor_seats | Quantity of assets on subscription for the Epdm CAD Editor Seats rollup at the cutoff. | 0.00 | 0.00% |
| 383 | assets_on_subs_fdm | Quantity of assets on subscription for the FDM rollup at the cutoff. | 0.00 | 0.00% |
| 382 | assets_on_subs_formlabs | Quantity of assets on subscription for the Formlabs rollup at the cutoff. | 0.00 | 0.00% |
| 428 | assets_on_subs_creaform | Quantity of assets on subscription for the Creaform rollup at the cutoff. | 0.00 | 0.00% |
| 380 | assets_on_subs_hv_simulation | Quantity of assets on subscription for the HV Simulation rollup at the cutoff. | 0.00 | 0.00% |
| 379 | assets_on_subs_metals | Quantity of assets on subscription for the Metals rollup at the cutoff. | 0.00 | 0.00% |
| 378 | assets_on_subs_misc_seats | Quantity of assets on subscription for the Misc Seats rollup at the cutoff. | 0.00 | 0.00% |
| 377 | assets_on_subs_none | Quantity of assets on subscription for the None rollup at the cutoff. | 0.00 | 0.00% |
| 528 | assets_on_subs_other_misc | Quantity of assets on subscription for the Other Misc rollup at the cutoff. | 0.00 | 0.00% |
| 391 | assets_on_subs_p3 | Quantity of assets on subscription for the P3 rollup at the cutoff. | 0.00 | 0.00% |
| 390 | assets_on_subs_polyjet | Quantity of assets on subscription for the Polyjet rollup at the cutoff. | 0.00 | 0.00% |
| 413 | assets_on_subs_draftsight | Quantity of assets on subscription for the Draftsight rollup at the cutoff. | 0.00 | 0.00% |
| 365 | assets_off_subs_am_software | Quantity of churned/off-subscription assets for the AM Software rollup at the cutoff. | 0.00 | 0.00% |
| 366 | assets_on_subs_yxc_renewal | Quantity of assets on subscription for the YXC Renewal rollup at the cutoff. | 0.00 | 0.00% |
| 367 | assets_on_subs_unidentified | Quantity of assets on subscription for the Unidentified rollup at the cutoff. | 0.00 | 0.00% |
| 368 | assets_on_subs_training | Quantity of assets on subscription for the Training rollup at the cutoff. | 0.00 | 0.00% |
| 370 | assets_on_subs_service | Quantity of assets on subscription for the Service rollup at the cutoff. | 0.00 | 0.00% |
| 369 | assets_on_subs_simulation | Quantity of assets on subscription for the Simulation rollup at the cutoff. | 0.00 | 0.00% |
| 396 | assets_on_subs_swx_pro_prem | Quantity of assets on subscription for the SWX PRO Prem rollup at the cutoff. | 0.00 | 0.00% |
| 397 | assets_on_subs_swx_core | Quantity of assets on subscription for the SWX Core rollup at the cutoff. | 0.00 | 0.00% |
| 385 | assets_on_subs_sw_electrical | Quantity of assets on subscription for the SW Electrical rollup at the cutoff. | 0.00 | 0.00% |
| 398 | assets_on_subs_swood | Quantity of assets on subscription for the Swood rollup at the cutoff. | 0.00 | 0.00% |
| 399 | assets_on_subs_sw_plastics | Quantity of assets on subscription for the SW Plastics rollup at the cutoff. | 0.00 | 0.00% |
| 384 | assets_on_subs_sw_inspection | Quantity of assets on subscription for the SW Inspection rollup at the cutoff. | 0.00 | 0.00% |
| 387 | assets_on_subs_saf | Quantity of assets on subscription for the SAF rollup at the cutoff. | 0.00 | 0.00% |
| 386 | assets_on_subs_sla | Quantity of assets on subscription for the SLA rollup at the cutoff. | 0.00 | 0.00% |
| 388 | assets_on_subs_pro_prem_new_uap | Quantity of assets on subscription for the PRO Prem NEW UAP rollup at the cutoff. | 0.00 | 0.00% |
| 402 | assets_on_subs_post_processing | Quantity of assets on subscription for the Post Processing rollup at the cutoff. | 0.00 | 0.00% |
| 381 | assets_on_subs_geomagic | Quantity of assets on subscription for the Geomagic rollup at the cutoff. | 0.00 | 0.00% |
| 373 | assets_off_subs_hv_simulation | Quantity of churned/off-subscription assets for the HV Simulation rollup at the cutoff. | 0.00 | 0.00% |
| 374 | assets_off_subs_geomagic | Quantity of churned/off-subscription assets for the Geomagic rollup at the cutoff. | 0.00 | 0.00% |
| 432 | assets_off_subs_am_support | Quantity of churned/off-subscription assets for the AM Support rollup at the cutoff. | 0.00 | 0.00% |
| 430 | assets_off_subs_artec | Quantity of churned/off-subscription assets for the Artec rollup at the cutoff. | 0.00 | 0.00% |
| 429 | assets_off_subs_catia | Quantity of churned/off-subscription assets for the CATIA rollup at the cutoff. | 0.00 | 0.00% |
| 415 | assets_off_subs_consumables | Quantity of churned/off-subscription assets for the Consumables rollup at the cutoff. | 0.00 | 0.00% |
| 375 | assets_off_subs_delmia | Quantity of churned/off-subscription assets for the Delmia rollup at the cutoff. | 0.00 | 0.00% |
| 431 | assets_off_subs_altium_pcbworks | Quantity of churned/off-subscription assets for the Altium Pcbworks rollup at the cutoff. | 0.00 | 0.00% |
| 439 | assets_off_subs_metals | Quantity of churned/off-subscription assets for the Metals rollup at the cutoff. | 0.00 | 0.00% |
| 434 | assets_off_subs_sla | Quantity of churned/off-subscription assets for the SLA rollup at the cutoff. | 0.00 | 0.00% |
| 438 | assets_off_subs_other_misc | Quantity of churned/off-subscription assets for the Other Misc rollup at the cutoff. | 0.00 | 0.00% |
| 437 | assets_off_subs_p3 | Quantity of churned/off-subscription assets for the P3 rollup at the cutoff. | 0.00 | 0.00% |
| 435 | assets_off_subs_saf | Quantity of churned/off-subscription assets for the SAF rollup at the cutoff. | 0.00 | 0.00% |
| 436 | assets_off_subs_pro_prem_new_uap | Quantity of churned/off-subscription assets for the PRO Prem NEW UAP rollup at the cutoff. | 0.00 | 0.00% |
| 433 | assets_off_subs_sw_plastics | Quantity of churned/off-subscription assets for the SW Plastics rollup at the cutoff. | 0.00 | 0.00% |

### Lifecycle & cadence (17 features)
Tenure, interpurchase interval, and monthly trend features that describe customer cadence and lifecycle stage.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 3 | last_gap_days | Days since the most recent transaction at the cutoff. | 2,351.82 | 6.29% |
| 5 | gp_monthly_slope_12m | Trend (slope) of monthly gross profit over the last 12 months. | 1,351.86 | 3.62% |
| 6 | tenure_days | Days between the customer's first transaction and the cutoff. | 1,287.66 | 3.45% |
| 8 | gp_monthly_std_12m | Volatility (standard deviation) of monthly gross profit in the last 12 months. | 1,019.76 | 2.73% |
| 15 | tx_monthly_slope_12m | Trend (slope) of monthly transaction counts over the last 12 months. | 526.94 | 1.41% |
| 20 | ipi_median_days | Median number of days between consecutive transactions. | 429.23 | 1.15% |
| 41 | ipi_mean_days | Average number of days between consecutive transactions. | 181.31 | 0.49% |
| 104 | lifecycle__all__active_months__24m | Count of active months with any transactions in the last 24 months. | 50.26 | 0.13% |
| 108 | tx_monthly_std_12m | Volatility (standard deviation) of monthly transaction counts in the last 12 months. | 48.50 | 0.13% |
| 226 | lifecycle__all__tenure_bucket__lt3m | Indicator that customer tenure at cutoff falls in the <3 months bucket. | 4.19 | 0.01% |
| 536 | lifecycle__all__tenure_days__life | Lifecycle metric capturing tenure days over the customer lifetime. | 0.00 | 0.00% |
| 535 | lifecycle__all__tenure_months__life | Lifecycle metric capturing tenure months over the customer lifetime. | 0.00 | 0.00% |
| 534 | lifecycle__all__tenure_bucket__3to6m | Indicator that customer tenure at cutoff falls in the 3–6 months bucket. | 0.00 | 0.00% |
| 533 | lifecycle__all__tenure_bucket__6to12m | Indicator that customer tenure at cutoff falls in the 6–12 months bucket. | 0.00 | 0.00% |
| 532 | lifecycle__all__tenure_bucket__1to2y | Indicator that customer tenure at cutoff falls in the 1–2 years bucket. | 0.00 | 0.00% |
| 518 | lifecycle__all__tenure_bucket__ge2y | Indicator that customer tenure at cutoff falls in the ≥2 years bucket. | 0.00 | 0.00% |
| 478 | lifecycle__all__gap_days__life | Lifecycle metric capturing gap days over the customer lifetime. | 0.00 | 0.00% |

### Branch & rep coverage (80 features)
Normalized shares of transactions handled by the top sales branches and account reps.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 24 | rep_share_cindy_tubbs | Share of raw sales-log transactions handled by rep Cindy Tubbs. | 290.44 | 0.78% |
| 31 | branch_share_ca_santa_ana | Share of raw sales-log transactions attributed to the CA Santa ANA branch. Values sum to ≤1 across tracked branches. | 217.82 | 0.58% |
| 47 | branch_share_indiana | Share of raw sales-log transactions attributed to the Indiana branch. Values sum to ≤1 across tracked branches. | 140.33 | 0.38% |
| 57 | rep_share_stephen_gordon | Share of raw sales-log transactions handled by rep Stephen Gordon. | 122.15 | 0.33% |
| 63 | rep_share_andrew_johnson | Share of raw sales-log transactions handled by rep Andrew Johnson. | 110.20 | 0.29% |
| 67 | branch_share_ohio | Share of raw sales-log transactions attributed to the Ohio branch. Values sum to ≤1 across tracked branches. | 103.86 | 0.28% |
| 70 | rep_share_christina_shoaf | Share of raw sales-log transactions handled by rep Christina Shoaf. | 98.28 | 0.26% |
| 72 | branch_share_ca_san_diego | Share of raw sales-log transactions attributed to the CA SAN Diego branch. Values sum to ≤1 across tracked branches. | 96.83 | 0.26% |
| 75 | branch_share_colorado | Share of raw sales-log transactions attributed to the Colorado branch. Values sum to ≤1 across tracked branches. | 91.76 | 0.25% |
| 78 | rep_share_duyen_lam | Share of raw sales-log transactions handled by rep Duyen LAM. | 84.74 | 0.23% |
| 79 | branch_share_ca_norcal | Share of raw sales-log transactions attributed to the CA Norcal branch. Values sum to ≤1 across tracked branches. | 82.13 | 0.22% |
| 80 | rep_share_nicholas_koelliker | Share of raw sales-log transactions handled by rep Nicholas Koelliker. | 81.66 | 0.22% |
| 81 | rep_share_john_hanson | Share of raw sales-log transactions handled by rep John Hanson. | 80.46 | 0.22% |
| 88 | rep_share_duke_metu | Share of raw sales-log transactions handled by rep Duke Metu. | 67.94 | 0.18% |
| 98 | rep_share_cynthia_judy | Share of raw sales-log transactions handled by rep Cynthia Judy. | 57.71 | 0.15% |
| 103 | rep_share_carol_ban | Share of raw sales-log transactions handled by rep Carol BAN. | 50.56 | 0.14% |
| 109 | branch_share_utah | Share of raw sales-log transactions attributed to the Utah branch. Values sum to ≤1 across tracked branches. | 47.85 | 0.13% |
| 111 | rep_share_brandon_smith | Share of raw sales-log transactions handled by rep Brandon Smith. | 45.47 | 0.12% |
| 112 | rep_share_michael_dietzen | Share of raw sales-log transactions handled by rep Michael Dietzen. | 45.26 | 0.12% |
| 117 | branch_share_iowa | Share of raw sales-log transactions attributed to the Iowa branch. Values sum to ≤1 across tracked branches. | 40.82 | 0.11% |
| 118 | rep_share_julie_tautges | Share of raw sales-log transactions handled by rep Julie Tautges. | 40.52 | 0.11% |
| 121 | branch_share_minnesota | Share of raw sales-log transactions attributed to the Minnesota branch. Values sum to ≤1 across tracked branches. | 38.27 | 0.10% |
| 122 | branch_share_illinois | Share of raw sales-log transactions attributed to the Illinois branch. Values sum to ≤1 across tracked branches. | 36.35 | 0.10% |
| 126 | rep_share_jesus_moraga | Share of raw sales-log transactions handled by rep Jesus Moraga. | 33.26 | 0.09% |
| 127 | rep_share_ryan_ladle | Share of raw sales-log transactions handled by rep Ryan Ladle. | 32.52 | 0.09% |
| 133 | rep_share_rosie_ortega | Share of raw sales-log transactions handled by rep Rosie Ortega. | 29.53 | 0.08% |
| 135 | rep_share_bill_boudewyns | Share of raw sales-log transactions handled by rep Bill Boudewyns. | 28.75 | 0.08% |
| 136 | rep_share_alex_rathe | Share of raw sales-log transactions handled by rep Alex Rathe. | 28.32 | 0.08% |
| 137 | rep_share_ross_lee | Share of raw sales-log transactions handled by rep Ross LEE. | 27.96 | 0.07% |
| 139 | branch_share_ca_los_angeles | Share of raw sales-log transactions attributed to the CA LOS Angeles branch. Values sum to ≤1 across tracked branches. | 27.60 | 0.07% |
| 146 | branch_share_texas | Share of raw sales-log transactions attributed to the Texas branch. Values sum to ≤1 across tracked branches. | 25.29 | 0.07% |
| 147 | branch_share_pennsylvania | Share of raw sales-log transactions attributed to the Pennsylvania branch. Values sum to ≤1 across tracked branches. | 25.25 | 0.07% |
| 148 | branch_share_new_jersey | Share of raw sales-log transactions attributed to the NEW Jersey branch. Values sum to ≤1 across tracked branches. | 24.86 | 0.07% |
| 153 | rep_share_coulson_hess | Share of raw sales-log transactions handled by rep Coulson Hess. | 22.68 | 0.06% |
| 155 | rep_share_krinski_golden | Share of raw sales-log transactions handled by rep Krinski Golden. | 21.86 | 0.06% |
| 158 | branch_share_washington | Share of raw sales-log transactions attributed to the Washington branch. Values sum to ≤1 across tracked branches. | 21.63 | 0.06% |
| 162 | branch_share_michigan | Share of raw sales-log transactions attributed to the Michigan branch. Values sum to ≤1 across tracked branches. | 20.29 | 0.05% |
| 164 | rep_share_sarah_corbin | Share of raw sales-log transactions handled by rep Sarah Corbin. | 20.04 | 0.05% |
| 166 | branch_share_massachusetts | Share of raw sales-log transactions attributed to the Massachusetts branch. Values sum to ≤1 across tracked branches. | 19.21 | 0.05% |
| 167 | rep_share_julie_zais | Share of raw sales-log transactions handled by rep Julie Zais. | 18.78 | 0.05% |
| 168 | rep_share_aaron_herbner | Share of raw sales-log transactions handled by rep Aaron Herbner. | 18.65 | 0.05% |
| 169 | branch_share_arizona | Share of raw sales-log transactions attributed to the Arizona branch. Values sum to ≤1 across tracked branches. | 18.58 | 0.05% |
| 172 | rep_share_david_hunt | Share of raw sales-log transactions handled by rep David Hunt. | 17.40 | 0.05% |
| 174 | rep_share_whitney_street | Share of raw sales-log transactions handled by rep Whitney Street. | 16.78 | 0.04% |
| 186 | rep_share_victor_pimentel | Share of raw sales-log transactions handled by rep Victor Pimentel. | 12.90 | 0.03% |
| 189 | rep_share_michael_johnson | Share of raw sales-log transactions handled by rep Michael Johnson. | 12.41 | 0.03% |
| 190 | branch_share_missouri | Share of raw sales-log transactions attributed to the Missouri branch. Values sum to ≤1 across tracked branches. | 12.19 | 0.03% |
| 192 | branch_share_idaho | Share of raw sales-log transactions attributed to the Idaho branch. Values sum to ≤1 across tracked branches. | 11.68 | 0.03% |
| 193 | branch_share_wisconsin | Share of raw sales-log transactions attributed to the Wisconsin branch. Values sum to ≤1 across tracked branches. | 11.43 | 0.03% |
| 194 | rep_share_carlin_merrill | Share of raw sales-log transactions handled by rep Carlin Merrill. | 11.42 | 0.03% |
| 196 | rep_share_matthew_everett | Share of raw sales-log transactions handled by rep Matthew Everett. | 10.78 | 0.03% |
| 197 | branch_share_oregon | Share of raw sales-log transactions attributed to the Oregon branch. Values sum to ≤1 across tracked branches. | 10.26 | 0.03% |
| 201 | rep_share_mycroft_roe | Share of raw sales-log transactions handled by rep Mycroft ROE. | 9.33 | 0.02% |
| 202 | branch_share_kansas | Share of raw sales-log transactions attributed to the Kansas branch. Values sum to ≤1 across tracked branches. | 9.30 | 0.02% |
| 205 | rep_share_robert_baack | Share of raw sales-log transactions handled by rep Robert Baack. | 8.06 | 0.02% |
| 208 | rep_share_rick_radzai | Share of raw sales-log transactions handled by rep Rick Radzai. | 7.61 | 0.02% |
| 210 | branch_share_oklahoma | Share of raw sales-log transactions attributed to the Oklahoma branch. Values sum to ≤1 across tracked branches. | 6.96 | 0.02% |
| 217 | rep_share_mandy_douglas | Share of raw sales-log transactions handled by rep Mandy Douglas. | 6.09 | 0.02% |
| 219 | rep_share_jarred_jackson | Share of raw sales-log transactions handled by rep Jarred Jackson. | 5.55 | 0.01% |
| 220 | rep_share_rob_lambrecht | Share of raw sales-log transactions handled by rep ROB Lambrecht. | 5.21 | 0.01% |
| 221 | branch_share_kentucky | Share of raw sales-log transactions attributed to the Kentucky branch. Values sum to ≤1 across tracked branches. | 5.03 | 0.01% |
| 222 | branch_share_florida | Share of raw sales-log transactions attributed to the Florida branch. Values sum to ≤1 across tracked branches. | 4.79 | 0.01% |
| 223 | rep_share_william_eyler | Share of raw sales-log transactions handled by rep William Eyler. | 4.51 | 0.01% |
| 228 | rep_share_am_quotes | Share of raw sales-log transactions handled by rep AM Quotes. | 4.01 | 0.01% |
| 229 | rep_share_jonathan_husar | Share of raw sales-log transactions handled by rep Jonathan Husar. | 3.85 | 0.01% |
| 233 | rep_share_kristi_fischer | Share of raw sales-log transactions handled by rep Kristi Fischer. | 2.89 | 0.01% |
| 240 | rep_share_sam_scholes | Share of raw sales-log transactions handled by rep SAM Scholes. | 1.75 | 0.00% |
| 241 | rep_share_bryan_dalton | Share of raw sales-log transactions handled by rep Bryan Dalton. | 1.55 | 0.00% |
| 243 | branch_share_georgia | Share of raw sales-log transactions attributed to the Georgia branch. Values sum to ≤1 across tracked branches. | 1.40 | 0.00% |
| 245 | rep_share_lukasz_jaszczur | Share of raw sales-log transactions handled by rep Lukasz Jaszczur. | 1.12 | 0.00% |
| 246 | rep_share_kirk_brown | Share of raw sales-log transactions handled by rep Kirk Brown. | 1.11 | 0.00% |
| 247 | rep_share_joel_berens | Share of raw sales-log transactions handled by rep Joel Berens. | 0.89 | 0.00% |
| 248 | rep_share_jason_wood | Share of raw sales-log transactions handled by rep Jason Wood. | 0.85 | 0.00% |
| 519 | branch_share_canada | Share of raw sales-log transactions attributed to the Canada branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 531 | branch_share_new_mexico | Share of raw sales-log transactions attributed to the NEW Mexico branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 517 | branch_share_new_york | Share of raw sales-log transactions attributed to the NEW York branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 509 | rep_share_christopher_rhyndress | Share of raw sales-log transactions handled by rep Christopher Rhyndress. | 0.00 | 0.00% |
| 510 | rep_share_austin_etter | Share of raw sales-log transactions handled by rep Austin Etter. | 0.00 | 0.00% |
| 555 | rep_share_nancy_evans | Share of raw sales-log transactions handled by rep Nancy Evans. | 0.00 | 0.00% |
| 554 | rep_share_suke_lee | Share of raw sales-log transactions handled by rep Suke LEE. | 0.00 | 0.00% |

### Industry interactions (48 features)
Feature crosses between industry indicators and engagement scalars (services GP, avg GP, diversity, growth).

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 28 | is_aerospace_and_defense_x_avg_gp | Interaction between the Aerospace AND Defense industry flag and normalized average gross profit per transaction. | 247.34 | 0.66% |
| 35 | is_services_x_avg_gp | Interaction between the Services industry flag and normalized average gross profit per transaction. | 194.25 | 0.52% |
| 38 | is_aerospace_and_defense_x_growth | Interaction between the Aerospace AND Defense industry flag and gross-profit growth ratio (2024 vs. 2023). | 187.75 | 0.50% |
| 39 | is_services_x_growth | Interaction between the Services industry flag and gross-profit growth ratio (2024 vs. 2023). | 187.11 | 0.50% |
| 44 | is_industrial_machinery_x_avg_gp | Interaction between the Industrial Machinery industry flag and normalized average gross profit per transaction. | 157.98 | 0.42% |
| 45 | is_consumer_goods_x_avg_gp | Interaction between the Consumer Goods industry flag and normalized average gross profit per transaction. | 149.85 | 0.40% |
| 46 | is_manufactured_products_x_growth | Interaction between the Manufactured Products industry flag and gross-profit growth ratio (2024 vs. 2023). | 145.09 | 0.39% |
| 51 | is_heavy_equip_and_ind_components_x_avg_gp | Interaction between the Heavy Equip AND IND Components industry flag and normalized average gross profit per transaction. | 136.75 | 0.37% |
| 53 | is_medical_devices_and_life_sciences_x_avg_gp | Interaction between the Medical Devices AND Life Sciences industry flag and normalized average gross profit per transaction. | 130.94 | 0.35% |
| 55 | is_industrial_machinery_x_growth | Interaction between the Industrial Machinery industry flag and gross-profit growth ratio (2024 vs. 2023). | 124.33 | 0.33% |
| 56 | is_high_tech_x_avg_gp | Interaction between the High Tech industry flag and normalized average gross profit per transaction. | 123.95 | 0.33% |
| 68 | is_building_and_construction_x_avg_gp | Interaction between the Building AND Construction industry flag and normalized average gross profit per transaction. | 101.89 | 0.27% |
| 82 | is_heavy_equip_and_ind_components_x_growth | Interaction between the Heavy Equip AND IND Components industry flag and gross-profit growth ratio (2024 vs. 2023). | 80.29 | 0.21% |
| 83 | is_high_tech_x_growth | Interaction between the High Tech industry flag and gross-profit growth ratio (2024 vs. 2023). | 79.58 | 0.21% |
| 87 | is_education_and_research_x_avg_gp | Interaction between the Education AND Research industry flag and normalized average gross profit per transaction. | 77.29 | 0.21% |
| 90 | is_automotive_and_transportation_x_avg_gp | Interaction between the Automotive AND Transportation industry flag and normalized average gross profit per transaction. | 65.56 | 0.18% |
| 93 | is_consumer_goods_x_growth | Interaction between the Consumer Goods industry flag and gross-profit growth ratio (2024 vs. 2023). | 59.62 | 0.16% |
| 94 | is_medical_devices_and_life_sciences_x_growth | Interaction between the Medical Devices AND Life Sciences industry flag and gross-profit growth ratio (2024 vs. 2023). | 59.27 | 0.16% |
| 95 | is_automotive_and_transportation_x_growth | Interaction between the Automotive AND Transportation industry flag and gross-profit growth ratio (2024 vs. 2023). | 59.22 | 0.16% |
| 105 | is_mold_tool_and_die_x_growth | Interaction between the Mold Tool AND DIE industry flag and gross-profit growth ratio (2024 vs. 2023). | 50.08 | 0.13% |
| 106 | is_manufactured_products_x_avg_gp | Interaction between the Manufactured Products industry flag and normalized average gross profit per transaction. | 49.95 | 0.13% |
| 134 | is_manufactured_products_x_diversity | Interaction between the Manufactured Products industry flag and normalized product diversity score. | 29.02 | 0.08% |
| 144 | is_heavy_equip_and_ind_components_x_diversity | Interaction between the Heavy Equip AND IND Components industry flag and normalized product diversity score. | 25.79 | 0.07% |
| 156 | is_aerospace_and_defense_x_services | Interaction between the Aerospace AND Defense industry flag and normalized Services gross profit engagement. | 21.81 | 0.06% |
| 170 | is_mold_tool_and_die_x_avg_gp | Interaction between the Mold Tool AND DIE industry flag and normalized average gross profit per transaction. | 18.51 | 0.05% |
| 173 | is_building_and_construction_x_growth | Interaction between the Building AND Construction industry flag and gross-profit growth ratio (2024 vs. 2023). | 17.24 | 0.05% |
| 175 | is_automotive_and_transportation_x_services | Interaction between the Automotive AND Transportation industry flag and normalized Services gross profit engagement. | 16.63 | 0.04% |
| 181 | is_heavy_equip_and_ind_components_x_services | Interaction between the Heavy Equip AND IND Components industry flag and normalized Services gross profit engagement. | 14.50 | 0.04% |
| 184 | is_building_and_construction_x_services | Interaction between the Building AND Construction industry flag and normalized Services gross profit engagement. | 13.26 | 0.04% |
| 185 | is_industrial_machinery_x_services | Interaction between the Industrial Machinery industry flag and normalized Services gross profit engagement. | 12.92 | 0.03% |
| 187 | is_medical_devices_and_life_sciences_x_diversity | Interaction between the Medical Devices AND Life Sciences industry flag and normalized product diversity score. | 12.88 | 0.03% |
| 195 | is_consumer_goods_x_services | Interaction between the Consumer Goods industry flag and normalized Services gross profit engagement. | 10.87 | 0.03% |
| 211 | is_services_x_diversity | Interaction between the Services industry flag and normalized product diversity score. | 6.95 | 0.02% |
| 230 | is_aerospace_and_defense_x_diversity | Interaction between the Aerospace AND Defense industry flag and normalized product diversity score. | 3.19 | 0.01% |
| 231 | is_education_and_research_x_diversity | Interaction between the Education AND Research industry flag and normalized product diversity score. | 3.09 | 0.01% |
| 234 | is_high_tech_x_services | Interaction between the High Tech industry flag and normalized Services gross profit engagement. | 2.71 | 0.01% |
| 235 | is_building_and_construction_x_diversity | Interaction between the Building AND Construction industry flag and normalized product diversity score. | 2.53 | 0.01% |
| 516 | is_services_x_services | Interaction between the Services industry flag and normalized Services gross profit engagement. | 0.00 | 0.00% |
| 524 | is_education_and_research_x_services | Interaction between the Education AND Research industry flag and normalized Services gross profit engagement. | 0.00 | 0.00% |
| 527 | is_medical_devices_and_life_sciences_x_services | Interaction between the Medical Devices AND Life Sciences industry flag and normalized Services gross profit engagement. | 0.00 | 0.00% |
| 525 | is_mold_tool_and_die_x_services | Interaction between the Mold Tool AND DIE industry flag and normalized Services gross profit engagement. | 0.00 | 0.00% |
| 526 | is_manufactured_products_x_services | Interaction between the Manufactured Products industry flag and normalized Services gross profit engagement. | 0.00 | 0.00% |
| 359 | is_industrial_machinery_x_diversity | Interaction between the Industrial Machinery industry flag and normalized product diversity score. | 0.00 | 0.00% |
| 358 | is_high_tech_x_diversity | Interaction between the High Tech industry flag and normalized product diversity score. | 0.00 | 0.00% |
| 357 | is_automotive_and_transportation_x_diversity | Interaction between the Automotive AND Transportation industry flag and normalized product diversity score. | 0.00 | 0.00% |
| 371 | is_mold_tool_and_die_x_diversity | Interaction between the Mold Tool AND DIE industry flag and normalized product diversity score. | 0.00 | 0.00% |
| 372 | is_consumer_goods_x_diversity | Interaction between the Consumer Goods industry flag and normalized product diversity score. | 0.00 | 0.00% |
| 520 | is_education_and_research_x_growth | Interaction between the Education AND Research industry flag and gross-profit growth ratio (2024 vs. 2023). | 0.00 | 0.00% |

### Core transaction history (5 features)
Lifetime-level aggregates built from the full transaction history, including totals and annual views.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 4 | avg_transaction_gp | Average gross profit per transaction over the customer's full history. | 1,885.59 | 5.05% |
| 13 | total_gp_all_time | Cumulative gross profit across the customer's entire history. | 681.48 | 1.82% |
| 92 | total_transactions_all_time | Lifetime count of transactions across all product divisions. | 61.77 | 0.17% |
| 521 | transactions_last_2y | Number of transactions recorded in the last two calendar years. | 0.00 | 0.00% |
| 522 | total_gp_last_2y | Gross profit summed over the most recent two calendar years. | 0.00 | 0.00% |

### Industry segmentation (52 features)
Industry and sub-industry dummies derived from dim_customer enrichment and used for segmentation.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 18 | is_services | Industry indicator for the Services category (1 if customer matches). | 474.46 | 1.27% |
| 48 | is_mold_tool_and_die | Industry indicator for the Mold Tool AND DIE category (1 if customer matches). | 140.31 | 0.38% |
| 49 | is_medical_devices_and_life_sciences | Industry indicator for the Medical Devices AND Life Sciences category (1 if customer matches). | 139.14 | 0.37% |
| 54 | is_sub_08_5_packaged_goods_canned_frozen_foods_bakery_liquor_and_cosmetics | Indicator for customers in the 08 5 Packaged Goods Canned Frozen Foods Bakery Liquor AND Cosmetics sub-industry (derived from dim_customer). | 130.50 | 0.35% |
| 64 | is_sub_02_2_aircraft_parts_and_accessories | Indicator for customers in the 02 2 Aircraft Parts AND Accessories sub-industry (derived from dim_customer). | 109.51 | 0.29% |
| 69 | is_consumer_goods | Industry indicator for the Consumer Goods category (1 if customer matches). | 99.62 | 0.27% |
| 71 | is_building_and_construction | Industry indicator for the Building AND Construction category (1 if customer matches). | 98.04 | 0.26% |
| 76 | is_sub_13_1_engineering_services | Indicator for customers in the 13 1 Engineering Services sub-industry (derived from dim_customer). | 86.56 | 0.23% |
| 85 | is_sub_12_5_education | Indicator for customers in the 12 5 Education sub-industry (derived from dim_customer). | 78.39 | 0.21% |
| 96 | is_education_and_research | Industry indicator for the Education AND Research category (1 if customer matches). | 58.22 | 0.16% |
| 101 | is_sub_11_5_subcontractor_flooring_roofing_walls_site_prep | Indicator for customers in the 11 5 Subcontractor Flooring Roofing Walls Site Prep sub-industry (derived from dim_customer). | 52.95 | 0.14% |
| 102 | is_sub_01_4_automotive_and_transportation_services | Indicator for customers in the 01 4 Automotive AND Transportation Services sub-industry (derived from dim_customer). | 52.05 | 0.14% |
| 107 | is_sub_04_4_metalworking_machinery | Indicator for customers in the 04 4 Metalworking Machinery sub-industry (derived from dim_customer). | 48.51 | 0.13% |
| 110 | is_sub_02_3_space_systems_missiles_arms_and_other_defense | Indicator for customers in the 02 3 Space Systems Missiles Arms AND Other Defense sub-industry (derived from dim_customer). | 47.26 | 0.13% |
| 114 | is_plant_and_process | Industry indicator for the Plant AND Process category (1 if customer matches). | 44.34 | 0.12% |
| 116 | is_high_tech | Industry indicator for the High Tech category (1 if customer matches). | 41.79 | 0.11% |
| 123 | is_sub_06_1_heavy_equipment_elevators_conveyors_cranes_hoists_and_farm | Indicator for customers in the 06 1 Heavy Equipment Elevators Conveyors Cranes Hoists AND Farm sub-industry (derived from dim_customer). | 35.68 | 0.10% |
| 124 | is_sub_05_1_tools_and_dies | Indicator for customers in the 05 1 Tools AND Dies sub-industry (derived from dim_customer). | 35.20 | 0.09% |
| 129 | is_sub_09_2_medical_devices_and_equipment_incl_lab_apparatus_and_surgical_devices | Indicator for customers in the 09 2 Medical Devices AND Equipment Incl LAB Apparatus AND Surgical Devices sub-industry (derived from dim_customer). | 31.31 | 0.08% |
| 131 | is_manufactured_products | Industry indicator for the Manufactured Products category (1 if customer matches). | 30.15 | 0.08% |
| 132 | is_sub_11_4_subcontractor_structural_steel_and_wood | Indicator for customers in the 11 4 Subcontractor Structural Steel AND Wood sub-industry (derived from dim_customer). | 30.07 | 0.08% |
| 138 | is_sub_07_3_scientific_and_process_control_instruments | Indicator for customers in the 07 3 Scientific AND Process Control Instruments sub-industry (derived from dim_customer). | 27.73 | 0.07% |
| 140 | is_sub_04_1_packaging_machinery | Indicator for customers in the 04 1 Packaging Machinery sub-industry (derived from dim_customer). | 27.30 | 0.07% |
| 141 | is_sub_12_6_other_services | Indicator for customers in the 12 6 Other Services sub-industry (derived from dim_customer). | 27.27 | 0.07% |
| 142 | is_sub_11_2_general_contractors_and_builders | Indicator for customers in the 11 2 General Contractors AND Builders sub-industry (derived from dim_customer). | 26.70 | 0.07% |
| 145 | is_sub_05_3_plastics_molding | Indicator for customers in the 05 3 Plastics Molding sub-industry (derived from dim_customer). | 25.55 | 0.07% |
| 149 | is_sub_02_1_aircraft_manufacture_or_assembly | Indicator for customers in the 02 1 Aircraft Manufacture OR Assembly sub-industry (derived from dim_customer). | 24.73 | 0.07% |
| 150 | is_sub_07_1_pc_peripherals_and_software | Indicator for customers in the 07 1 PC Peripherals AND Software sub-industry (derived from dim_customer). | 24.31 | 0.07% |
| 151 | is_sub_07_6_semiconductors_and_related_devices_including_pcb | Indicator for customers in the 07 6 Semiconductors AND Related Devices Including PCB sub-industry (derived from dim_customer). | 23.36 | 0.06% |
| 152 | is_sub_04_5_other_industrial_machinery | Indicator for customers in the 04 5 Other Industrial Machinery sub-industry (derived from dim_customer). | 23.20 | 0.06% |
| 159 | is_aerospace_and_defense | Industry indicator for the Aerospace AND Defense category (1 if customer matches). | 20.77 | 0.06% |
| 160 | is_automotive_and_transportation | Industry indicator for the Automotive AND Transportation category (1 if customer matches). | 20.58 | 0.06% |
| 165 | is_energy | Industry indicator for the Energy category (1 if customer matches). | 19.23 | 0.05% |
| 178 | is_sub_10_6_oil_and_gas_petroleum | Indicator for customers in the 10 6 OIL AND GAS Petroleum sub-industry (derived from dim_customer). | 15.15 | 0.04% |
| 182 | is_sub_05_4_fabricated_metal_products | Indicator for customers in the 05 4 Fabricated Metal Products sub-industry (derived from dim_customer). | 14.23 | 0.04% |
| 183 | is_sub_06_2_valves_pipes_fittings_pulleys_bearings | Indicator for customers in the 06 2 Valves Pipes Fittings Pulleys Bearings sub-industry (derived from dim_customer). | 14.20 | 0.04% |
| 188 | is_chemicals_and_related_products | Industry indicator for the Chemicals AND Related Products category (1 if customer matches). | 12.72 | 0.03% |
| 191 | is_sub_education_and_research | Indicator for customers in the Education AND Research sub-industry (derived from dim_customer). | 11.94 | 0.03% |
| 199 | is_heavy_equip_and_ind_components | Industry indicator for the Heavy Equip AND IND Components category (1 if customer matches). | 9.57 | 0.03% |
| 204 | is_sub_08_3_personal_goods_and_leisure_luggage_sports_toys_music_and_books | Indicator for customers in the 08 3 Personal Goods AND Leisure Luggage Sports Toys Music AND Books sub-industry (derived from dim_customer). | 8.26 | 0.02% |
| 206 | is_industrial_machinery | Industry indicator for the Industrial Machinery category (1 if customer matches). | 7.83 | 0.02% |
| 207 | is_sub_01_3_auto_parts_and_accessories | Indicator for customers in the 01 3 Auto Parts AND Accessories sub-industry (derived from dim_customer). | 7.78 | 0.02% |
| 216 | is_sub_07_5_telecommunication_and_navigation | Indicator for customers in the 07 5 Telecommunication AND Navigation sub-industry (derived from dim_customer). | 6.25 | 0.02% |
| 218 | is_sub_06_3_heating_and_refrigeration_equipment_furnaces_oven | Indicator for customers in the 06 3 Heating AND Refrigeration Equipment Furnaces Oven sub-industry (derived from dim_customer). | 5.86 | 0.02% |
| 236 | is_sub_07_7_electrical_components_capacitors_batteries_lighting | Indicator for customers in the 07 7 Electrical Components Capacitors Batteries Lighting sub-industry (derived from dim_customer). | 2.50 | 0.01% |
| 242 | is_dental | Industry indicator for the Dental category (1 if customer matches). | 1.53 | 0.00% |
| 395 | industry_sub | Raw sub-industry label from dim_customer used for enrichment. | 0.00 | 0.00% |
| 408 | industry | Raw industry label from dim_customer used for enrichment. | 0.00 | 0.00% |
| 393 | is_health_care | Industry indicator for the Health Care category (1 if customer matches). | 0.00 | 0.00% |
| 394 | is_packaging | Industry indicator for the Packaging category (1 if customer matches). | 0.00 | 0.00% |
| 392 | is_electromagnetic | Industry indicator for the Electromagnetic category (1 if customer matches). | 0.00 | 0.00% |
| 356 | is_materials | Industry indicator for the Materials category (1 if customer matches). | 0.00 | 0.00% |

### Division mix & share (12 features)
Twelve-month gross-profit share features across major product divisions.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 21 | success plan_gp_share_12m | Share of 12-month gross profit attributed to Success Plan offerings. | 348.20 | 0.93% |
| 27 | solidworks_gp_share_12m | Share of 12-month gross profit attributed to Solidworks offerings. | 250.59 | 0.67% |
| 29 | training_gp_share_12m | Share of 12-month gross profit attributed to Training offerings. | 234.00 | 0.63% |
| 52 | maintenance_gp_share_12m | Share of 12-month gross profit attributed to Maintenance offerings. | 133.94 | 0.36% |
| 60 | hardware_gp_share_12m | Share of 12-month gross profit attributed to Hardware offerings. | 115.95 | 0.31% |
| 66 | services_gp_share_12m | Share of 12-month gross profit attributed to Services offerings. | 105.09 | 0.28% |
| 86 | simulation_gp_share_12m | Share of 12-month gross profit attributed to Simulation offerings. | 78.34 | 0.21% |
| 128 | pdm_gp_share_12m | Share of 12-month gross profit attributed to PDM offerings. | 32.22 | 0.09% |
| 161 | cpe_gp_share_12m | Share of 12-month gross profit attributed to CPE offerings. | 20.32 | 0.05% |
| 163 | scanning_gp_share_12m | Share of 12-month gross profit attributed to Scanning offerings. | 20.12 | 0.05% |
| 171 | camworks_gp_share_12m | Share of 12-month gross profit attributed to Camworks offerings. | 17.56 | 0.05% |
| 360 | post_processing_gp_share_12m | Share of 12-month gross profit attributed to Post Processing offerings. | 0.00 | 0.00% |

### Recent window stats (6 features)
Raw trailing-window counts and gross-profit statistics (including margin proxies) computed over multiple horizons.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 14 | gp_mean_last_24m | Average gross profit per transaction in the trailing 24-month window. | 616.21 | 1.65% |
| 16 | gp_sum_last_24m | Gross profit summed over the trailing 24-month window ending at the cutoff. | 510.93 | 1.37% |
| 113 | margin__all__gp_pct__24m | Signed gross-profit margin proxy for all divisions over the trailing 24 months (GP / |GP|). | 44.57 | 0.12% |
| 143 | tx_count_last_24m | Transaction count in the trailing 24-month window ending at the cutoff. | 26.03 | 0.07% |
| 179 | margin__div__gp_pct__24m | Signed gross-profit margin proxy for the target division over the trailing 24 months (GP / |GP|). | 15.10 | 0.04% |
| 523 | avg_gp_per_tx_last_24m | Average gross profit per transaction in the trailing 24-month window (zero when no transactions). | 0.00 | 0.00% |

### Binary flags (3 features)
One-hot indicators sourced from sales logs (e.g., Account Coverage Review, new-customer flags).

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 11 | ever_bought_solidworks | Indicator that the customer has ever purchased Solidworks. | 814.56 | 2.18% |
| 477 | ever_acr | Flag indicating the account was ever marked as an Account Coverage Review (ACR) in sales logs. | 0.00 | 0.00% |
| 476 | ever_new_customer | Flag indicating the account was ever marked as a new customer in sales logs. | 0.00 | 0.00% |

### Adjacent division engagement (5 features)
Signals that capture activity in adjacent GoEngineer divisions (Services, Simulation, Hardware, Training).

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 19 | hardware_transaction_count | Number of transactions tied to the Hardware division. | 436.37 | 1.17% |
| 36 | total_training_gp | Gross profit generated from Training SKU transactions. | 190.30 | 0.51% |
| 50 | total_services_gp | Gross profit generated from Services transactions across history. | 137.87 | 0.37% |
| 120 | services_transaction_count | Number of transactions tied to the Services division. | 38.87 | 0.10% |
| 209 | simulation_transaction_count | Number of transactions tied to the Simulation division. | 7.48 | 0.02% |

### RFM derived metrics (20 features)
RFM-style engineered metrics with division/all scope windows, tail-masked variants, and 12-month deltas/ratios.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 43 | rfm__div__gp_sum__24m | Gross profit total for the target division over the trailing 24-month window ending at the cutoff. | 158.37 | 0.42% |
| 73 | rfm__div__gp_mean__24m | Average gross profit per transaction for the target division over the trailing 24-month window ending at the cutoff. | 93.65 | 0.25% |
| 100 | rfm__div__tx_n__24m | Transaction count for the target division over the trailing 24-month window ending at the cutoff. | 56.68 | 0.15% |
| 125 | rfm__all__gp_sum__24m | Gross profit total for all divisions over the trailing 24-month window ending at the cutoff. | 33.65 | 0.09% |
| 198 | rfm__all__tx_n__ratio_12m_prev12m | Ratio of transaction count for all divisions: last 12 months divided by the preceding 12 months. | 10.05 | 0.03% |
| 200 | rfm__all__gp_sum__ratio_12m_prev12m | Ratio of gross profit for all divisions: last 12 months divided by the preceding 12 months. | 9.34 | 0.02% |
| 213 | rfm__all__gp_sum__delta_12m_prev12m | Change in gross profit for all divisions: last 12 months minus the preceding 12 months. | 6.65 | 0.02% |
| 239 | rfm__div__gp_sum__24m_off60d | Gross profit total for the target division over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 1.84 | 0.00% |
| 529 | rfm__all__gp_mean__24m_off60d | Average gross profit per transaction for all divisions over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 530 | rfm__all__gp_sum__24m_off60d | Gross profit total for all divisions over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 586 | rfm__all__tx_n__24m_off60d | Transaction count for all divisions over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 364 | rfm__div__tx_n__24m_off60d | Transaction count for the target division over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 474 | rfm__all__gp_mean__24m | Average gross profit per transaction for all divisions over the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 475 | rfm__all__tx_n__24m | Transaction count for all divisions over the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 376 | rfm__div__gp_mean__24m_off60d | Average gross profit per transaction for the target division over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 473 | rfm__all__tx_n__delta_12m_prev12m | Change in transaction count for all divisions: last 12 months minus the preceding 12 months. | 0.00 | 0.00% |
| 472 | rfm__div__gp_sum__delta_12m_prev12m | Change in gross profit for the target division: last 12 months minus the preceding 12 months. | 0.00 | 0.00% |
| 471 | rfm__div__gp_sum__ratio_12m_prev12m | Ratio of gross profit for the target division: last 12 months divided by the preceding 12 months. | 0.00 | 0.00% |
| 470 | rfm__div__tx_n__delta_12m_prev12m | Change in transaction count for the target division: last 12 months minus the preceding 12 months. | 0.00 | 0.00% |
| 537 | rfm__div__tx_n__ratio_12m_prev12m | Ratio of transaction count for the target division: last 12 months divided by the preceding 12 months. | 0.00 | 0.00% |

### Diversity & seasonality (10 features)
Features that measure SKU breadth, division breadth, and seasonal transaction mix across quarters.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 58 | q1_share_24m | Share of transactions in the last 24 months that occurred in Q1. | 121.50 | 0.33% |
| 59 | q2_share_24m | Share of transactions in the last 24 months that occurred in Q2. | 117.65 | 0.31% |
| 97 | product_diversity_score | Distinct product divisions purchased over the customer's history. | 57.82 | 0.15% |
| 99 | sku_diversity_score | Distinct SKUs purchased over the customer's history. | 57.11 | 0.15% |
| 362 | q3_share_24m | Share of transactions in the last 24 months that occurred in Q3. | 0.00 | 0.00% |
| 361 | q4_share_24m | Share of transactions in the last 24 months that occurred in Q4. | 0.00 | 0.00% |
| 545 | season__all__q1_share__24m | Share of transactions occurring in calendar quarter Q1 during the last 24 months. | 0.00 | 0.00% |
| 557 | season__all__q2_share__24m | Share of transactions occurring in calendar quarter Q2 during the last 24 months. | 0.00 | 0.00% |
| 543 | season__all__q3_share__24m | Share of transactions occurring in calendar quarter Q3 during the last 24 months. | 0.00 | 0.00% |
| 542 | season__all__q4_share__24m | Share of transactions occurring in calendar quarter Q4 during the last 24 months. | 0.00 | 0.00% |

### Market basket & affinity (2 features)
Lagged market-basket lift aggregates that proxy Solidworks affinity from prior SKU purchases.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 30 | mb_lift_mean_lag60d | Mean market-basket lift of SKUs purchased at least 60 days before cutoff. | 218.80 | 0.59% |
| 62 | mb_lift_max_lag60d | Maximum market-basket lift of SKUs purchased at least 60 days before cutoff. | 111.81 | 0.30% |

### Growth & momentum (1 features)
Growth signals that compare recent performance across years (e.g., 2024 vs. 2023 gross profit).

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 33 | growth_ratio_24_over_23 | Ratio of 2024 gross profit to 2023 gross profit as a momentum signal. | 203.64 | 0.54% |

### Missingness indicators (287 features)
Binary flags appended when add_missingness_flags is enabled, marking engineered features that were null.

| Rank | Feature | Description | Gain | Gain % |
| ---: | --- | --- | ---: | ---: |
| 541 | total_transactions_all_time_missing | Missing-value indicator for `total_transactions_all_time` (1 if the engineered value was null or unavailable). Base feature: Lifetime count of transactions across all product divisions. | 0.00 | 0.00% |
| 540 | transactions_last_2y_missing | Missing-value indicator for `transactions_last_2y` (1 if the engineered value was null or unavailable). Base feature: Number of transactions recorded in the last two calendar years. | 0.00 | 0.00% |
| 539 | total_gp_all_time_missing | Missing-value indicator for `total_gp_all_time` (1 if the engineered value was null or unavailable). Base feature: Cumulative gross profit across the customer's entire history. | 0.00 | 0.00% |
| 538 | total_gp_last_2y_missing | Missing-value indicator for `total_gp_last_2y` (1 if the engineered value was null or unavailable). Base feature: Gross profit summed over the most recent two calendar years. | 0.00 | 0.00% |
| 553 | avg_transaction_gp_missing | Missing-value indicator for `avg_transaction_gp` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction over the customer's full history. | 0.00 | 0.00% |
| 552 | services_transaction_count_missing | Missing-value indicator for `services_transaction_count` (1 if the engineered value was null or unavailable). Base feature: Number of transactions tied to the Services division. | 0.00 | 0.00% |
| 551 | simulation_transaction_count_missing | Missing-value indicator for `simulation_transaction_count` (1 if the engineered value was null or unavailable). Base feature: Number of transactions tied to the Simulation division. | 0.00 | 0.00% |
| 550 | hardware_transaction_count_missing | Missing-value indicator for `hardware_transaction_count` (1 if the engineered value was null or unavailable). Base feature: Number of transactions tied to the Hardware division. | 0.00 | 0.00% |
| 549 | total_services_gp_missing | Missing-value indicator for `total_services_gp` (1 if the engineered value was null or unavailable). Base feature: Gross profit generated from Services transactions across history. | 0.00 | 0.00% |
| 548 | total_training_gp_missing | Missing-value indicator for `total_training_gp` (1 if the engineered value was null or unavailable). Base feature: Gross profit generated from Training SKU transactions. | 0.00 | 0.00% |
| 547 | gp_2024_missing | Missing-value indicator for `gp_2024` (1 if the engineered value was null or unavailable). Base feature: Gross profit booked in calendar year 2024. | 0.00 | 0.00% |
| 546 | gp_2023_missing | Missing-value indicator for `gp_2023` (1 if the engineered value was null or unavailable). Base feature: Gross profit booked in calendar year 2023. | 0.00 | 0.00% |
| 496 | product_diversity_score_missing | Missing-value indicator for `product_diversity_score` (1 if the engineered value was null or unavailable). Base feature: Distinct product divisions purchased over the customer's history. | 0.00 | 0.00% |
| 495 | sku_diversity_score_missing | Missing-value indicator for `sku_diversity_score` (1 if the engineered value was null or unavailable). Base feature: Distinct SKUs purchased over the customer's history. | 0.00 | 0.00% |
| 494 | tx_count_last_3m_missing | Missing-value indicator for `tx_count_last_3m` (1 if the engineered value was null or unavailable). Base feature: Transaction count in the trailing 3-month window ending at the cutoff. | 0.00 | 0.00% |
| 493 | gp_sum_last_3m_missing | Missing-value indicator for `gp_sum_last_3m` (1 if the engineered value was null or unavailable). Base feature: Gross profit summed over the trailing 3-month window ending at the cutoff. | 0.00 | 0.00% |
| 505 | gp_mean_last_3m_missing | Missing-value indicator for `gp_mean_last_3m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction in the trailing 3-month window. | 0.00 | 0.00% |
| 491 | avg_gp_per_tx_last_3m_missing | Missing-value indicator for `avg_gp_per_tx_last_3m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction in the trailing 3-month window (zero when no transactions). | 0.00 | 0.00% |
| 490 | tx_count_last_6m_missing | Missing-value indicator for `tx_count_last_6m` (1 if the engineered value was null or unavailable). Base feature: Transaction count in the trailing 6-month window ending at the cutoff. | 0.00 | 0.00% |
| 489 | gp_sum_last_6m_missing | Missing-value indicator for `gp_sum_last_6m` (1 if the engineered value was null or unavailable). Base feature: Gross profit summed over the trailing 6-month window ending at the cutoff. | 0.00 | 0.00% |
| 504 | gp_mean_last_6m_missing | Missing-value indicator for `gp_mean_last_6m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction in the trailing 6-month window. | 0.00 | 0.00% |
| 503 | avg_gp_per_tx_last_6m_missing | Missing-value indicator for `avg_gp_per_tx_last_6m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction in the trailing 6-month window (zero when no transactions). | 0.00 | 0.00% |
| 502 | tx_count_last_12m_missing | Missing-value indicator for `tx_count_last_12m` (1 if the engineered value was null or unavailable). Base feature: Transaction count in the trailing 12-month window ending at the cutoff. | 0.00 | 0.00% |
| 501 | gp_sum_last_12m_missing | Missing-value indicator for `gp_sum_last_12m` (1 if the engineered value was null or unavailable). Base feature: Gross profit summed over the trailing 12-month window ending at the cutoff. | 0.00 | 0.00% |
| 500 | gp_mean_last_12m_missing | Missing-value indicator for `gp_mean_last_12m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction in the trailing 12-month window. | 0.00 | 0.00% |
| 499 | avg_gp_per_tx_last_12m_missing | Missing-value indicator for `avg_gp_per_tx_last_12m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction in the trailing 12-month window (zero when no transactions). | 0.00 | 0.00% |
| 498 | tx_count_last_24m_missing | Missing-value indicator for `tx_count_last_24m` (1 if the engineered value was null or unavailable). Base feature: Transaction count in the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 497 | gp_sum_last_24m_missing | Missing-value indicator for `gp_sum_last_24m` (1 if the engineered value was null or unavailable). Base feature: Gross profit summed over the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 461 | gp_mean_last_24m_missing | Missing-value indicator for `gp_mean_last_24m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction in the trailing 24-month window. | 0.00 | 0.00% |
| 460 | avg_gp_per_tx_last_24m_missing | Missing-value indicator for `avg_gp_per_tx_last_24m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction in the trailing 24-month window (zero when no transactions). | 0.00 | 0.00% |
| 459 | margin__all__gp_pct__24m_missing | Missing-value indicator for `margin__all__gp_pct__24m` (1 if the engineered value was null or unavailable). Base feature: Signed gross-profit margin proxy for all divisions over the trailing 24 months (GP / |GP|). | 0.00 | 0.00% |
| 458 | rfm__all__tx_n__24m_off60d_missing | Missing-value indicator for `rfm__all__tx_n__24m_off60d` (1 if the engineered value was null or unavailable). Base feature: Transaction count for all divisions over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 457 | rfm__all__gp_sum__24m_off60d_missing | Missing-value indicator for `rfm__all__gp_sum__24m_off60d` (1 if the engineered value was null or unavailable). Base feature: Gross profit total for all divisions over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 456 | rfm__all__gp_mean__24m_off60d_missing | Missing-value indicator for `rfm__all__gp_mean__24m_off60d` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction for all divisions over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 455 | rfm__div__tx_n__24m_missing | Missing-value indicator for `rfm__div__tx_n__24m` (1 if the engineered value was null or unavailable). Base feature: Transaction count for the target division over the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 492 | rfm__div__gp_sum__24m_missing | Missing-value indicator for `rfm__div__gp_sum__24m` (1 if the engineered value was null or unavailable). Base feature: Gross profit total for the target division over the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 469 | rfm__div__gp_mean__24m_missing | Missing-value indicator for `rfm__div__gp_mean__24m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction for the target division over the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 468 | margin__div__gp_pct__24m_missing | Missing-value indicator for `margin__div__gp_pct__24m` (1 if the engineered value was null or unavailable). Base feature: Signed gross-profit margin proxy for the target division over the trailing 24 months (GP / |GP|). | 0.00 | 0.00% |
| 467 | rfm__div__tx_n__24m_off60d_missing | Missing-value indicator for `rfm__div__tx_n__24m_off60d` (1 if the engineered value was null or unavailable). Base feature: Transaction count for the target division over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 479 | rfm__div__gp_sum__24m_off60d_missing | Missing-value indicator for `rfm__div__gp_sum__24m_off60d` (1 if the engineered value was null or unavailable). Base feature: Gross profit total for the target division over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 465 | rfm__div__gp_mean__24m_off60d_missing | Missing-value indicator for `rfm__div__gp_mean__24m_off60d` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction for the target division over the trailing 24-month window ending at the cutoff. Window stops 60 days before the cutoff to enforce a tail mask. | 0.00 | 0.00% |
| 464 | gp_monthly_slope_12m_missing | Missing-value indicator for `gp_monthly_slope_12m` (1 if the engineered value was null or unavailable). Base feature: Trend (slope) of monthly gross profit over the last 12 months. | 0.00 | 0.00% |
| 463 | gp_monthly_std_12m_missing | Missing-value indicator for `gp_monthly_std_12m` (1 if the engineered value was null or unavailable). Base feature: Volatility (standard deviation) of monthly gross profit in the last 12 months. | 0.00 | 0.00% |
| 462 | tx_monthly_slope_12m_missing | Missing-value indicator for `tx_monthly_slope_12m` (1 if the engineered value was null or unavailable). Base feature: Trend (slope) of monthly transaction counts over the last 12 months. | 0.00 | 0.00% |
| 620 | tx_monthly_std_12m_missing | Missing-value indicator for `tx_monthly_std_12m` (1 if the engineered value was null or unavailable). Base feature: Volatility (standard deviation) of monthly transaction counts in the last 12 months. | 0.00 | 0.00% |
| 619 | tenure_days_missing | Missing-value indicator for `tenure_days` (1 if the engineered value was null or unavailable). Base feature: Days between the customer's first transaction and the cutoff. | 0.00 | 0.00% |
| 618 | ipi_median_days_missing | Missing-value indicator for `ipi_median_days` (1 if the engineered value was null or unavailable). Base feature: Median number of days between consecutive transactions. | 0.00 | 0.00% |
| 617 | ipi_mean_days_missing | Missing-value indicator for `ipi_mean_days` (1 if the engineered value was null or unavailable). Base feature: Average number of days between consecutive transactions. | 0.00 | 0.00% |
| 616 | last_gap_days_missing | Missing-value indicator for `last_gap_days` (1 if the engineered value was null or unavailable). Base feature: Days since the most recent transaction at the cutoff. | 0.00 | 0.00% |
| 615 | lifecycle__all__active_months__24m_missing | Missing-value indicator for `lifecycle__all__active_months__24m` (1 if the engineered value was null or unavailable). Base feature: Count of active months with any transactions in the last 24 months. | 0.00 | 0.00% |
| 614 | q1_share_24m_missing | Missing-value indicator for `q1_share_24m` (1 if the engineered value was null or unavailable). Base feature: Share of transactions in the last 24 months that occurred in Q1. | 0.00 | 0.00% |
| 613 | q2_share_24m_missing | Missing-value indicator for `q2_share_24m` (1 if the engineered value was null or unavailable). Base feature: Share of transactions in the last 24 months that occurred in Q2. | 0.00 | 0.00% |
| 628 | q3_share_24m_missing | Missing-value indicator for `q3_share_24m` (1 if the engineered value was null or unavailable). Base feature: Share of transactions in the last 24 months that occurred in Q3. | 0.00 | 0.00% |
| 627 | q4_share_24m_missing | Missing-value indicator for `q4_share_24m` (1 if the engineered value was null or unavailable). Base feature: Share of transactions in the last 24 months that occurred in Q4. | 0.00 | 0.00% |
| 626 | camworks_gp_share_12m_missing | Missing-value indicator for `camworks_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Camworks offerings. | 0.00 | 0.00% |
| 625 | cpe_gp_share_12m_missing | Missing-value indicator for `cpe_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to CPE offerings. | 0.00 | 0.00% |
| 624 | hardware_gp_share_12m_missing | Missing-value indicator for `hardware_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Hardware offerings. | 0.00 | 0.00% |
| 623 | maintenance_gp_share_12m_missing | Missing-value indicator for `maintenance_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Maintenance offerings. | 0.00 | 0.00% |
| 635 | pdm_gp_share_12m_missing | Missing-value indicator for `pdm_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to PDM offerings. | 0.00 | 0.00% |
| 621 | post_processing_gp_share_12m_missing | Missing-value indicator for `post_processing_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Post Processing offerings. | 0.00 | 0.00% |
| 480 | scanning_gp_share_12m_missing | Missing-value indicator for `scanning_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Scanning offerings. | 0.00 | 0.00% |
| 570 | services_gp_share_12m_missing | Missing-value indicator for `services_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Services offerings. | 0.00 | 0.00% |
| 634 | simulation_gp_share_12m_missing | Missing-value indicator for `simulation_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Simulation offerings. | 0.00 | 0.00% |
| 633 | solidworks_gp_share_12m_missing | Missing-value indicator for `solidworks_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Solidworks offerings. | 0.00 | 0.00% |
| 632 | success plan_gp_share_12m_missing | Missing-value indicator for `success plan_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Success Plan offerings. | 0.00 | 0.00% |
| 631 | training_gp_share_12m_missing | Missing-value indicator for `training_gp_share_12m` (1 if the engineered value was null or unavailable). Base feature: Share of 12-month gross profit attributed to Training offerings. | 0.00 | 0.00% |
| 630 | ever_bought_solidworks_missing | Missing-value indicator for `ever_bought_solidworks` (1 if the engineered value was null or unavailable). Base feature: Indicator that the customer has ever purchased Solidworks. | 0.00 | 0.00% |
| 629 | branch_share_arizona_missing | Missing-value indicator for `branch_share_arizona` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Arizona branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 488 | branch_share_ca_los_angeles_missing | Missing-value indicator for `branch_share_ca_los_angeles` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the CA LOS Angeles branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 487 | branch_share_ca_norcal_missing | Missing-value indicator for `branch_share_ca_norcal` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the CA Norcal branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 486 | branch_share_ca_san_diego_missing | Missing-value indicator for `branch_share_ca_san_diego` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the CA SAN Diego branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 485 | branch_share_ca_santa_ana_missing | Missing-value indicator for `branch_share_ca_santa_ana` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the CA Santa ANA branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 484 | branch_share_canada_missing | Missing-value indicator for `branch_share_canada` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Canada branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 483 | branch_share_colorado_missing | Missing-value indicator for `branch_share_colorado` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Colorado branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 482 | branch_share_florida_missing | Missing-value indicator for `branch_share_florida` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Florida branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 481 | branch_share_georgia_missing | Missing-value indicator for `branch_share_georgia` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Georgia branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 640 | branch_share_idaho_missing | Missing-value indicator for `branch_share_idaho` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Idaho branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 639 | branch_share_illinois_missing | Missing-value indicator for `branch_share_illinois` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Illinois branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 638 | branch_share_indiana_missing | Missing-value indicator for `branch_share_indiana` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Indiana branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 637 | branch_share_iowa_missing | Missing-value indicator for `branch_share_iowa` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Iowa branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 636 | branch_share_kansas_missing | Missing-value indicator for `branch_share_kansas` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Kansas branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 622 | branch_share_kentucky_missing | Missing-value indicator for `branch_share_kentucky` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Kentucky branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 582 | branch_share_massachusetts_missing | Missing-value indicator for `branch_share_massachusetts` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Massachusetts branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 581 | branch_share_michigan_missing | Missing-value indicator for `branch_share_michigan` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Michigan branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 661 | branch_share_minnesota_missing | Missing-value indicator for `branch_share_minnesota` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Minnesota branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 647 | branch_share_missouri_missing | Missing-value indicator for `branch_share_missouri` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Missouri branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 646 | branch_share_new_jersey_missing | Missing-value indicator for `branch_share_new_jersey` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the NEW Jersey branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 645 | branch_share_new_mexico_missing | Missing-value indicator for `branch_share_new_mexico` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the NEW Mexico branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 644 | branch_share_new_york_missing | Missing-value indicator for `branch_share_new_york` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the NEW York branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 643 | branch_share_ohio_missing | Missing-value indicator for `branch_share_ohio` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Ohio branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 642 | branch_share_oklahoma_missing | Missing-value indicator for `branch_share_oklahoma` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Oklahoma branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 641 | branch_share_oregon_missing | Missing-value indicator for `branch_share_oregon` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Oregon branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 656 | branch_share_pennsylvania_missing | Missing-value indicator for `branch_share_pennsylvania` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Pennsylvania branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 655 | branch_share_texas_missing | Missing-value indicator for `branch_share_texas` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Texas branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 654 | branch_share_utah_missing | Missing-value indicator for `branch_share_utah` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Utah branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 653 | branch_share_washington_missing | Missing-value indicator for `branch_share_washington` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Washington branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 652 | branch_share_wisconsin_missing | Missing-value indicator for `branch_share_wisconsin` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions attributed to the Wisconsin branch. Values sum to ≤1 across tracked branches. | 0.00 | 0.00% |
| 651 | rep_share_am_quotes_missing | Missing-value indicator for `rep_share_am_quotes` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep AM Quotes. | 0.00 | 0.00% |
| 650 | rep_share_aaron_herbner_missing | Missing-value indicator for `rep_share_aaron_herbner` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Aaron Herbner. | 0.00 | 0.00% |
| 649 | rep_share_alex_rathe_missing | Missing-value indicator for `rep_share_alex_rathe` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Alex Rathe. | 0.00 | 0.00% |
| 612 | rep_share_andrew_johnson_missing | Missing-value indicator for `rep_share_andrew_johnson` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Andrew Johnson. | 0.00 | 0.00% |
| 611 | rep_share_austin_etter_missing | Missing-value indicator for `rep_share_austin_etter` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Austin Etter. | 0.00 | 0.00% |
| 610 | rep_share_bill_boudewyns_missing | Missing-value indicator for `rep_share_bill_boudewyns` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Bill Boudewyns. | 0.00 | 0.00% |
| 648 | rep_share_brandon_smith_missing | Missing-value indicator for `rep_share_brandon_smith` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Brandon Smith. | 0.00 | 0.00% |
| 660 | rep_share_bryan_dalton_missing | Missing-value indicator for `rep_share_bryan_dalton` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Bryan Dalton. | 0.00 | 0.00% |
| 659 | rep_share_carlin_merrill_missing | Missing-value indicator for `rep_share_carlin_merrill` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Carlin Merrill. | 0.00 | 0.00% |
| 658 | rep_share_carol_ban_missing | Missing-value indicator for `rep_share_carol_ban` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Carol BAN. | 0.00 | 0.00% |
| 657 | rep_share_christina_shoaf_missing | Missing-value indicator for `rep_share_christina_shoaf` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Christina Shoaf. | 0.00 | 0.00% |
| 608 | rep_share_christopher_rhyndress_missing | Missing-value indicator for `rep_share_christopher_rhyndress` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Christopher Rhyndress. | 0.00 | 0.00% |
| 607 | rep_share_cindy_tubbs_missing | Missing-value indicator for `rep_share_cindy_tubbs` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Cindy Tubbs. | 0.00 | 0.00% |
| 606 | rep_share_coulson_hess_missing | Missing-value indicator for `rep_share_coulson_hess` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Coulson Hess. | 0.00 | 0.00% |
| 605 | rep_share_cynthia_judy_missing | Missing-value indicator for `rep_share_cynthia_judy` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Cynthia Judy. | 0.00 | 0.00% |
| 604 | rep_share_david_hunt_missing | Missing-value indicator for `rep_share_david_hunt` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep David Hunt. | 0.00 | 0.00% |
| 603 | rep_share_duke_metu_missing | Missing-value indicator for `rep_share_duke_metu` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Duke Metu. | 0.00 | 0.00% |
| 602 | rep_share_duyen_lam_missing | Missing-value indicator for `rep_share_duyen_lam` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Duyen LAM. | 0.00 | 0.00% |
| 601 | rep_share_jarred_jackson_missing | Missing-value indicator for `rep_share_jarred_jackson` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Jarred Jackson. | 0.00 | 0.00% |
| 564 | rep_share_jason_wood_missing | Missing-value indicator for `rep_share_jason_wood` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Jason Wood. | 0.00 | 0.00% |
| 563 | rep_share_jesus_moraga_missing | Missing-value indicator for `rep_share_jesus_moraga` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Jesus Moraga. | 0.00 | 0.00% |
| 562 | rep_share_joel_berens_missing | Missing-value indicator for `rep_share_joel_berens` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Joel Berens. | 0.00 | 0.00% |
| 561 | rep_share_john_hanson_missing | Missing-value indicator for `rep_share_john_hanson` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep John Hanson. | 0.00 | 0.00% |
| 560 | rep_share_jonathan_husar_missing | Missing-value indicator for `rep_share_jonathan_husar` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Jonathan Husar. | 0.00 | 0.00% |
| 559 | rep_share_julie_tautges_missing | Missing-value indicator for `rep_share_julie_tautges` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Julie Tautges. | 0.00 | 0.00% |
| 558 | rep_share_julie_zais_missing | Missing-value indicator for `rep_share_julie_zais` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Julie Zais. | 0.00 | 0.00% |
| 596 | rep_share_kirk_brown_missing | Missing-value indicator for `rep_share_kirk_brown` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Kirk Brown. | 0.00 | 0.00% |
| 572 | rep_share_krinski_golden_missing | Missing-value indicator for `rep_share_krinski_golden` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Krinski Golden. | 0.00 | 0.00% |
| 571 | rep_share_kristi_fischer_missing | Missing-value indicator for `rep_share_kristi_fischer` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Kristi Fischer. | 0.00 | 0.00% |
| 583 | rep_share_lukasz_jaszczur_missing | Missing-value indicator for `rep_share_lukasz_jaszczur` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Lukasz Jaszczur. | 0.00 | 0.00% |
| 569 | rep_share_mandy_douglas_missing | Missing-value indicator for `rep_share_mandy_douglas` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Mandy Douglas. | 0.00 | 0.00% |
| 568 | rep_share_matthew_everett_missing | Missing-value indicator for `rep_share_matthew_everett` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Matthew Everett. | 0.00 | 0.00% |
| 567 | rep_share_michael_dietzen_missing | Missing-value indicator for `rep_share_michael_dietzen` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Michael Dietzen. | 0.00 | 0.00% |
| 566 | rep_share_michael_johnson_missing | Missing-value indicator for `rep_share_michael_johnson` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Michael Johnson. | 0.00 | 0.00% |
| 565 | rep_share_mycroft_roe_missing | Missing-value indicator for `rep_share_mycroft_roe` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Mycroft ROE. | 0.00 | 0.00% |
| 580 | rep_share_nancy_evans_missing | Missing-value indicator for `rep_share_nancy_evans` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Nancy Evans. | 0.00 | 0.00% |
| 579 | rep_share_nicholas_koelliker_missing | Missing-value indicator for `rep_share_nicholas_koelliker` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Nicholas Koelliker. | 0.00 | 0.00% |
| 578 | rep_share_rick_radzai_missing | Missing-value indicator for `rep_share_rick_radzai` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Rick Radzai. | 0.00 | 0.00% |
| 577 | rep_share_rob_lambrecht_missing | Missing-value indicator for `rep_share_rob_lambrecht` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep ROB Lambrecht. | 0.00 | 0.00% |
| 576 | rep_share_robert_baack_missing | Missing-value indicator for `rep_share_robert_baack` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Robert Baack. | 0.00 | 0.00% |
| 575 | rep_share_rosie_ortega_missing | Missing-value indicator for `rep_share_rosie_ortega` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Rosie Ortega. | 0.00 | 0.00% |
| 574 | rep_share_ross_lee_missing | Missing-value indicator for `rep_share_ross_lee` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Ross LEE. | 0.00 | 0.00% |
| 573 | rep_share_ryan_ladle_missing | Missing-value indicator for `rep_share_ryan_ladle` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Ryan Ladle. | 0.00 | 0.00% |
| 318 | rep_share_sam_scholes_missing | Missing-value indicator for `rep_share_sam_scholes` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep SAM Scholes. | 0.00 | 0.00% |
| 317 | rep_share_sarah_corbin_missing | Missing-value indicator for `rep_share_sarah_corbin` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Sarah Corbin. | 0.00 | 0.00% |
| 316 | rep_share_stephen_gordon_missing | Missing-value indicator for `rep_share_stephen_gordon` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Stephen Gordon. | 0.00 | 0.00% |
| 315 | rep_share_suke_lee_missing | Missing-value indicator for `rep_share_suke_lee` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Suke LEE. | 0.00 | 0.00% |
| 314 | rep_share_victor_pimentel_missing | Missing-value indicator for `rep_share_victor_pimentel` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Victor Pimentel. | 0.00 | 0.00% |
| 313 | rep_share_whitney_street_missing | Missing-value indicator for `rep_share_whitney_street` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep Whitney Street. | 0.00 | 0.00% |
| 325 | rep_share_william_eyler_missing | Missing-value indicator for `rep_share_william_eyler` (1 if the engineered value was null or unavailable). Base feature: Share of raw sales-log transactions handled by rep William Eyler. | 0.00 | 0.00% |
| 311 | mb_lift_max_lag60d_missing | Missing-value indicator for `mb_lift_max_lag60d` (1 if the engineered value was null or unavailable). Base feature: Maximum market-basket lift of SKUs purchased at least 60 days before cutoff. | 0.00 | 0.00% |
| 584 | mb_lift_mean_lag60d_missing | Missing-value indicator for `mb_lift_mean_lag60d` (1 if the engineered value was null or unavailable). Base feature: Mean market-basket lift of SKUs purchased at least 60 days before cutoff. | 0.00 | 0.00% |
| 424 | assets_rollup_3dx_revenue_missing | Missing-value indicator for `assets_rollup_3dx_revenue` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the 3DX Revenue rollup at the cutoff. | 0.00 | 0.00% |
| 324 | assets_rollup_am_software_missing | Missing-value indicator for `assets_rollup_am_software` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the AM Software rollup at the cutoff. | 0.00 | 0.00% |
| 323 | assets_rollup_am_support_missing | Missing-value indicator for `assets_rollup_am_support` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the AM Support rollup at the cutoff. | 0.00 | 0.00% |
| 322 | assets_rollup_altium_pcbworks_missing | Missing-value indicator for `assets_rollup_altium_pcbworks` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Altium Pcbworks rollup at the cutoff. | 0.00 | 0.00% |
| 321 | assets_rollup_artec_missing | Missing-value indicator for `assets_rollup_artec` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Artec rollup at the cutoff. | 0.00 | 0.00% |
| 320 | assets_rollup_camworks_seats_missing | Missing-value indicator for `assets_rollup_camworks_seats` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Camworks Seats rollup at the cutoff. | 0.00 | 0.00% |
| 319 | assets_rollup_catia_missing | Missing-value indicator for `assets_rollup_catia` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the CATIA rollup at the cutoff. | 0.00 | 0.00% |
| 592 | assets_rollup_consumables_missing | Missing-value indicator for `assets_rollup_consumables` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Consumables rollup at the cutoff. | 0.00 | 0.00% |
| 591 | assets_rollup_creaform_missing | Missing-value indicator for `assets_rollup_creaform` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Creaform rollup at the cutoff. | 0.00 | 0.00% |
| 590 | assets_rollup_delmia_missing | Missing-value indicator for `assets_rollup_delmia` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Delmia rollup at the cutoff. | 0.00 | 0.00% |
| 589 | assets_rollup_draftsight_missing | Missing-value indicator for `assets_rollup_draftsight` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Draftsight rollup at the cutoff. | 0.00 | 0.00% |
| 588 | assets_rollup_epdm_cad_editor_seats_missing | Missing-value indicator for `assets_rollup_epdm_cad_editor_seats` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Epdm CAD Editor Seats rollup at the cutoff. | 0.00 | 0.00% |
| 587 | assets_rollup_fdm_missing | Missing-value indicator for `assets_rollup_fdm` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the FDM rollup at the cutoff. | 0.00 | 0.00% |
| 466 | assets_rollup_formlabs_missing | Missing-value indicator for `assets_rollup_formlabs` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Formlabs rollup at the cutoff. | 0.00 | 0.00% |
| 585 | assets_rollup_geomagic_missing | Missing-value indicator for `assets_rollup_geomagic` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Geomagic rollup at the cutoff. | 0.00 | 0.00% |
| 600 | assets_rollup_hv_simulation_missing | Missing-value indicator for `assets_rollup_hv_simulation` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the HV Simulation rollup at the cutoff. | 0.00 | 0.00% |
| 599 | assets_rollup_metals_missing | Missing-value indicator for `assets_rollup_metals` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Metals rollup at the cutoff. | 0.00 | 0.00% |
| 598 | assets_rollup_misc_seats_missing | Missing-value indicator for `assets_rollup_misc_seats` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Misc Seats rollup at the cutoff. | 0.00 | 0.00% |
| 597 | assets_rollup_none_missing | Missing-value indicator for `assets_rollup_none` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the None rollup at the cutoff. | 0.00 | 0.00% |
| 609 | assets_rollup_other_misc_missing | Missing-value indicator for `assets_rollup_other_misc` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Other Misc rollup at the cutoff. | 0.00 | 0.00% |
| 595 | assets_rollup_p3_missing | Missing-value indicator for `assets_rollup_p3` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the P3 rollup at the cutoff. | 0.00 | 0.00% |
| 594 | assets_rollup_polyjet_missing | Missing-value indicator for `assets_rollup_polyjet` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Polyjet rollup at the cutoff. | 0.00 | 0.00% |
| 593 | assets_rollup_post_processing_missing | Missing-value indicator for `assets_rollup_post_processing` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Post Processing rollup at the cutoff. | 0.00 | 0.00% |
| 351 | assets_rollup_pro_prem_new_uap_missing | Missing-value indicator for `assets_rollup_pro_prem_new_uap` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the PRO Prem NEW UAP rollup at the cutoff. | 0.00 | 0.00% |
| 337 | assets_rollup_saf_missing | Missing-value indicator for `assets_rollup_saf` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the SAF rollup at the cutoff. | 0.00 | 0.00% |
| 336 | assets_rollup_sla_missing | Missing-value indicator for `assets_rollup_sla` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the SLA rollup at the cutoff. | 0.00 | 0.00% |
| 335 | assets_rollup_sw_electrical_missing | Missing-value indicator for `assets_rollup_sw_electrical` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the SW Electrical rollup at the cutoff. | 0.00 | 0.00% |
| 334 | assets_rollup_sw_inspection_missing | Missing-value indicator for `assets_rollup_sw_inspection` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the SW Inspection rollup at the cutoff. | 0.00 | 0.00% |
| 333 | assets_rollup_sw_plastics_missing | Missing-value indicator for `assets_rollup_sw_plastics` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the SW Plastics rollup at the cutoff. | 0.00 | 0.00% |
| 332 | assets_rollup_swood_missing | Missing-value indicator for `assets_rollup_swood` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Swood rollup at the cutoff. | 0.00 | 0.00% |
| 331 | assets_rollup_swx_core_missing | Missing-value indicator for `assets_rollup_swx_core` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the SWX Core rollup at the cutoff. | 0.00 | 0.00% |
| 346 | assets_rollup_swx_pro_prem_missing | Missing-value indicator for `assets_rollup_swx_pro_prem` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the SWX PRO Prem rollup at the cutoff. | 0.00 | 0.00% |
| 345 | assets_rollup_service_missing | Missing-value indicator for `assets_rollup_service` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Service rollup at the cutoff. | 0.00 | 0.00% |
| 344 | assets_rollup_simulation_missing | Missing-value indicator for `assets_rollup_simulation` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Simulation rollup at the cutoff. | 0.00 | 0.00% |
| 343 | assets_rollup_training_missing | Missing-value indicator for `assets_rollup_training` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Training rollup at the cutoff. | 0.00 | 0.00% |
| 342 | assets_rollup_unidentified_missing | Missing-value indicator for `assets_rollup_unidentified` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the Unidentified rollup at the cutoff. | 0.00 | 0.00% |
| 341 | assets_rollup_yxc_renewal_missing | Missing-value indicator for `assets_rollup_yxc_renewal` (1 if the engineered value was null or unavailable). Base feature: Active asset quantity for the YXC Renewal rollup at the cutoff. | 0.00 | 0.00% |
| 340 | assets_active_total_missing | Missing-value indicator for `assets_active_total` (1 if the engineered value was null or unavailable). Base feature: Total quantity of active assets at the cutoff. | 0.00 | 0.00% |
| 339 | assets_tenure_days_missing | Missing-value indicator for `assets_tenure_days` (1 if the engineered value was null or unavailable). Base feature: Days since the first effective asset purchase at the cutoff. | 0.00 | 0.00% |
| 302 | assets_bad_purchase_share_missing | Missing-value indicator for `assets_bad_purchase_share` (1 if the engineered value was null or unavailable). Base feature: Share of active assets with missing or invalid purchase dates. | 0.00 | 0.00% |
| 301 | assets_on_subs_total_missing | Missing-value indicator for `assets_on_subs_total` (1 if the engineered value was null or unavailable). Base feature: Total quantity of assets currently on subscription at the cutoff. | 0.00 | 0.00% |
| 300 | assets_off_subs_total_missing | Missing-value indicator for `assets_off_subs_total` (1 if the engineered value was null or unavailable). Base feature: Total quantity of assets that have churned off subscription before the cutoff. | 0.00 | 0.00% |
| 338 | assets_on_subs_3dx_revenue_missing | Missing-value indicator for `assets_on_subs_3dx_revenue` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the 3DX Revenue rollup at the cutoff. | 0.00 | 0.00% |
| 350 | assets_on_subs_am_software_missing | Missing-value indicator for `assets_on_subs_am_software` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the AM Software rollup at the cutoff. | 0.00 | 0.00% |
| 349 | assets_on_subs_am_support_missing | Missing-value indicator for `assets_on_subs_am_support` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the AM Support rollup at the cutoff. | 0.00 | 0.00% |
| 348 | assets_on_subs_altium_pcbworks_missing | Missing-value indicator for `assets_on_subs_altium_pcbworks` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Altium Pcbworks rollup at the cutoff. | 0.00 | 0.00% |
| 347 | assets_on_subs_artec_missing | Missing-value indicator for `assets_on_subs_artec` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Artec rollup at the cutoff. | 0.00 | 0.00% |
| 310 | assets_on_subs_camworks_seats_missing | Missing-value indicator for `assets_on_subs_camworks_seats` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Camworks Seats rollup at the cutoff. | 0.00 | 0.00% |
| 309 | assets_on_subs_catia_missing | Missing-value indicator for `assets_on_subs_catia` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the CATIA rollup at the cutoff. | 0.00 | 0.00% |
| 308 | assets_on_subs_consumables_missing | Missing-value indicator for `assets_on_subs_consumables` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Consumables rollup at the cutoff. | 0.00 | 0.00% |
| 307 | assets_on_subs_creaform_missing | Missing-value indicator for `assets_on_subs_creaform` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Creaform rollup at the cutoff. | 0.00 | 0.00% |
| 306 | assets_on_subs_delmia_missing | Missing-value indicator for `assets_on_subs_delmia` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Delmia rollup at the cutoff. | 0.00 | 0.00% |
| 305 | assets_on_subs_draftsight_missing | Missing-value indicator for `assets_on_subs_draftsight` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Draftsight rollup at the cutoff. | 0.00 | 0.00% |
| 304 | assets_on_subs_epdm_cad_editor_seats_missing | Missing-value indicator for `assets_on_subs_epdm_cad_editor_seats` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Epdm CAD Editor Seats rollup at the cutoff. | 0.00 | 0.00% |
| 303 | assets_on_subs_fdm_missing | Missing-value indicator for `assets_on_subs_fdm` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the FDM rollup at the cutoff. | 0.00 | 0.00% |
| 256 | assets_on_subs_formlabs_missing | Missing-value indicator for `assets_on_subs_formlabs` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Formlabs rollup at the cutoff. | 0.00 | 0.00% |
| 255 | assets_on_subs_geomagic_missing | Missing-value indicator for `assets_on_subs_geomagic` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Geomagic rollup at the cutoff. | 0.00 | 0.00% |
| 254 | assets_on_subs_hv_simulation_missing | Missing-value indicator for `assets_on_subs_hv_simulation` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the HV Simulation rollup at the cutoff. | 0.00 | 0.00% |
| 252 | assets_on_subs_metals_missing | Missing-value indicator for `assets_on_subs_metals` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Metals rollup at the cutoff. | 0.00 | 0.00% |
| 251 | assets_on_subs_misc_seats_missing | Missing-value indicator for `assets_on_subs_misc_seats` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Misc Seats rollup at the cutoff. | 0.00 | 0.00% |
| 250 | assets_on_subs_none_missing | Missing-value indicator for `assets_on_subs_none` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the None rollup at the cutoff. | 0.00 | 0.00% |
| 298 | assets_on_subs_other_misc_missing | Missing-value indicator for `assets_on_subs_other_misc` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Other Misc rollup at the cutoff. | 0.00 | 0.00% |
| 286 | assets_on_subs_p3_missing | Missing-value indicator for `assets_on_subs_p3` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the P3 rollup at the cutoff. | 0.00 | 0.00% |
| 264 | assets_on_subs_polyjet_missing | Missing-value indicator for `assets_on_subs_polyjet` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Polyjet rollup at the cutoff. | 0.00 | 0.00% |
| 262 | assets_on_subs_post_processing_missing | Missing-value indicator for `assets_on_subs_post_processing` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Post Processing rollup at the cutoff. | 0.00 | 0.00% |
| 261 | assets_on_subs_pro_prem_new_uap_missing | Missing-value indicator for `assets_on_subs_pro_prem_new_uap` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the PRO Prem NEW UAP rollup at the cutoff. | 0.00 | 0.00% |
| 273 | assets_on_subs_saf_missing | Missing-value indicator for `assets_on_subs_saf` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the SAF rollup at the cutoff. | 0.00 | 0.00% |
| 260 | assets_on_subs_sla_missing | Missing-value indicator for `assets_on_subs_sla` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the SLA rollup at the cutoff. | 0.00 | 0.00% |
| 259 | assets_on_subs_sw_electrical_missing | Missing-value indicator for `assets_on_subs_sw_electrical` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the SW Electrical rollup at the cutoff. | 0.00 | 0.00% |
| 258 | assets_on_subs_sw_inspection_missing | Missing-value indicator for `assets_on_subs_sw_inspection` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the SW Inspection rollup at the cutoff. | 0.00 | 0.00% |
| 257 | assets_on_subs_sw_plastics_missing | Missing-value indicator for `assets_on_subs_sw_plastics` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the SW Plastics rollup at the cutoff. | 0.00 | 0.00% |
| 265 | assets_on_subs_swood_missing | Missing-value indicator for `assets_on_subs_swood` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Swood rollup at the cutoff. | 0.00 | 0.00% |
| 266 | assets_on_subs_swx_core_missing | Missing-value indicator for `assets_on_subs_swx_core` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the SWX Core rollup at the cutoff. | 0.00 | 0.00% |
| 267 | assets_on_subs_swx_pro_prem_missing | Missing-value indicator for `assets_on_subs_swx_pro_prem` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the SWX PRO Prem rollup at the cutoff. | 0.00 | 0.00% |
| 271 | assets_on_subs_service_missing | Missing-value indicator for `assets_on_subs_service` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Service rollup at the cutoff. | 0.00 | 0.00% |
| 272 | assets_on_subs_simulation_missing | Missing-value indicator for `assets_on_subs_simulation` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Simulation rollup at the cutoff. | 0.00 | 0.00% |
| 270 | assets_on_subs_training_missing | Missing-value indicator for `assets_on_subs_training` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Training rollup at the cutoff. | 0.00 | 0.00% |
| 269 | assets_on_subs_unidentified_missing | Missing-value indicator for `assets_on_subs_unidentified` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the Unidentified rollup at the cutoff. | 0.00 | 0.00% |
| 268 | assets_on_subs_yxc_renewal_missing | Missing-value indicator for `assets_on_subs_yxc_renewal` (1 if the engineered value was null or unavailable). Base feature: Quantity of assets on subscription for the YXC Renewal rollup at the cutoff. | 0.00 | 0.00% |
| 330 | assets_off_subs_3dx_revenue_missing | Missing-value indicator for `assets_off_subs_3dx_revenue` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the 3DX Revenue rollup at the cutoff. | 0.00 | 0.00% |
| 329 | assets_off_subs_am_software_missing | Missing-value indicator for `assets_off_subs_am_software` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the AM Software rollup at the cutoff. | 0.00 | 0.00% |
| 312 | assets_off_subs_am_support_missing | Missing-value indicator for `assets_off_subs_am_support` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the AM Support rollup at the cutoff. | 0.00 | 0.00% |
| 327 | assets_off_subs_altium_pcbworks_missing | Missing-value indicator for `assets_off_subs_altium_pcbworks` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Altium Pcbworks rollup at the cutoff. | 0.00 | 0.00% |
| 326 | assets_off_subs_artec_missing | Missing-value indicator for `assets_off_subs_artec` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Artec rollup at the cutoff. | 0.00 | 0.00% |
| 297 | assets_off_subs_camworks_seats_missing | Missing-value indicator for `assets_off_subs_camworks_seats` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Camworks Seats rollup at the cutoff. | 0.00 | 0.00% |
| 253 | assets_off_subs_catia_missing | Missing-value indicator for `assets_off_subs_catia` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the CATIA rollup at the cutoff. | 0.00 | 0.00% |
| 263 | assets_off_subs_consumables_missing | Missing-value indicator for `assets_off_subs_consumables` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Consumables rollup at the cutoff. | 0.00 | 0.00% |
| 427 | assets_off_subs_creaform_missing | Missing-value indicator for `assets_off_subs_creaform` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Creaform rollup at the cutoff. | 0.00 | 0.00% |
| 426 | assets_off_subs_delmia_missing | Missing-value indicator for `assets_off_subs_delmia` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Delmia rollup at the cutoff. | 0.00 | 0.00% |
| 425 | assets_off_subs_draftsight_missing | Missing-value indicator for `assets_off_subs_draftsight` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Draftsight rollup at the cutoff. | 0.00 | 0.00% |
| 363 | assets_off_subs_epdm_cad_editor_seats_missing | Missing-value indicator for `assets_off_subs_epdm_cad_editor_seats` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Epdm CAD Editor Seats rollup at the cutoff. | 0.00 | 0.00% |
| 423 | assets_off_subs_fdm_missing | Missing-value indicator for `assets_off_subs_fdm` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the FDM rollup at the cutoff. | 0.00 | 0.00% |
| 422 | assets_off_subs_geomagic_missing | Missing-value indicator for `assets_off_subs_geomagic` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Geomagic rollup at the cutoff. | 0.00 | 0.00% |
| 421 | assets_off_subs_hv_simulation_missing | Missing-value indicator for `assets_off_subs_hv_simulation` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the HV Simulation rollup at the cutoff. | 0.00 | 0.00% |
| 420 | assets_off_subs_metals_missing | Missing-value indicator for `assets_off_subs_metals` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Metals rollup at the cutoff. | 0.00 | 0.00% |
| 280 | assets_off_subs_misc_seats_missing | Missing-value indicator for `assets_off_subs_misc_seats` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Misc Seats rollup at the cutoff. | 0.00 | 0.00% |
| 279 | assets_off_subs_none_missing | Missing-value indicator for `assets_off_subs_none` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the None rollup at the cutoff. | 0.00 | 0.00% |
| 278 | assets_off_subs_other_misc_missing | Missing-value indicator for `assets_off_subs_other_misc` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Other Misc rollup at the cutoff. | 0.00 | 0.00% |
| 277 | assets_off_subs_p3_missing | Missing-value indicator for `assets_off_subs_p3` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the P3 rollup at the cutoff. | 0.00 | 0.00% |
| 276 | assets_off_subs_polyjet_missing | Missing-value indicator for `assets_off_subs_polyjet` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Polyjet rollup at the cutoff. | 0.00 | 0.00% |
| 275 | assets_off_subs_post_processing_missing | Missing-value indicator for `assets_off_subs_post_processing` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Post Processing rollup at the cutoff. | 0.00 | 0.00% |
| 274 | assets_off_subs_pro_prem_new_uap_missing | Missing-value indicator for `assets_off_subs_pro_prem_new_uap` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the PRO Prem NEW UAP rollup at the cutoff. | 0.00 | 0.00% |
| 328 | assets_off_subs_saf_missing | Missing-value indicator for `assets_off_subs_saf` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the SAF rollup at the cutoff. | 0.00 | 0.00% |
| 288 | assets_off_subs_sla_missing | Missing-value indicator for `assets_off_subs_sla` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the SLA rollup at the cutoff. | 0.00 | 0.00% |
| 287 | assets_off_subs_sw_electrical_missing | Missing-value indicator for `assets_off_subs_sw_electrical` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the SW Electrical rollup at the cutoff. | 0.00 | 0.00% |
| 299 | assets_off_subs_sw_inspection_missing | Missing-value indicator for `assets_off_subs_sw_inspection` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the SW Inspection rollup at the cutoff. | 0.00 | 0.00% |
| 285 | assets_off_subs_sw_plastics_missing | Missing-value indicator for `assets_off_subs_sw_plastics` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the SW Plastics rollup at the cutoff. | 0.00 | 0.00% |
| 284 | assets_off_subs_swx_core_missing | Missing-value indicator for `assets_off_subs_swx_core` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the SWX Core rollup at the cutoff. | 0.00 | 0.00% |
| 283 | assets_off_subs_swx_pro_prem_missing | Missing-value indicator for `assets_off_subs_swx_pro_prem` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the SWX PRO Prem rollup at the cutoff. | 0.00 | 0.00% |
| 282 | assets_off_subs_service_missing | Missing-value indicator for `assets_off_subs_service` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Service rollup at the cutoff. | 0.00 | 0.00% |
| 281 | assets_off_subs_simulation_missing | Missing-value indicator for `assets_off_subs_simulation` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Simulation rollup at the cutoff. | 0.00 | 0.00% |
| 296 | assets_off_subs_training_missing | Missing-value indicator for `assets_off_subs_training` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Training rollup at the cutoff. | 0.00 | 0.00% |
| 295 | assets_off_subs_unidentified_missing | Missing-value indicator for `assets_off_subs_unidentified` (1 if the engineered value was null or unavailable). Base feature: Quantity of churned/off-subscription assets for the Unidentified rollup at the cutoff. | 0.00 | 0.00% |
| 294 | ever_acr_missing | Missing-value indicator for `ever_acr` (1 if the engineered value was null or unavailable). Base feature: Flag indicating the account was ever marked as an Account Coverage Review (ACR) in sales logs. | 0.00 | 0.00% |
| 293 | ever_new_customer_missing | Missing-value indicator for `ever_new_customer` (1 if the engineered value was null or unavailable). Base feature: Flag indicating the account was ever marked as a new customer in sales logs. | 0.00 | 0.00% |
| 292 | rfm__all__tx_n__24m_missing | Missing-value indicator for `rfm__all__tx_n__24m` (1 if the engineered value was null or unavailable). Base feature: Transaction count for all divisions over the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 291 | rfm__all__gp_sum__24m_missing | Missing-value indicator for `rfm__all__gp_sum__24m` (1 if the engineered value was null or unavailable). Base feature: Gross profit total for all divisions over the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 290 | rfm__all__gp_mean__24m_missing | Missing-value indicator for `rfm__all__gp_mean__24m` (1 if the engineered value was null or unavailable). Base feature: Average gross profit per transaction for all divisions over the trailing 24-month window ending at the cutoff. | 0.00 | 0.00% |
| 289 | rfm__all__gp_sum__delta_12m_prev12m_missing | Missing-value indicator for `rfm__all__gp_sum__delta_12m_prev12m` (1 if the engineered value was null or unavailable). Base feature: Change in gross profit for all divisions: last 12 months minus the preceding 12 months. | 0.00 | 0.00% |
| 447 | rfm__all__gp_sum__ratio_12m_prev12m_missing | Missing-value indicator for `rfm__all__gp_sum__ratio_12m_prev12m` (1 if the engineered value was null or unavailable). Base feature: Ratio of gross profit for all divisions: last 12 months divided by the preceding 12 months. | 0.00 | 0.00% |
| 446 | rfm__all__tx_n__delta_12m_prev12m_missing | Missing-value indicator for `rfm__all__tx_n__delta_12m_prev12m` (1 if the engineered value was null or unavailable). Base feature: Change in transaction count for all divisions: last 12 months minus the preceding 12 months. | 0.00 | 0.00% |
| 445 | rfm__all__tx_n__ratio_12m_prev12m_missing | Missing-value indicator for `rfm__all__tx_n__ratio_12m_prev12m` (1 if the engineered value was null or unavailable). Base feature: Ratio of transaction count for all divisions: last 12 months divided by the preceding 12 months. | 0.00 | 0.00% |
| 444 | rfm__div__gp_sum__delta_12m_prev12m_missing | Missing-value indicator for `rfm__div__gp_sum__delta_12m_prev12m` (1 if the engineered value was null or unavailable). Base feature: Change in gross profit for the target division: last 12 months minus the preceding 12 months. | 0.00 | 0.00% |
| 443 | rfm__div__gp_sum__ratio_12m_prev12m_missing | Missing-value indicator for `rfm__div__gp_sum__ratio_12m_prev12m` (1 if the engineered value was null or unavailable). Base feature: Ratio of gross profit for the target division: last 12 months divided by the preceding 12 months. | 0.00 | 0.00% |
| 442 | rfm__div__tx_n__delta_12m_prev12m_missing | Missing-value indicator for `rfm__div__tx_n__delta_12m_prev12m` (1 if the engineered value was null or unavailable). Base feature: Change in transaction count for the target division: last 12 months minus the preceding 12 months. | 0.00 | 0.00% |
| 454 | rfm__div__tx_n__ratio_12m_prev12m_missing | Missing-value indicator for `rfm__div__tx_n__ratio_12m_prev12m` (1 if the engineered value was null or unavailable). Base feature: Ratio of transaction count for the target division: last 12 months divided by the preceding 12 months. | 0.00 | 0.00% |
| 440 | lifecycle__all__tenure_days__life_missing | Missing-value indicator for `lifecycle__all__tenure_days__life` (1 if the engineered value was null or unavailable). Base feature: Lifecycle metric capturing tenure days over the customer lifetime. | 0.00 | 0.00% |
| 403 | lifecycle__all__tenure_months__life_missing | Missing-value indicator for `lifecycle__all__tenure_months__life` (1 if the engineered value was null or unavailable). Base feature: Lifecycle metric capturing tenure months over the customer lifetime. | 0.00 | 0.00% |
| 441 | lifecycle__all__tenure_bucket__lt3m_missing | Missing-value indicator for `lifecycle__all__tenure_bucket__lt3m` (1 if the engineered value was null or unavailable). Base feature: Indicator that customer tenure at cutoff falls in the <3 months bucket. | 0.00 | 0.00% |
| 453 | lifecycle__all__tenure_bucket__3to6m_missing | Missing-value indicator for `lifecycle__all__tenure_bucket__3to6m` (1 if the engineered value was null or unavailable). Base feature: Indicator that customer tenure at cutoff falls in the 3–6 months bucket. | 0.00 | 0.00% |
| 452 | lifecycle__all__tenure_bucket__6to12m_missing | Missing-value indicator for `lifecycle__all__tenure_bucket__6to12m` (1 if the engineered value was null or unavailable). Base feature: Indicator that customer tenure at cutoff falls in the 6–12 months bucket. | 0.00 | 0.00% |
| 451 | lifecycle__all__tenure_bucket__1to2y_missing | Missing-value indicator for `lifecycle__all__tenure_bucket__1to2y` (1 if the engineered value was null or unavailable). Base feature: Indicator that customer tenure at cutoff falls in the 1–2 years bucket. | 0.00 | 0.00% |
| 450 | lifecycle__all__tenure_bucket__ge2y_missing | Missing-value indicator for `lifecycle__all__tenure_bucket__ge2y` (1 if the engineered value was null or unavailable). Base feature: Indicator that customer tenure at cutoff falls in the ≥2 years bucket. | 0.00 | 0.00% |
| 449 | lifecycle__all__gap_days__life_missing | Missing-value indicator for `lifecycle__all__gap_days__life` (1 if the engineered value was null or unavailable). Base feature: Lifecycle metric capturing gap days over the customer lifetime. | 0.00 | 0.00% |
| 448 | season__all__q1_share__24m_missing | Missing-value indicator for `season__all__q1_share__24m` (1 if the engineered value was null or unavailable). Base feature: Share of transactions occurring in calendar quarter Q1 during the last 24 months. | 0.00 | 0.00% |
| 411 | season__all__q2_share__24m_missing | Missing-value indicator for `season__all__q2_share__24m` (1 if the engineered value was null or unavailable). Base feature: Share of transactions occurring in calendar quarter Q2 during the last 24 months. | 0.00 | 0.00% |
| 410 | season__all__q3_share__24m_missing | Missing-value indicator for `season__all__q3_share__24m` (1 if the engineered value was null or unavailable). Base feature: Share of transactions occurring in calendar quarter Q3 during the last 24 months. | 0.00 | 0.00% |
| 409 | season__all__q4_share__24m_missing | Missing-value indicator for `season__all__q4_share__24m` (1 if the engineered value was null or unavailable). Base feature: Share of transactions occurring in calendar quarter Q4 during the last 24 months. | 0.00 | 0.00% |
