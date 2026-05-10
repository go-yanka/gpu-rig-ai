# CBIC RAG Eval Run

Run dir: `runs\p1_3_evalB_20260422_041318`
Score: **383.0 / 1439.0** (26.62%)
Items: 170  |  Errors: 0
Latency ms — median 33129.0  p95 61089  max 75736

## By category

| Category | N | Points | Max | % | Errors |
|---|---:|---:|---:|---:|---:|
| central_excise | 10 | 23.0 | 88.0 | 26.14 | 0 |
| customs | 45 | 71.0 | 384.0 | 18.49 | 0 |
| gst | 81 | 215.0 | 681.0 | 31.57 | 0 |
| others | 14 | 26.0 | 121.0 | 21.49 | 0 |
| service_tax | 20 | 48.0 | 165.0 | 29.09 | 0 |

## Per-item

| id | cat | diff | pts | max | judge | lat_ms | err |
|---|---|---|---:|---:|---:|---:|---|
| gst_pos_001 | gst | basic | 0.0 | 9.0 | 0 | 31972 |  |
| gst_pos_002 | gst | intermediate | 3.0 | 9.0 | 1 | 40656 |  |
| gst_pos_003 | gst | basic | 0.0 | 8.0 | 0 | 23940 |  |
| gst_pos_004 | gst | intermediate | 0.0 | 8.0 | 0 | 24104 |  |
| gst_pos_005 | gst | complex | 3.0 | 13.0 | 1 | 47082 |  |
| gst_cs_001 | gst | intermediate | 6.0 | 10.0 | 2 | 29591 |  |
| gst_cs_002 | gst | basic | 4.0 | 8.0 | 2 | 29562 |  |
| gst_itc_001 | gst | basic | 3.0 | 8.0 | 3 | 52780 |  |
| gst_itc_002 | gst | intermediate | 3.0 | 10.0 | 1 | 45344 |  |
| gst_itc_003 | gst | intermediate | 4.0 | 10.0 | 2 | 33662 |  |
| gst_itc_004 | gst | basic | 3.0 | 10.0 | 2 | 45816 |  |
| gst_tos_001 | gst | basic | 4.0 | 9.0 | 1 | 36964 |  |
| gst_tos_002 | gst | basic | 3.0 | 9.0 | 1 | 28575 |  |
| gst_inv_001 | gst | basic | 0.0 | 8.0 | 0 | 22917 |  |
| gst_inv_002 | gst | intermediate | 0.0 | 9.0 | 0 | 18289 |  |
| gst_rcm_001 | gst | basic | 7.0 | 9.0 | 3 | 28795 |  |
| gst_rcm_002 | gst | intermediate | 0.0 | 9.0 | 0 | 20553 |  |
| gst_ref_001 | gst | intermediate | 7.0 | 9.0 | 3 | 37909 |  |
| gst_ewb_001 | gst | basic | 4.0 | 9.0 | 2 | 25556 |  |
| gst_exp_001 | gst | intermediate | 7.0 | 10.0 | 3 | 43183 |  |
| cus_val_001 | customs | basic | 0.0 | 8.0 | 0 | 21514 |  |
| cus_val_002 | customs | intermediate | 0.0 | 10.0 | 0 | 36978 |  |
| cus_cls_001 | customs | basic | 2.0 | 10.0 | 1 | 30435 |  |
| cus_wh_001 | customs | intermediate | 3.0 | 10.0 | 1 | 27062 |  |
| cus_db_001 | customs | basic | 5.0 | 9.0 | 2 | 31389 |  |
| cus_db_002 | customs | intermediate | 0.0 | 8.0 | 0 | 24939 |  |
| cus_svb_001 | customs | intermediate | 4.0 | 8.0 | 2 | 25068 |  |
| cus_ar_001 | customs | basic | 6.0 | 11.0 | 2 | 37579 |  |
| cus_cls_002 | customs | complex | 0.0 | 11.0 | 0 | 19938 |  |
| cus_ig_001 | customs | basic | 3.0 | 11.0 | 2 | 30760 |  |
| exc_val_001 | central_excise | basic | 0.0 | 9.0 | 0 | 25391 |  |
| exc_val_002 | central_excise | intermediate | 3.0 | 9.0 | 1 | 26760 |  |
| exc_cen_001 | central_excise | basic | 0.0 | 9.0 | 0 | 24862 |  |
| exc_cen_002 | central_excise | intermediate | 7.0 | 9.0 | 2 | 33626 |  |
| exc_man_001 | central_excise | basic | 3.0 | 9.0 | 2 | 23781 |  |
| exc_man_002 | central_excise | intermediate | 4.0 | 10.0 | 2 | 26578 |  |
| exc_ssi_001 | central_excise | basic | 0.0 | 8.0 | 0 | 15674 |  |
| exc_ssi_002 | central_excise | complex | 1.0 | 8.0 | 1 | 25133 |  |
| st_val_001 | service_tax | basic | 0.0 | 9.0 | 0 | 27514 |  |
| st_nl_001 | service_tax | basic | 4.0 | 11.0 | 1 | 32856 |  |
| st_exp_001 | service_tax | complex | 0.0 | 10.0 | 0 | 26925 |  |
| st_pos_001 | service_tax | basic | 4.0 | 7.0 | 2 | 34564 |  |
| st_rcm_001 | service_tax | intermediate | 4.0 | 10.0 | 1 | 30361 |  |
| st_lev_001 | service_tax | basic | 4.0 | 10.0 | 2 | 30448 |  |
| oth_ap_001 | others | basic | 2.0 | 10.0 | 1 | 33137 |  |
| oth_ap_002 | others | basic | 5.0 | 10.0 | 2 | 26489 |  |
| oth_ap_003 | others | intermediate | 4.0 | 10.0 | 2 | 30694 |  |
| oth_gaar_001 | others | basic | 1.0 | 8.0 | 1 | 17908 |  |
| oth_pen_001 | others | complex | 3.0 | 13.0 | 1 | 37826 |  |
| oth_cls_001 | others | intermediate | 3.0 | 10.0 | 1 | 35250 |  |
| gst_rate_001 | gst | basic | 0.0 | 7.0 | 0 | 12935 |  |
| gst_sac_001 | gst | basic | 3.0 | 7.0 | 1 | 34952 |  |
| gst_exempt_001 | gst | basic | 0.0 | 7.0 | 0 | 23773 |  |
| gst_cess_001 | gst | basic | 0.0 | 8.0 | 0 | 22464 |  |
| customs_rate_001 | customs | basic | 3.0 | 8.0 | 1 | 32404 |  |
| gst_rate_002 | gst | basic | 5.0 | 7.0 | 3 | 31216 |  |
| gst_hsn_001 | gst | basic | 4.0 | 8.0 | 3 | 26102 |  |
| gst_sac_002 | gst | basic | 2.0 | 7.0 | 1 | 24236 |  |
| gst_notif_001 | gst | basic | 1.0 | 6.0 | 1 | 17635 |  |
| customs_rate_002 | customs | basic | 0.0 | 8.0 | 0 | 25145 |  |
| gst_rate_003 | gst | basic | 0.0 | 7.0 | 0 | 19796 |  |
| gst_cess_002 | gst | basic | 0.0 | 7.0 | 0 | 23643 |  |
| gst_ccy_supply_001 | gst | intermediate | 5.0 | 9.0 | 2 | 38903 |  |
| customs_notif_001 | customs | intermediate | 2.0 | 7.0 | 1 | 35517 |  |
| gst_notif_002 | gst | intermediate | 3.0 | 7.0 | 1 | 33259 |  |
| gst_inverted_001 | gst | intermediate | 4.0 | 8.0 | 3 | 54172 |  |
| gst_sac_003 | gst | intermediate | 0.0 | 7.0 | 0 | 23821 |  |
| gst_exempt_002 | gst | intermediate | 2.0 | 8.0 | 1 | 34332 |  |
| customs_rate_003 | customs | intermediate | 3.0 | 10.0 | 2 | 31224 |  |
| gst_hsn_002 | gst | intermediate | 5.0 | 8.0 | 2 | 42754 |  |
| gst_rate_004 | gst | intermediate | 4.0 | 6.0 | 3 | 50371 |  |
| gst_inverted_002 | gst | complex | 2.0 | 8.0 | 1 | 72784 |  |
| gst_ccy_supply_002 | gst | complex | 3.0 | 10.0 | 2 | 43526 |  |
| customs_rate_004 | customs | complex | 2.0 | 9.0 | 1 | 58080 |  |
| gst_notif_003 | gst | complex | 3.0 | 8.0 | 1 | 54513 |  |
| gst_refuse_001 | gst | basic | 0.0 | 7.0 | 0 | 26988 |  |
| customs_refuse_002 | customs | basic | 0.0 | 7.0 | 0 | 18517 |  |
| gst_refuse_003 | gst | basic | 2.0 | 8.0 | 1 | 34109 |  |
| others_refuse_004 | others | basic | 0.0 | 7.0 | 1 | 31541 |  |
| others_refuse_005 | others | basic | 0.0 | 8.0 | 0 | 20125 |  |
| gst_refuse_006 | gst | basic | 0.0 | 7.0 | 0 | 24547 |  |
| customs_refuse_007 | customs | basic | 0.0 | 7.0 | 0 | 24771 |  |
| gst_refuse_008 | gst | basic | 0.0 | 7.0 | 1 | 32011 |  |
| others_refuse_009 | others | basic | 1.0 | 8.0 | 1 | 22991 |  |
| others_refuse_010 | others | basic | 0.0 | 7.0 | 0 | 20078 |  |
| gst_complex_001 | gst | complex | 2.0 | 8.0 | 1 | 65901 |  |
| gst_complex_002 | gst | complex | 2.0 | 10.0 | 1 | 61089 |  |
| gst_complex_003 | gst | complex | 3.0 | 11.0 | 2 | 43009 |  |
| gst_complex_004 | gst | complex | 4.0 | 9.0 | 2 | 72974 |  |
| gst_complex_005 | gst | complex | 4.0 | 9.0 | 2 | 49543 |  |
| gst_complex_006 | gst | complex | 3.0 | 9.0 | 1 | 52865 |  |
| gst_complex_007 | gst | complex | 3.0 | 9.0 | 1 | 54612 |  |
| gst_complex_008 | gst | complex | 2.0 | 8.0 | 1 | 74039 |  |
| gst_complex_009 | gst | complex | 4.0 | 11.0 | 3 | 64216 |  |
| gst_complex_010 | gst | complex | 4.0 | 11.0 | 1 | 52232 |  |
| gst_complex_011 | gst | complex | 4.0 | 9.0 | 2 | 46655 |  |
| gst_complex_012 | gst | complex | 0.0 | 10.0 | 0 | 43359 |  |
| gst_complex_013 | gst | complex | 3.0 | 8.0 | 1 | 59867 |  |
| gst_complex_014 | gst | complex | 6.0 | 10.0 | 3 | 50837 |  |
| gst_complex_015 | gst | complex | 3.0 | 9.0 | 2 | 43925 |  |
| customs_complex_016 | customs | complex | 0.0 | 9.0 | 0 | 36845 |  |
| customs_complex_017 | customs | complex | 0.0 | 8.0 | 0 | 45428 |  |
| customs_complex_018 | customs | complex | 3.0 | 9.0 | 1 | 51563 |  |
| customs_complex_019 | customs | complex | 2.0 | 9.0 | 1 | 49589 |  |
| customs_complex_020 | customs | complex | 0.0 | 8.0 | 0 | 41901 |  |
| central_excise_complex_021 | central_excise | complex | 2.0 | 8.0 | 1 | 75736 |  |
| central_excise_complex_022 | central_excise | complex | 3.0 | 9.0 | 2 | 39811 |  |
| service_tax_complex_023 | service_tax | complex | 3.0 | 9.0 | 1 | 73345 |  |
| service_tax_complex_024 | service_tax | complex | 4.0 | 9.0 | 2 | 71463 |  |
| others_complex_025 | others | complex | 2.0 | 9.0 | 1 | 46461 |  |
| st_service_tax_001 | service_tax | basic | 3.0 | 7.0 | 2 | 34560 |  |
| st_service_tax_002 | service_tax | intermediate | 0.0 | 7.0 | 0 | 26106 |  |
| st_service_tax_003 | service_tax | advanced | 4.0 | 7.0 | 3 | 34612 |  |
| st_service_tax_004 | service_tax | basic | 3.0 | 7.0 | 2 | 37824 |  |
| st_service_tax_005 | service_tax | intermediate | 0.0 | 6.0 | 0 | 26395 |  |
| st_service_tax_006 | service_tax | intermediate | 3.0 | 9.0 | 1 | 30475 |  |
| st_service_tax_007 | service_tax | basic | 0.0 | 8.0 | 0 | 22864 |  |
| st_service_tax_008 | service_tax | basic | 3.0 | 8.0 | 2 | 26663 |  |
| st_service_tax_009 | service_tax | advanced | 2.0 | 9.0 | 1 | 33756 |  |
| st_service_tax_010 | service_tax | basic | 3.0 | 8.0 | 2 | 36952 |  |
| st_service_tax_011 | service_tax | intermediate | 0.0 | 7.0 | 0 | 34701 |  |
| st_service_tax_012 | service_tax | basic | 4.0 | 7.0 | 2 | 29702 |  |
| st_transition_001 | gst | intermediate | 6.0 | 9.0 | 3 | 42211 |  |
| st_transition_002 | gst | advanced | 0.0 | 8.0 | 0 | 41204 |  |
| st_transition_003 | gst | intermediate | 3.0 | 8.0 | 1 | 42715 |  |
| st_transition_004 | gst | basic | 0.0 | 9.0 | 0 | 19849 |  |
| st_transition_005 | gst | advanced | 2.0 | 8.0 | 1 | 32523 |  |
| st_refuse_001 | others | basic | 1.0 | 7.0 | 1 | 32624 |  |
| st_refuse_002 | others | basic | 1.0 | 7.0 | 0 | 16918 |  |
| st_refuse_003 | others | intermediate | 3.0 | 7.0 | 1 | 39481 |  |
| cust_customs_valuation_001 | customs | advanced | 1.0 | 8.0 | 1 | 51232 |  |
| cust_customs_valuation_002 | customs | intermediate | 0.0 | 8.0 | 0 | 23915 |  |
| cust_customs_valuation_003 | customs | advanced | 1.0 | 7.0 | 2 | 44935 |  |
| cust_customs_valuation_004 | customs | intermediate | 2.0 | 8.0 | 1 | 40262 |  |
| cust_customs_valuation_005 | customs | basic | 1.0 | 10.0 | 1 | 28660 |  |
| cust_customs_classification_001 | customs | advanced | 0.0 | 8.0 | 0 | 24781 |  |
| cust_customs_classification_002 | customs | intermediate | 0.0 | 7.0 | 0 | 28658 |  |
| cust_customs_classification_003 | customs | basic | 7.0 | 7.0 | 3 | 38455 |  |
| cust_customs_classification_004 | customs | intermediate | 0.0 | 7.0 | 0 | 23245 |  |
| cust_customs_drawback_001 | customs | intermediate | 4.0 | 9.0 | 2 | 37206 |  |
| cust_customs_drawback_002 | customs | advanced | 3.0 | 9.0 | 2 | 50151 |  |
| cust_customs_drawback_003 | customs | intermediate | 1.0 | 9.0 | 1 | 46074 |  |
| cust_customs_drawback_004 | customs | advanced | 3.0 | 9.0 | 2 | 40568 |  |
| cust_customs_warehouse_001 | customs | basic | 0.0 | 9.0 | 0 | 32595 |  |
| cust_customs_warehouse_002 | customs | advanced | 2.0 | 8.0 | 2 | 32446 |  |
| cust_customs_warehouse_003 | customs | intermediate | 0.0 | 9.0 | 0 | 27398 |  |
| cust_customs_adviolation_001 | customs | intermediate | 0.0 | 10.0 | 0 | 33121 |  |
| cust_customs_adviolation_002 | customs | advanced | 0.0 | 8.0 | 0 | 23482 |  |
| cust_customs_exemption_001 | customs | intermediate | 2.0 | 7.0 | 2 | 47811 |  |
| cust_customs_exemption_002 | customs | advanced | 2.0 | 8.0 | 1 | 45747 |  |
| oth_appeals_001 | gst | intermediate | 0.0 | 8.0 | 0 | 28025 |  |
| oth_appeals_002 | gst | intermediate | 0.0 | 8.0 | 0 | 26277 |  |
| oth_appeals_003 | customs | basic | 4.0 | 9.0 | 2 | 40737 |  |
| oth_appeals_004 | customs | intermediate | 0.0 | 8.0 | 0 | 25777 |  |
| oth_appeals_005 | gst | basic | 0.0 | 7.0 | 0 | 25899 |  |
| oth_penalty_001 | gst | intermediate | 4.0 | 7.0 | 3 | 36328 |  |
| oth_penalty_002 | gst | basic | 1.0 | 8.0 | 1 | 31508 |  |
| oth_penalty_003 | gst | advanced | 0.0 | 8.0 | 0 | 23453 |  |
| oth_penalty_004 | customs | intermediate | 0.0 | 7.0 | 0 | 17596 |  |
| oth_advance_ruling_001 | gst | basic | 0.0 | 8.0 | 0 | 20117 |  |
| oth_advance_ruling_002 | gst | intermediate | 0.0 | 7.0 | 0 | 32937 |  |
| oth_advance_ruling_003 | gst | advanced | 5.0 | 9.0 | 2 | 45550 |  |
| oth_advance_ruling_004 | gst | basic | 6.0 | 8.0 | 3 | 46091 |  |
| oth_anti_evasion_001 | gst | intermediate | 6.0 | 7.0 | 3 | 35287 |  |
| oth_anti_evasion_002 | gst | intermediate | 6.0 | 7.0 | 3 | 41605 |  |
| oth_anti_evasion_003 | gst | basic | 6.0 | 8.0 | 3 | 50411 |  |
| oth_offences_001 | gst | intermediate | 5.0 | 10.0 | 3 | 41168 |  |
| oth_offences_002 | gst | advanced | 4.0 | 9.0 | 2 | 45115 |  |
| oth_interest_001 | gst | intermediate | 0.0 | 8.0 | 0 | 38227 |  |
| oth_interest_002 | gst | basic | 5.0 | 8.0 | 2 | 44795 |  |