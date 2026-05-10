# CBIC RAG Eval Run

Run dir: `D:\_gpu_rig_ai\eval\runs\p2_1_evalA_20260422_081545`
Score: **569.0 / 1439.0** (39.54%)
Items: 170  |  Errors: 0
Latency ms — median 26225.5  p95 52941  max 86425

## By category

| Category | N | Points | Max | % | Errors |
|---|---:|---:|---:|---:|---:|
| central_excise | 10 | 43.0 | 88.0 | 48.86 | 0 |
| customs | 45 | 135.0 | 384.0 | 35.16 | 0 |
| gst | 81 | 284.0 | 681.0 | 41.7 | 0 |
| others | 14 | 35.0 | 121.0 | 28.93 | 0 |
| service_tax | 20 | 72.0 | 165.0 | 43.64 | 0 |

## Per-item

| id | cat | diff | pts | max | judge | lat_ms | err |
|---|---|---|---:|---:|---:|---:|---|
| gst_pos_001 | gst | basic | 4.0 | 9.0 | 1 | 34604 |  |
| gst_pos_002 | gst | intermediate | 4.0 | 9.0 | 1 | 26835 |  |
| gst_pos_003 | gst | basic | 4.0 | 8.0 | 3 | 23518 |  |
| gst_pos_004 | gst | intermediate | 5.0 | 8.0 | 3 | 24454 |  |
| gst_pos_005 | gst | complex | 4.0 | 13.0 | 2 | 29103 |  |
| gst_cs_001 | gst | intermediate | 6.0 | 10.0 | 3 | 86425 |  |
| gst_cs_002 | gst | basic | 3.0 | 8.0 | 2 | 24122 |  |
| gst_itc_001 | gst | basic | 4.0 | 8.0 | 3 | 26768 |  |
| gst_itc_002 | gst | intermediate | 4.0 | 10.0 | 2 | 52941 |  |
| gst_itc_003 | gst | intermediate | 2.0 | 10.0 | 2 | 24726 |  |
| gst_itc_004 | gst | basic | 3.0 | 10.0 | 2 | 59044 |  |
| gst_tos_001 | gst | basic | 4.0 | 9.0 | 1 | 26082 |  |
| gst_tos_002 | gst | basic | 5.0 | 9.0 | 2 | 29274 |  |
| gst_inv_001 | gst | basic | 6.0 | 8.0 | 3 | 23455 |  |
| gst_inv_002 | gst | intermediate | 2.0 | 9.0 | 1 | 29227 |  |
| gst_rcm_001 | gst | basic | 6.0 | 9.0 | 3 | 25808 |  |
| gst_rcm_002 | gst | intermediate | 6.0 | 9.0 | 3 | 19524 |  |
| gst_ref_001 | gst | intermediate | 8.0 | 9.0 | 3 | 28235 |  |
| gst_ewb_001 | gst | basic | 7.0 | 9.0 | 3 | 23482 |  |
| gst_exp_001 | gst | intermediate | 6.0 | 10.0 | 3 | 25957 |  |
| cus_val_001 | customs | basic | 5.0 | 8.0 | 3 | 24469 |  |
| cus_val_002 | customs | intermediate | 3.0 | 10.0 | 2 | 66806 |  |
| cus_cls_001 | customs | basic | 1.0 | 10.0 | 1 | 30091 |  |
| cus_wh_001 | customs | intermediate | 2.0 | 10.0 | 1 | 23366 |  |
| cus_db_001 | customs | basic | 5.0 | 9.0 | 2 | 26113 |  |
| cus_db_002 | customs | intermediate | 2.0 | 8.0 | 2 | 63882 |  |
| cus_svb_001 | customs | intermediate | 6.0 | 8.0 | 3 | 22723 |  |
| cus_ar_001 | customs | basic | 5.0 | 11.0 | 1 | 59573 |  |
| cus_cls_002 | customs | complex | 2.0 | 11.0 | 1 | 53737 |  |
| cus_ig_001 | customs | basic | 6.0 | 11.0 | 3 | 58789 |  |
| exc_val_001 | central_excise | basic | 6.0 | 9.0 | 3 | 27547 |  |
| exc_val_002 | central_excise | intermediate | 5.0 | 9.0 | 3 | 25766 |  |
| exc_cen_001 | central_excise | basic | 7.0 | 9.0 | 3 | 29545 |  |
| exc_cen_002 | central_excise | intermediate | 8.0 | 9.0 | 3 | 26232 |  |
| exc_man_001 | central_excise | basic | 3.0 | 9.0 | 2 | 26380 |  |
| exc_man_002 | central_excise | intermediate | 5.0 | 10.0 | 3 | 25323 |  |
| exc_ssi_001 | central_excise | basic | 1.0 | 8.0 | 1 | 24496 |  |
| exc_ssi_002 | central_excise | complex | 3.0 | 8.0 | 2 | 30351 |  |
| st_val_001 | service_tax | basic | 7.0 | 9.0 | 3 | 29770 |  |
| st_nl_001 | service_tax | basic | 6.0 | 11.0 | 3 | 31223 |  |
| st_exp_001 | service_tax | complex | 4.0 | 10.0 | 2 | 25161 |  |
| st_pos_001 | service_tax | basic | 4.0 | 7.0 | 3 | 25934 |  |
| st_rcm_001 | service_tax | intermediate | 3.0 | 10.0 | 1 | 24290 |  |
| st_lev_001 | service_tax | basic | 3.0 | 10.0 | 2 | 24232 |  |
| oth_ap_001 | others | basic | 5.0 | 10.0 | 3 | 32328 |  |
| oth_ap_002 | others | basic | 5.0 | 10.0 | 3 | 23612 |  |
| oth_ap_003 | others | intermediate | 3.0 | 10.0 | 2 | 24622 |  |
| oth_gaar_001 | others | basic | 4.0 | 8.0 | 3 | 26001 |  |
| oth_pen_001 | others | complex | 4.0 | 13.0 | 2 | 30051 |  |
| oth_cls_001 | others | intermediate | 4.0 | 10.0 | 3 | 22458 |  |
| gst_rate_001 | gst | basic | 3.0 | 7.0 | 3 | 39 |  |
| gst_sac_001 | gst | basic | 3.0 | 7.0 | 2 | 22991 |  |
| gst_exempt_001 | gst | basic | 2.0 | 7.0 | 1 | 25853 |  |
| gst_cess_001 | gst | basic | 1.0 | 8.0 | 1 | 45 |  |
| customs_rate_001 | customs | basic | 2.0 | 8.0 | 1 | 24943 |  |
| gst_rate_002 | gst | basic | 5.0 | 7.0 | 2 | 29849 |  |
| gst_hsn_001 | gst | basic | 1.0 | 8.0 | 1 | 30 |  |
| gst_sac_002 | gst | basic | 4.0 | 7.0 | 3 | 24125 |  |
| gst_notif_001 | gst | basic | 1.0 | 6.0 | 1 | 24 |  |
| customs_rate_002 | customs | basic | 2.0 | 8.0 | 1 | 29156 |  |
| gst_rate_003 | gst | basic | 3.0 | 7.0 | 1 | 21645 |  |
| gst_cess_002 | gst | basic | 1.0 | 7.0 | 1 | 42 |  |
| gst_ccy_supply_001 | gst | intermediate | 1.0 | 9.0 | 1 | 51 |  |
| customs_notif_001 | customs | intermediate | 1.0 | 7.0 | 1 | 27546 |  |
| gst_notif_002 | gst | intermediate | 2.0 | 7.0 | 1 | 27635 |  |
| gst_inverted_001 | gst | intermediate | 4.0 | 8.0 | 3 | 29469 |  |
| gst_sac_003 | gst | intermediate | 3.0 | 7.0 | 2 | 24034 |  |
| gst_exempt_002 | gst | intermediate | 1.0 | 8.0 | 1 | 41 |  |
| customs_rate_003 | customs | intermediate | 3.0 | 10.0 | 3 | 27746 |  |
| gst_hsn_002 | gst | intermediate | 4.0 | 8.0 | 2 | 29410 |  |
| gst_rate_004 | gst | intermediate | 4.0 | 6.0 | 3 | 33039 |  |
| gst_inverted_002 | gst | complex | 2.0 | 8.0 | 2 | 28304 |  |
| gst_ccy_supply_002 | gst | complex | 2.0 | 10.0 | 1 | 30159 |  |
| customs_rate_004 | customs | complex | 2.0 | 9.0 | 2 | 26742 |  |
| gst_notif_003 | gst | complex | 3.0 | 8.0 | 2 | 25555 |  |
| gst_refuse_001 | gst | basic | 3.0 | 7.0 | 2 | 23401 |  |
| customs_refuse_002 | customs | basic | 0.0 | 7.0 | 1 | 27470 |  |
| gst_refuse_003 | gst | basic | 2.0 | 8.0 | 1 | 24355 |  |
| others_refuse_004 | others | basic | 0.0 | 7.0 | 1 | 22496 |  |
| others_refuse_005 | others | basic | -1.0 | 8.0 | 1 | 21820 |  |
| gst_refuse_006 | gst | basic | 2.0 | 7.0 | 2 | 25303 |  |
| customs_refuse_007 | customs | basic | 2.0 | 7.0 | 2 | 26692 |  |
| gst_refuse_008 | gst | basic | 0.0 | 7.0 | 2 | 28371 |  |
| others_refuse_009 | others | basic | 2.0 | 8.0 | 1 | 26608 |  |
| others_refuse_010 | others | basic | -1.0 | 7.0 | 1 | 25030 |  |
| gst_complex_001 | gst | complex | 3.0 | 8.0 | 2 | 34003 |  |
| gst_complex_002 | gst | complex | 3.0 | 10.0 | 2 | 25994 |  |
| gst_complex_003 | gst | complex | 3.0 | 11.0 | 2 | 25486 |  |
| gst_complex_004 | gst | complex | 4.0 | 9.0 | 3 | 27847 |  |
| gst_complex_005 | gst | complex | 4.0 | 9.0 | 2 | 29642 |  |
| gst_complex_006 | gst | complex | 3.0 | 9.0 | 2 | 26547 |  |
| gst_complex_007 | gst | complex | 5.0 | 9.0 | 3 | 28406 |  |
| gst_complex_008 | gst | complex | 1.0 | 8.0 | 1 | 23743 |  |
| gst_complex_009 | gst | complex | 2.0 | 11.0 | 2 | 30337 |  |
| gst_complex_010 | gst | complex | 6.0 | 11.0 | 3 | 30155 |  |
| gst_complex_011 | gst | complex | 2.0 | 9.0 | 1 | 30957 |  |
| gst_complex_012 | gst | complex | 2.0 | 10.0 | 2 | 34508 |  |
| gst_complex_013 | gst | complex | 2.0 | 8.0 | 1 | 27686 |  |
| gst_complex_014 | gst | complex | 6.0 | 10.0 | 3 | 27228 |  |
| gst_complex_015 | gst | complex | 3.0 | 9.0 | 3 | 26158 |  |
| customs_complex_016 | customs | complex | 3.0 | 9.0 | 2 | 27500 |  |
| customs_complex_017 | customs | complex | 2.0 | 8.0 | 2 | 28007 |  |
| customs_complex_018 | customs | complex | 2.0 | 9.0 | 2 | 28886 |  |
| customs_complex_019 | customs | complex | 4.0 | 9.0 | 3 | 31551 |  |
| customs_complex_020 | customs | complex | 1.0 | 8.0 | 1 | 25678 |  |
| central_excise_complex_021 | central_excise | complex | 3.0 | 8.0 | 3 | 30001 |  |
| central_excise_complex_022 | central_excise | complex | 2.0 | 9.0 | 1 | 27186 |  |
| service_tax_complex_023 | service_tax | complex | 2.0 | 9.0 | 1 | 33559 |  |
| service_tax_complex_024 | service_tax | complex | 3.0 | 9.0 | 2 | 29753 |  |
| others_complex_025 | others | complex | 3.0 | 9.0 | 2 | 25409 |  |
| st_service_tax_001 | service_tax | basic | 4.0 | 7.0 | 3 | 24619 |  |
| st_service_tax_002 | service_tax | intermediate | 5.0 | 7.0 | 3 | 26114 |  |
| st_service_tax_003 | service_tax | advanced | 3.0 | 7.0 | 3 | 25197 |  |
| st_service_tax_004 | service_tax | basic | 2.0 | 7.0 | 1 | 27032 |  |
| st_service_tax_005 | service_tax | intermediate | -1.0 | 6.0 | 2 | 25781 |  |
| st_service_tax_006 | service_tax | intermediate | 3.0 | 9.0 | 1 | 26940 |  |
| st_service_tax_007 | service_tax | basic | 4.0 | 8.0 | 2 | 26794 |  |
| st_service_tax_008 | service_tax | basic | 2.0 | 8.0 | 2 | 20356 |  |
| st_service_tax_009 | service_tax | advanced | 4.0 | 9.0 | 3 | 24861 |  |
| st_service_tax_010 | service_tax | basic | 5.0 | 8.0 | 3 | 26999 |  |
| st_service_tax_011 | service_tax | intermediate | 3.0 | 7.0 | 3 | 28422 |  |
| st_service_tax_012 | service_tax | basic | 6.0 | 7.0 | 3 | 25148 |  |
| st_transition_001 | gst | intermediate | 6.0 | 9.0 | 3 | 30531 |  |
| st_transition_002 | gst | advanced | 2.0 | 8.0 | 2 | 28245 |  |
| st_transition_003 | gst | intermediate | 4.0 | 8.0 | 1 | 25201 |  |
| st_transition_004 | gst | basic | 4.0 | 9.0 | 3 | 30755 |  |
| st_transition_005 | gst | advanced | 3.0 | 8.0 | 1 | 26335 |  |
| st_refuse_001 | others | basic | 2.0 | 7.0 | 2 | 23223 |  |
| st_refuse_002 | others | basic | 1.0 | 7.0 | 3 | 24778 |  |
| st_refuse_003 | others | intermediate | 4.0 | 7.0 | 2 | 26414 |  |
| cust_customs_valuation_001 | customs | advanced | 2.0 | 8.0 | 2 | 20335 |  |
| cust_customs_valuation_002 | customs | intermediate | 3.0 | 8.0 | 3 | 24837 |  |
| cust_customs_valuation_003 | customs | advanced | 3.0 | 7.0 | 3 | 25306 |  |
| cust_customs_valuation_004 | customs | intermediate | 3.0 | 8.0 | 3 | 24975 |  |
| cust_customs_valuation_005 | customs | basic | 3.0 | 10.0 | 1 | 30631 |  |
| cust_customs_classification_001 | customs | advanced | 6.0 | 8.0 | 3 | 23758 |  |
| cust_customs_classification_002 | customs | intermediate | 4.0 | 7.0 | 3 | 28168 |  |
| cust_customs_classification_003 | customs | basic | 7.0 | 7.0 | 3 | 26946 |  |
| cust_customs_classification_004 | customs | intermediate | 1.0 | 7.0 | 1 | 26159 |  |
| cust_customs_drawback_001 | customs | intermediate | 4.0 | 9.0 | 3 | 25506 |  |
| cust_customs_drawback_002 | customs | advanced | 2.0 | 9.0 | 2 | 25087 |  |
| cust_customs_drawback_003 | customs | intermediate | 1.0 | 9.0 | 1 | 24896 |  |
| cust_customs_drawback_004 | customs | advanced | 5.0 | 9.0 | 3 | 27083 |  |
| cust_customs_warehouse_001 | customs | basic | 6.0 | 9.0 | 3 | 25774 |  |
| cust_customs_warehouse_002 | customs | advanced | 1.0 | 8.0 | 1 | 25892 |  |
| cust_customs_warehouse_003 | customs | intermediate | 2.0 | 9.0 | 1 | 23516 |  |
| cust_customs_adviolation_001 | customs | intermediate | 5.0 | 10.0 | 3 | 26619 |  |
| cust_customs_adviolation_002 | customs | advanced | 6.0 | 8.0 | 3 | 59083 |  |
| cust_customs_exemption_001 | customs | intermediate | 3.0 | 7.0 | 3 | 26752 |  |
| cust_customs_exemption_002 | customs | advanced | 1.0 | 8.0 | 1 | 27998 |  |
| oth_appeals_001 | gst | intermediate | 3.0 | 8.0 | 3 | 30759 |  |
| oth_appeals_002 | gst | intermediate | 1.0 | 8.0 | 1 | 23977 |  |
| oth_appeals_003 | customs | basic | 1.0 | 9.0 | 1 | 26371 |  |
| oth_appeals_004 | customs | intermediate | 2.0 | 8.0 | 1 | 28928 |  |
| oth_appeals_005 | gst | basic | 5.0 | 7.0 | 3 | 24836 |  |
| oth_penalty_001 | gst | intermediate | 3.0 | 7.0 | 3 | 24584 |  |
| oth_penalty_002 | gst | basic | 1.0 | 8.0 | 2 | 25725 |  |
| oth_penalty_003 | gst | advanced | 7.0 | 8.0 | 3 | 26436 |  |
| oth_penalty_004 | customs | intermediate | 3.0 | 7.0 | 2 | 25666 |  |
| oth_advance_ruling_001 | gst | basic | 3.0 | 8.0 | 2 | 25932 |  |
| oth_advance_ruling_002 | gst | intermediate | 2.0 | 7.0 | 2 | 26478 |  |
| oth_advance_ruling_003 | gst | advanced | 6.0 | 9.0 | 3 | 24374 |  |
| oth_advance_ruling_004 | gst | basic | 6.0 | 8.0 | 3 | 26801 |  |
| oth_anti_evasion_001 | gst | intermediate | 5.0 | 7.0 | 3 | 24770 |  |
| oth_anti_evasion_002 | gst | intermediate | 5.0 | 7.0 | 3 | 25065 |  |
| oth_anti_evasion_003 | gst | basic | 4.0 | 8.0 | 3 | 25110 |  |
| oth_offences_001 | gst | intermediate | 4.0 | 10.0 | 3 | 28371 |  |
| oth_offences_002 | gst | advanced | 3.0 | 9.0 | 3 | 26059 |  |
| oth_interest_001 | gst | intermediate | 2.0 | 8.0 | 2 | 27946 |  |
| oth_interest_002 | gst | basic | 6.0 | 8.0 | 3 | 26219 |  |