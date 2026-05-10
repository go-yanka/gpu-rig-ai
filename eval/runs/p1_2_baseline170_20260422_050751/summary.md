# CBIC RAG Eval Run

Run dir: `runs\p1_2_baseline170_20260422_050751`
Score: **582.0 / 1439.0** (40.44%)
Items: 170  |  Errors: 0
Latency ms — median 26267.5  p95 33057  max 57411

## By category

| Category | N | Points | Max | % | Errors |
|---|---:|---:|---:|---:|---:|
| central_excise | 10 | 42.0 | 88.0 | 47.73 | 0 |
| customs | 45 | 133.0 | 384.0 | 34.64 | 0 |
| gst | 81 | 302.0 | 681.0 | 44.35 | 0 |
| others | 14 | 34.0 | 121.0 | 28.1 | 0 |
| service_tax | 20 | 71.0 | 165.0 | 43.03 | 0 |

## Per-item

| id | cat | diff | pts | max | judge | lat_ms | err |
|---|---|---|---:|---:|---:|---:|---|
| gst_pos_001 | gst | basic | 4.0 | 9.0 | 1 | 25109 |  |
| gst_pos_002 | gst | intermediate | 4.0 | 9.0 | 1 | 26920 |  |
| gst_pos_003 | gst | basic | 4.0 | 8.0 | 3 | 23375 |  |
| gst_pos_004 | gst | intermediate | 5.0 | 8.0 | 3 | 24472 |  |
| gst_pos_005 | gst | complex | 4.0 | 13.0 | 2 | 28941 |  |
| gst_cs_001 | gst | intermediate | 6.0 | 10.0 | 3 | 26569 |  |
| gst_cs_002 | gst | basic | 3.0 | 8.0 | 2 | 23983 |  |
| gst_itc_001 | gst | basic | 4.0 | 8.0 | 3 | 24870 |  |
| gst_itc_002 | gst | intermediate | 4.0 | 10.0 | 2 | 26713 |  |
| gst_itc_003 | gst | intermediate | 2.0 | 10.0 | 2 | 25322 |  |
| gst_itc_004 | gst | basic | 3.0 | 10.0 | 2 | 23793 |  |
| gst_tos_001 | gst | basic | 4.0 | 9.0 | 1 | 52216 |  |
| gst_tos_002 | gst | basic | 5.0 | 9.0 | 2 | 25326 |  |
| gst_inv_001 | gst | basic | 6.0 | 8.0 | 3 | 53156 |  |
| gst_inv_002 | gst | intermediate | 2.0 | 9.0 | 1 | 29109 |  |
| gst_rcm_001 | gst | basic | 6.0 | 9.0 | 3 | 25685 |  |
| gst_rcm_002 | gst | intermediate | 6.0 | 9.0 | 3 | 49400 |  |
| gst_ref_001 | gst | intermediate | 8.0 | 9.0 | 3 | 28070 |  |
| gst_ewb_001 | gst | basic | 7.0 | 9.0 | 3 | 24382 |  |
| gst_exp_001 | gst | intermediate | 6.0 | 10.0 | 3 | 25883 |  |
| cus_val_001 | customs | basic | 5.0 | 8.0 | 3 | 24318 |  |
| cus_val_002 | customs | intermediate | 3.0 | 10.0 | 2 | 25119 |  |
| cus_cls_001 | customs | basic | 1.0 | 10.0 | 1 | 29946 |  |
| cus_wh_001 | customs | intermediate | 2.0 | 10.0 | 1 | 23407 |  |
| cus_db_001 | customs | basic | 5.0 | 9.0 | 2 | 26192 |  |
| cus_db_002 | customs | intermediate | 2.0 | 8.0 | 2 | 25710 |  |
| cus_svb_001 | customs | intermediate | 6.0 | 8.0 | 3 | 22666 |  |
| cus_ar_001 | customs | basic | 5.0 | 11.0 | 1 | 23731 |  |
| cus_cls_002 | customs | complex | 2.0 | 11.0 | 1 | 24076 |  |
| cus_ig_001 | customs | basic | 4.0 | 11.0 | 2 | 28077 |  |
| exc_val_001 | central_excise | basic | 5.0 | 9.0 | 2 | 26271 |  |
| exc_val_002 | central_excise | intermediate | 5.0 | 9.0 | 3 | 25732 |  |
| exc_cen_001 | central_excise | basic | 7.0 | 9.0 | 3 | 57411 |  |
| exc_cen_002 | central_excise | intermediate | 8.0 | 9.0 | 3 | 26151 |  |
| exc_man_001 | central_excise | basic | 3.0 | 9.0 | 2 | 26254 |  |
| exc_man_002 | central_excise | intermediate | 5.0 | 10.0 | 3 | 25247 |  |
| exc_ssi_001 | central_excise | basic | 1.0 | 8.0 | 1 | 23576 |  |
| exc_ssi_002 | central_excise | complex | 3.0 | 8.0 | 2 | 30384 |  |
| st_val_001 | service_tax | basic | 7.0 | 9.0 | 3 | 29725 |  |
| st_nl_001 | service_tax | basic | 5.0 | 11.0 | 2 | 29012 |  |
| st_exp_001 | service_tax | complex | 4.0 | 10.0 | 2 | 25149 |  |
| st_pos_001 | service_tax | basic | 4.0 | 7.0 | 3 | 25955 |  |
| st_rcm_001 | service_tax | intermediate | 3.0 | 10.0 | 1 | 24412 |  |
| st_lev_001 | service_tax | basic | 3.0 | 10.0 | 2 | 24132 |  |
| oth_ap_001 | others | basic | 5.0 | 10.0 | 3 | 32259 |  |
| oth_ap_002 | others | basic | 5.0 | 10.0 | 3 | 23500 |  |
| oth_ap_003 | others | intermediate | 3.0 | 10.0 | 2 | 50328 |  |
| oth_gaar_001 | others | basic | 4.0 | 8.0 | 3 | 25854 |  |
| oth_pen_001 | others | complex | 4.0 | 13.0 | 2 | 29940 |  |
| oth_cls_001 | others | intermediate | 4.0 | 10.0 | 3 | 22470 |  |
| gst_rate_001 | gst | basic | 3.0 | 7.0 | 1 | 20926 |  |
| gst_sac_001 | gst | basic | 3.0 | 7.0 | 2 | 23030 |  |
| gst_exempt_001 | gst | basic | 2.0 | 7.0 | 1 | 24293 |  |
| gst_cess_001 | gst | basic | 6.0 | 8.0 | 3 | 24066 |  |
| customs_rate_001 | customs | basic | 2.0 | 8.0 | 1 | 25484 |  |
| gst_rate_002 | gst | basic | 5.0 | 7.0 | 2 | 29948 |  |
| gst_hsn_001 | gst | basic | 1.0 | 8.0 | 1 | 26068 |  |
| gst_sac_002 | gst | basic | 4.0 | 7.0 | 3 | 24605 |  |
| gst_notif_001 | gst | basic | 2.0 | 6.0 | 2 | 26275 |  |
| customs_rate_002 | customs | basic | 2.0 | 8.0 | 1 | 31042 |  |
| gst_rate_003 | gst | basic | 3.0 | 7.0 | 1 | 21559 |  |
| gst_cess_002 | gst | basic | 4.0 | 7.0 | 3 | 26378 |  |
| gst_ccy_supply_001 | gst | intermediate | 5.0 | 9.0 | 3 | 28540 |  |
| customs_notif_001 | customs | intermediate | 1.0 | 7.0 | 1 | 28739 |  |
| gst_notif_002 | gst | intermediate | 3.0 | 7.0 | 1 | 26781 |  |
| gst_inverted_001 | gst | intermediate | 4.0 | 8.0 | 3 | 29487 |  |
| gst_sac_003 | gst | intermediate | 3.0 | 7.0 | 2 | 23975 |  |
| gst_exempt_002 | gst | intermediate | 4.0 | 8.0 | 2 | 27001 |  |
| customs_rate_003 | customs | intermediate | 1.0 | 10.0 | 2 | 27533 |  |
| gst_hsn_002 | gst | intermediate | 4.0 | 8.0 | 2 | 26812 |  |
| gst_rate_004 | gst | intermediate | 4.0 | 6.0 | 3 | 33057 |  |
| gst_inverted_002 | gst | complex | 2.0 | 8.0 | 2 | 28338 |  |
| gst_ccy_supply_002 | gst | complex | 2.0 | 10.0 | 1 | 30180 |  |
| customs_rate_004 | customs | complex | 2.0 | 9.0 | 2 | 26866 |  |
| gst_notif_003 | gst | complex | 3.0 | 8.0 | 2 | 25663 |  |
| gst_refuse_001 | gst | basic | 3.0 | 7.0 | 2 | 23329 |  |
| customs_refuse_002 | customs | basic | 0.0 | 7.0 | 1 | 27703 |  |
| gst_refuse_003 | gst | basic | 2.0 | 8.0 | 1 | 24099 |  |
| others_refuse_004 | others | basic | 0.0 | 7.0 | 1 | 22505 |  |
| others_refuse_005 | others | basic | -1.0 | 8.0 | 1 | 21960 |  |
| gst_refuse_006 | gst | basic | 2.0 | 7.0 | 2 | 25358 |  |
| customs_refuse_007 | customs | basic | 2.0 | 7.0 | 2 | 26730 |  |
| gst_refuse_008 | gst | basic | 0.0 | 7.0 | 2 | 28463 |  |
| others_refuse_009 | others | basic | 1.0 | 8.0 | 1 | 26562 |  |
| others_refuse_010 | others | basic | -1.0 | 7.0 | 1 | 25117 |  |
| gst_complex_001 | gst | complex | 3.0 | 8.0 | 2 | 34087 |  |
| gst_complex_002 | gst | complex | 3.0 | 10.0 | 2 | 26662 |  |
| gst_complex_003 | gst | complex | 3.0 | 11.0 | 2 | 25521 |  |
| gst_complex_004 | gst | complex | 4.0 | 9.0 | 3 | 27915 |  |
| gst_complex_005 | gst | complex | 4.0 | 9.0 | 2 | 29619 |  |
| gst_complex_006 | gst | complex | 3.0 | 9.0 | 2 | 26763 |  |
| gst_complex_007 | gst | complex | 5.0 | 9.0 | 3 | 28355 |  |
| gst_complex_008 | gst | complex | 1.0 | 8.0 | 1 | 23847 |  |
| gst_complex_009 | gst | complex | 2.0 | 11.0 | 2 | 30404 |  |
| gst_complex_010 | gst | complex | 6.0 | 11.0 | 3 | 30220 |  |
| gst_complex_011 | gst | complex | 2.0 | 9.0 | 1 | 30928 |  |
| gst_complex_012 | gst | complex | 3.0 | 10.0 | 2 | 34772 |  |
| gst_complex_013 | gst | complex | 2.0 | 8.0 | 1 | 27609 |  |
| gst_complex_014 | gst | complex | 6.0 | 10.0 | 3 | 27154 |  |
| gst_complex_015 | gst | complex | 3.0 | 9.0 | 3 | 26102 |  |
| customs_complex_016 | customs | complex | 3.0 | 9.0 | 2 | 27385 |  |
| customs_complex_017 | customs | complex | 2.0 | 8.0 | 2 | 28064 |  |
| customs_complex_018 | customs | complex | 2.0 | 9.0 | 2 | 28947 |  |
| customs_complex_019 | customs | complex | 4.0 | 9.0 | 3 | 31597 |  |
| customs_complex_020 | customs | complex | 2.0 | 8.0 | 2 | 25759 |  |
| central_excise_complex_021 | central_excise | complex | 3.0 | 8.0 | 3 | 30056 |  |
| central_excise_complex_022 | central_excise | complex | 2.0 | 9.0 | 1 | 27194 |  |
| service_tax_complex_023 | service_tax | complex | 2.0 | 9.0 | 1 | 33621 |  |
| service_tax_complex_024 | service_tax | complex | 3.0 | 9.0 | 2 | 29921 |  |
| others_complex_025 | others | complex | 3.0 | 9.0 | 2 | 25466 |  |
| st_service_tax_001 | service_tax | basic | 4.0 | 7.0 | 3 | 24752 |  |
| st_service_tax_002 | service_tax | intermediate | 5.0 | 7.0 | 3 | 26076 |  |
| st_service_tax_003 | service_tax | advanced | 3.0 | 7.0 | 3 | 25312 |  |
| st_service_tax_004 | service_tax | basic | 2.0 | 7.0 | 1 | 27058 |  |
| st_service_tax_005 | service_tax | intermediate | -1.0 | 6.0 | 2 | 25861 |  |
| st_service_tax_006 | service_tax | intermediate | 3.0 | 9.0 | 1 | 26996 |  |
| st_service_tax_007 | service_tax | basic | 4.0 | 8.0 | 2 | 26882 |  |
| st_service_tax_008 | service_tax | basic | 2.0 | 8.0 | 2 | 20430 |  |
| st_service_tax_009 | service_tax | advanced | 4.0 | 9.0 | 3 | 24973 |  |
| st_service_tax_010 | service_tax | basic | 5.0 | 8.0 | 3 | 28278 |  |
| st_service_tax_011 | service_tax | intermediate | 3.0 | 7.0 | 3 | 28346 |  |
| st_service_tax_012 | service_tax | basic | 6.0 | 7.0 | 3 | 25209 |  |
| st_transition_001 | gst | intermediate | 6.0 | 9.0 | 3 | 30603 |  |
| st_transition_002 | gst | advanced | 2.0 | 8.0 | 2 | 28270 |  |
| st_transition_003 | gst | intermediate | 4.0 | 8.0 | 1 | 25207 |  |
| st_transition_004 | gst | basic | 4.0 | 9.0 | 3 | 30706 |  |
| st_transition_005 | gst | advanced | 3.0 | 8.0 | 1 | 26284 |  |
| st_refuse_001 | others | basic | 2.0 | 7.0 | 2 | 23190 |  |
| st_refuse_002 | others | basic | 1.0 | 7.0 | 3 | 24652 |  |
| st_refuse_003 | others | intermediate | 4.0 | 7.0 | 2 | 26264 |  |
| cust_customs_valuation_001 | customs | advanced | 2.0 | 8.0 | 2 | 20226 |  |
| cust_customs_valuation_002 | customs | intermediate | 3.0 | 8.0 | 3 | 24670 |  |
| cust_customs_valuation_003 | customs | advanced | 3.0 | 7.0 | 3 | 25364 |  |
| cust_customs_valuation_004 | customs | intermediate | 3.0 | 8.0 | 3 | 25156 |  |
| cust_customs_valuation_005 | customs | basic | 3.0 | 10.0 | 1 | 30618 |  |
| cust_customs_classification_001 | customs | advanced | 6.0 | 8.0 | 3 | 23760 |  |
| cust_customs_classification_002 | customs | intermediate | 4.0 | 7.0 | 3 | 28154 |  |
| cust_customs_classification_003 | customs | basic | 7.0 | 7.0 | 3 | 27122 |  |
| cust_customs_classification_004 | customs | intermediate | 1.0 | 7.0 | 1 | 26205 |  |
| cust_customs_drawback_001 | customs | intermediate | 5.0 | 9.0 | 3 | 26778 |  |
| cust_customs_drawback_002 | customs | advanced | 2.0 | 9.0 | 2 | 25129 |  |
| cust_customs_drawback_003 | customs | intermediate | 1.0 | 9.0 | 1 | 24944 |  |
| cust_customs_drawback_004 | customs | advanced | 5.0 | 9.0 | 3 | 27132 |  |
| cust_customs_warehouse_001 | customs | basic | 6.0 | 9.0 | 3 | 25806 |  |
| cust_customs_warehouse_002 | customs | advanced | 1.0 | 8.0 | 1 | 25869 |  |
| cust_customs_warehouse_003 | customs | intermediate | 2.0 | 9.0 | 1 | 23847 |  |
| cust_customs_adviolation_001 | customs | intermediate | 5.0 | 10.0 | 3 | 26663 |  |
| cust_customs_adviolation_002 | customs | advanced | 6.0 | 8.0 | 3 | 26095 |  |
| cust_customs_exemption_001 | customs | intermediate | 3.0 | 7.0 | 3 | 26923 |  |
| cust_customs_exemption_002 | customs | advanced | 1.0 | 8.0 | 1 | 28029 |  |
| oth_appeals_001 | gst | intermediate | 3.0 | 8.0 | 3 | 30857 |  |
| oth_appeals_002 | gst | intermediate | 1.0 | 8.0 | 1 | 24017 |  |
| oth_appeals_003 | customs | basic | 1.0 | 9.0 | 1 | 26545 |  |
| oth_appeals_004 | customs | intermediate | 2.0 | 8.0 | 1 | 29032 |  |
| oth_appeals_005 | gst | basic | 5.0 | 7.0 | 3 | 24973 |  |
| oth_penalty_001 | gst | intermediate | 3.0 | 7.0 | 3 | 24586 |  |
| oth_penalty_002 | gst | basic | 1.0 | 8.0 | 2 | 25777 |  |
| oth_penalty_003 | gst | advanced | 7.0 | 8.0 | 3 | 26648 |  |
| oth_penalty_004 | customs | intermediate | 3.0 | 7.0 | 2 | 25813 |  |
| oth_advance_ruling_001 | gst | basic | 3.0 | 8.0 | 2 | 26333 |  |
| oth_advance_ruling_002 | gst | intermediate | 2.0 | 7.0 | 2 | 26541 |  |
| oth_advance_ruling_003 | gst | advanced | 6.0 | 9.0 | 3 | 24487 |  |
| oth_advance_ruling_004 | gst | basic | 6.0 | 8.0 | 3 | 26922 |  |
| oth_anti_evasion_001 | gst | intermediate | 5.0 | 7.0 | 3 | 24855 |  |
| oth_anti_evasion_002 | gst | intermediate | 5.0 | 7.0 | 3 | 25145 |  |
| oth_anti_evasion_003 | gst | basic | 4.0 | 8.0 | 3 | 25209 |  |
| oth_offences_001 | gst | intermediate | 4.0 | 10.0 | 3 | 28365 |  |
| oth_offences_002 | gst | advanced | 3.0 | 9.0 | 3 | 25939 |  |
| oth_interest_001 | gst | intermediate | 2.0 | 8.0 | 2 | 27904 |  |
| oth_interest_002 | gst | basic | 6.0 | 8.0 | 3 | 26348 |  |