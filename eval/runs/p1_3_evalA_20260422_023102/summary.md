# CBIC RAG Eval Run

Run dir: `runs\p1_3_evalA_20260422_023102`
Score: **389.0 / 1439.0** (27.03%)
Items: 170  |  Errors: 0
Latency ms — median 31329.5  p95 64100  max 78297

## By category

| Category | N | Points | Max | % | Errors |
|---|---:|---:|---:|---:|---:|
| central_excise | 10 | 28.0 | 88.0 | 31.82 | 0 |
| customs | 45 | 75.0 | 384.0 | 19.53 | 0 |
| gst | 81 | 214.0 | 681.0 | 31.42 | 0 |
| others | 14 | 26.0 | 121.0 | 21.49 | 0 |
| service_tax | 20 | 46.0 | 165.0 | 27.88 | 0 |

## Per-item

| id | cat | diff | pts | max | judge | lat_ms | err |
|---|---|---|---:|---:|---:|---:|---|
| gst_pos_001 | gst | basic | 0.0 | 9.0 | 0 | 30252 |  |
| gst_pos_002 | gst | intermediate | 4.0 | 9.0 | 2 | 43969 |  |
| gst_pos_003 | gst | basic | 0.0 | 8.0 | 0 | 26032 |  |
| gst_pos_004 | gst | intermediate | 0.0 | 8.0 | 0 | 23966 |  |
| gst_pos_005 | gst | complex | 3.0 | 13.0 | 1 | 45739 |  |
| gst_cs_001 | gst | intermediate | 6.0 | 10.0 | 2 | 31348 |  |
| gst_cs_002 | gst | basic | 4.0 | 8.0 | 2 | 31096 |  |
| gst_itc_001 | gst | basic | 3.0 | 8.0 | 1 | 52544 |  |
| gst_itc_002 | gst | intermediate | 3.0 | 10.0 | 1 | 44914 |  |
| gst_itc_003 | gst | intermediate | 0.0 | 10.0 | 0 | 23092 |  |
| gst_itc_004 | gst | basic | 3.0 | 10.0 | 2 | 43809 |  |
| gst_tos_001 | gst | basic | 4.0 | 9.0 | 2 | 27217 |  |
| gst_tos_002 | gst | basic | 3.0 | 9.0 | 1 | 26755 |  |
| gst_inv_001 | gst | basic | 0.0 | 8.0 | 0 | 22930 |  |
| gst_inv_002 | gst | intermediate | 0.0 | 9.0 | 0 | 18479 |  |
| gst_rcm_001 | gst | basic | 7.0 | 9.0 | 3 | 27370 |  |
| gst_rcm_002 | gst | intermediate | 0.0 | 9.0 | 0 | 20654 |  |
| gst_ref_001 | gst | intermediate | 6.0 | 9.0 | 2 | 35656 |  |
| gst_ewb_001 | gst | basic | 5.0 | 9.0 | 2 | 27360 |  |
| gst_exp_001 | gst | intermediate | 7.0 | 10.0 | 3 | 40328 |  |
| cus_val_001 | customs | basic | 0.0 | 8.0 | 0 | 21772 |  |
| cus_val_002 | customs | intermediate | 0.0 | 10.0 | 0 | 17758 |  |
| cus_cls_001 | customs | basic | 2.0 | 10.0 | 1 | 28983 |  |
| cus_wh_001 | customs | intermediate | 3.0 | 10.0 | 1 | 30167 |  |
| cus_db_001 | customs | basic | 5.0 | 9.0 | 2 | 28586 |  |
| cus_db_002 | customs | intermediate | 0.0 | 8.0 | 0 | 24903 |  |
| cus_svb_001 | customs | intermediate | 5.0 | 8.0 | 2 | 25401 |  |
| cus_ar_001 | customs | basic | 6.0 | 11.0 | 2 | 35030 |  |
| cus_cls_002 | customs | complex | 0.0 | 11.0 | 0 | 20504 |  |
| cus_ig_001 | customs | basic | 3.0 | 11.0 | 2 | 30684 |  |
| exc_val_001 | central_excise | basic | 0.0 | 9.0 | 0 | 23042 |  |
| exc_val_002 | central_excise | intermediate | 4.0 | 9.0 | 2 | 25605 |  |
| exc_cen_001 | central_excise | basic | 0.0 | 9.0 | 0 | 24830 |  |
| exc_cen_002 | central_excise | intermediate | 7.0 | 9.0 | 2 | 32401 |  |
| exc_man_001 | central_excise | basic | 2.0 | 9.0 | 1 | 23584 |  |
| exc_man_002 | central_excise | intermediate | 5.0 | 10.0 | 2 | 24863 |  |
| exc_ssi_001 | central_excise | basic | 0.0 | 8.0 | 0 | 15600 |  |
| exc_ssi_002 | central_excise | complex | 4.0 | 8.0 | 2 | 30897 |  |
| st_val_001 | service_tax | basic | 0.0 | 9.0 | 0 | 27429 |  |
| st_nl_001 | service_tax | basic | 4.0 | 11.0 | 1 | 30986 |  |
| st_exp_001 | service_tax | complex | 0.0 | 10.0 | 0 | 26946 |  |
| st_pos_001 | service_tax | basic | 4.0 | 7.0 | 2 | 30640 |  |
| st_rcm_001 | service_tax | intermediate | 4.0 | 10.0 | 1 | 26244 |  |
| st_lev_001 | service_tax | basic | 4.0 | 10.0 | 2 | 28956 |  |
| oth_ap_001 | others | basic | 3.0 | 10.0 | 2 | 34333 |  |
| oth_ap_002 | others | basic | 6.0 | 10.0 | 3 | 26168 |  |
| oth_ap_003 | others | intermediate | 4.0 | 10.0 | 2 | 26963 |  |
| oth_gaar_001 | others | basic | 1.0 | 8.0 | 1 | 17796 |  |
| oth_pen_001 | others | complex | 3.0 | 13.0 | 1 | 67139 |  |
| oth_cls_001 | others | intermediate | 2.0 | 10.0 | 1 | 29865 |  |
| gst_rate_001 | gst | basic | 0.0 | 7.0 | 0 | 12871 |  |
| gst_sac_001 | gst | basic | 3.0 | 7.0 | 1 | 34540 |  |
| gst_exempt_001 | gst | basic | 0.0 | 7.0 | 0 | 23739 |  |
| gst_cess_001 | gst | basic | 0.0 | 8.0 | 0 | 22403 |  |
| customs_rate_001 | customs | basic | 3.0 | 8.0 | 1 | 31342 |  |
| gst_rate_002 | gst | basic | 6.0 | 7.0 | 3 | 32232 |  |
| gst_hsn_001 | gst | basic | 4.0 | 8.0 | 3 | 24214 |  |
| gst_sac_002 | gst | basic | 4.0 | 7.0 | 3 | 25811 |  |
| gst_notif_001 | gst | basic | 1.0 | 6.0 | 1 | 17627 |  |
| customs_rate_002 | customs | basic | 0.0 | 8.0 | 0 | 25182 |  |
| gst_rate_003 | gst | basic | 0.0 | 7.0 | 0 | 19911 |  |
| gst_cess_002 | gst | basic | 0.0 | 7.0 | 0 | 23613 |  |
| gst_ccy_supply_001 | gst | intermediate | 5.0 | 9.0 | 2 | 34371 |  |
| customs_notif_001 | customs | intermediate | 2.0 | 7.0 | 1 | 32411 |  |
| gst_notif_002 | gst | intermediate | 3.0 | 7.0 | 1 | 31108 |  |
| gst_inverted_001 | gst | intermediate | 5.0 | 8.0 | 3 | 45686 |  |
| gst_sac_003 | gst | intermediate | 0.0 | 7.0 | 0 | 23697 |  |
| gst_exempt_002 | gst | intermediate | 1.0 | 8.0 | 0 | 33817 |  |
| customs_rate_003 | customs | intermediate | 3.0 | 10.0 | 2 | 30207 |  |
| gst_hsn_002 | gst | intermediate | 3.0 | 8.0 | 1 | 35775 |  |
| gst_rate_004 | gst | intermediate | 4.0 | 6.0 | 3 | 43539 |  |
| gst_inverted_002 | gst | complex | 2.0 | 8.0 | 1 | 78297 |  |
| gst_ccy_supply_002 | gst | complex | 3.0 | 10.0 | 2 | 42103 |  |
| customs_rate_004 | customs | complex | 2.0 | 9.0 | 1 | 55192 |  |
| gst_notif_003 | gst | complex | 3.0 | 8.0 | 1 | 53614 |  |
| gst_refuse_001 | gst | basic | 0.0 | 7.0 | 0 | 26907 |  |
| customs_refuse_002 | customs | basic | 0.0 | 7.0 | 0 | 18492 |  |
| gst_refuse_003 | gst | basic | 2.0 | 8.0 | 1 | 32817 |  |
| others_refuse_004 | others | basic | 0.0 | 7.0 | 1 | 31317 |  |
| others_refuse_005 | others | basic | -1.0 | 8.0 | 1 | 25528 |  |
| gst_refuse_006 | gst | basic | 0.0 | 7.0 | 0 | 24584 |  |
| customs_refuse_007 | customs | basic | 0.0 | 7.0 | 0 | 24789 |  |
| gst_refuse_008 | gst | basic | 0.0 | 7.0 | 1 | 26178 |  |
| others_refuse_009 | others | basic | 1.0 | 8.0 | 1 | 23103 |  |
| others_refuse_010 | others | basic | 0.0 | 7.0 | 0 | 20166 |  |
| gst_complex_001 | gst | complex | 2.0 | 8.0 | 1 | 59682 |  |
| gst_complex_002 | gst | complex | 2.0 | 10.0 | 1 | 38779 |  |
| gst_complex_003 | gst | complex | 4.0 | 11.0 | 2 | 34941 |  |
| gst_complex_004 | gst | complex | 4.0 | 9.0 | 2 | 68976 |  |
| gst_complex_005 | gst | complex | 5.0 | 9.0 | 2 | 74931 |  |
| gst_complex_006 | gst | complex | 3.0 | 9.0 | 1 | 50556 |  |
| gst_complex_007 | gst | complex | 2.0 | 9.0 | 1 | 53229 |  |
| gst_complex_008 | gst | complex | 2.0 | 8.0 | 1 | 67779 |  |
| gst_complex_009 | gst | complex | 3.0 | 11.0 | 2 | 64100 |  |
| gst_complex_010 | gst | complex | 4.0 | 11.0 | 1 | 50340 |  |
| gst_complex_011 | gst | complex | 3.0 | 9.0 | 1 | 43145 |  |
| gst_complex_012 | gst | complex | 2.0 | 10.0 | 1 | 49038 |  |
| gst_complex_013 | gst | complex | 2.0 | 8.0 | 1 | 64758 |  |
| gst_complex_014 | gst | complex | 6.0 | 10.0 | 3 | 48327 |  |
| gst_complex_015 | gst | complex | 3.0 | 9.0 | 2 | 42308 |  |
| customs_complex_016 | customs | complex | 0.0 | 9.0 | 0 | 36785 |  |
| customs_complex_017 | customs | complex | 3.0 | 8.0 | 2 | 50338 |  |
| customs_complex_018 | customs | complex | 4.0 | 9.0 | 2 | 51220 |  |
| customs_complex_019 | customs | complex | 2.0 | 9.0 | 1 | 46124 |  |
| customs_complex_020 | customs | complex | 0.0 | 8.0 | 0 | 50976 |  |
| central_excise_complex_021 | central_excise | complex | 3.0 | 8.0 | 2 | 67660 |  |
| central_excise_complex_022 | central_excise | complex | 3.0 | 9.0 | 2 | 32662 |  |
| service_tax_complex_023 | service_tax | complex | 3.0 | 9.0 | 1 | 54543 |  |
| service_tax_complex_024 | service_tax | complex | 4.0 | 9.0 | 2 | 68455 |  |
| others_complex_025 | others | complex | 2.0 | 9.0 | 1 | 44684 |  |
| st_service_tax_001 | service_tax | basic | 3.0 | 7.0 | 2 | 33590 |  |
| st_service_tax_002 | service_tax | intermediate | 0.0 | 7.0 | 0 | 26930 |  |
| st_service_tax_003 | service_tax | advanced | 4.0 | 7.0 | 3 | 33562 |  |
| st_service_tax_004 | service_tax | basic | 1.0 | 7.0 | 1 | 35404 |  |
| st_service_tax_005 | service_tax | intermediate | 0.0 | 6.0 | 0 | 26515 |  |
| st_service_tax_006 | service_tax | intermediate | 3.0 | 9.0 | 1 | 29135 |  |
| st_service_tax_007 | service_tax | basic | 0.0 | 8.0 | 0 | 23539 |  |
| st_service_tax_008 | service_tax | basic | 3.0 | 8.0 | 2 | 23945 |  |
| st_service_tax_009 | service_tax | advanced | 2.0 | 9.0 | 1 | 32770 |  |
| st_service_tax_010 | service_tax | basic | 3.0 | 8.0 | 2 | 35441 |  |
| st_service_tax_011 | service_tax | intermediate | 0.0 | 7.0 | 0 | 34674 |  |
| st_service_tax_012 | service_tax | basic | 4.0 | 7.0 | 2 | 28007 |  |
| st_transition_001 | gst | intermediate | 6.0 | 9.0 | 3 | 50626 |  |
| st_transition_002 | gst | advanced | 0.0 | 8.0 | 0 | 38290 |  |
| st_transition_003 | gst | intermediate | 3.0 | 8.0 | 1 | 42534 |  |
| st_transition_004 | gst | basic | 0.0 | 9.0 | 0 | 19771 |  |
| st_transition_005 | gst | advanced | 2.0 | 8.0 | 1 | 36262 |  |
| st_refuse_001 | others | basic | 2.0 | 7.0 | 1 | 26672 |  |
| st_refuse_002 | others | basic | 1.0 | 7.0 | 0 | 16945 |  |
| st_refuse_003 | others | intermediate | 2.0 | 7.0 | 1 | 32617 |  |
| cust_customs_valuation_001 | customs | advanced | 1.0 | 8.0 | 1 | 52258 |  |
| cust_customs_valuation_002 | customs | intermediate | 0.0 | 8.0 | 0 | 31065 |  |
| cust_customs_valuation_003 | customs | advanced | 2.0 | 7.0 | 2 | 42605 |  |
| cust_customs_valuation_004 | customs | intermediate | 4.0 | 8.0 | 2 | 42980 |  |
| cust_customs_valuation_005 | customs | basic | 1.0 | 10.0 | 1 | 27288 |  |
| cust_customs_classification_001 | customs | advanced | 0.0 | 8.0 | 0 | 26406 |  |
| cust_customs_classification_002 | customs | intermediate | 0.0 | 7.0 | 0 | 30708 |  |
| cust_customs_classification_003 | customs | basic | 7.0 | 7.0 | 3 | 36807 |  |
| cust_customs_classification_004 | customs | intermediate | 0.0 | 7.0 | 0 | 23265 |  |
| cust_customs_drawback_001 | customs | intermediate | 4.0 | 9.0 | 2 | 30954 |  |
| cust_customs_drawback_002 | customs | advanced | 3.0 | 9.0 | 2 | 52126 |  |
| cust_customs_drawback_003 | customs | intermediate | 1.0 | 9.0 | 1 | 51196 |  |
| cust_customs_drawback_004 | customs | advanced | 0.0 | 9.0 | 0 | 30030 |  |
| cust_customs_warehouse_001 | customs | basic | 0.0 | 9.0 | 0 | 32566 |  |
| cust_customs_warehouse_002 | customs | advanced | 1.0 | 8.0 | 1 | 31764 |  |
| cust_customs_warehouse_003 | customs | intermediate | 0.0 | 9.0 | 0 | 27224 |  |
| cust_customs_adviolation_001 | customs | intermediate | 0.0 | 10.0 | 0 | 33366 |  |
| cust_customs_adviolation_002 | customs | advanced | 0.0 | 8.0 | 0 | 23623 |  |
| cust_customs_exemption_001 | customs | intermediate | 2.0 | 7.0 | 2 | 47538 |  |
| cust_customs_exemption_002 | customs | advanced | 2.0 | 8.0 | 1 | 46079 |  |
| oth_appeals_001 | gst | intermediate | 0.0 | 8.0 | 0 | 27751 |  |
| oth_appeals_002 | gst | intermediate | 0.0 | 8.0 | 0 | 26277 |  |
| oth_appeals_003 | customs | basic | 4.0 | 9.0 | 2 | 38592 |  |
| oth_appeals_004 | customs | intermediate | 0.0 | 8.0 | 0 | 18949 |  |
| oth_appeals_005 | gst | basic | 0.0 | 7.0 | 0 | 25933 |  |
| oth_penalty_001 | gst | intermediate | 4.0 | 7.0 | 3 | 33499 |  |
| oth_penalty_002 | gst | basic | 1.0 | 8.0 | 1 | 28658 |  |
| oth_penalty_003 | gst | advanced | 0.0 | 8.0 | 0 | 22043 |  |
| oth_penalty_004 | customs | intermediate | 0.0 | 7.0 | 0 | 17169 |  |
| oth_advance_ruling_001 | gst | basic | 0.0 | 8.0 | 0 | 20936 |  |
| oth_advance_ruling_002 | gst | intermediate | 0.0 | 7.0 | 0 | 32582 |  |
| oth_advance_ruling_003 | gst | advanced | 5.0 | 9.0 | 2 | 43465 |  |
| oth_advance_ruling_004 | gst | basic | 6.0 | 8.0 | 3 | 41488 |  |
| oth_anti_evasion_001 | gst | intermediate | 6.0 | 7.0 | 3 | 34069 |  |
| oth_anti_evasion_002 | gst | intermediate | 6.0 | 7.0 | 3 | 39140 |  |
| oth_anti_evasion_003 | gst | basic | 5.0 | 8.0 | 3 | 46309 |  |
| oth_offences_001 | gst | intermediate | 5.0 | 10.0 | 3 | 39484 |  |
| oth_offences_002 | gst | advanced | 4.0 | 9.0 | 3 | 42420 |  |
| oth_interest_001 | gst | intermediate | 0.0 | 8.0 | 0 | 39149 |  |
| oth_interest_002 | gst | basic | 7.0 | 8.0 | 3 | 45601 |  |