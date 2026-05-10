# CBIC RAG Eval Run

Run dir: `D:\_gpu_rig_ai\eval\runs\p1_2_postdeploy_20260422_003323`
Score: **103.0 / 319.0** (32.29%)
Items: 50  |  Errors: 0
Latency ms — median 25957.0  p95 31503  max 44893

## By category

| Category | N | Points | Max | % | Errors |
|---|---:|---:|---:|---:|---:|
| central_excise | 8 | 17.0 | 47.0 | 36.17 | 0 |
| customs | 10 | 17.0 | 66.0 | 25.76 | 0 |
| gst | 20 | 47.0 | 124.0 | 37.9 | 0 |
| others | 6 | 9.0 | 43.0 | 20.93 | 0 |
| service_tax | 6 | 13.0 | 39.0 | 33.33 | 0 |

## Per-item

| id | cat | diff | pts | max | judge | lat_ms | err |
|---|---|---|---:|---:|---:|---:|---|
| gst_pos_001 | gst | basic | 3.0 | 6.0 | None | 17624 |  |
| gst_pos_002 | gst | intermediate | 3.0 | 6.0 | None | 26804 |  |
| gst_pos_003 | gst | basic | 1.0 | 5.0 | None | 23471 |  |
| gst_pos_004 | gst | intermediate | 2.0 | 5.0 | None | 24523 |  |
| gst_pos_005 | gst | complex | 2.0 | 10.0 | None | 28535 |  |
| gst_cs_001 | gst | intermediate | 3.0 | 7.0 | None | 26653 |  |
| gst_cs_002 | gst | basic | 1.0 | 5.0 | None | 24138 |  |
| gst_itc_001 | gst | basic | 1.0 | 5.0 | None | 26683 |  |
| gst_itc_002 | gst | intermediate | 2.0 | 7.0 | None | 26763 |  |
| gst_itc_003 | gst | intermediate | 0.0 | 7.0 | None | 25349 |  |
| gst_itc_004 | gst | basic | 1.0 | 7.0 | None | 23916 |  |
| gst_tos_001 | gst | basic | 3.0 | 6.0 | None | 26051 |  |
| gst_tos_002 | gst | basic | 3.0 | 6.0 | None | 25343 |  |
| gst_inv_001 | gst | basic | 3.0 | 5.0 | None | 23665 |  |
| gst_inv_002 | gst | intermediate | 1.0 | 6.0 | None | 29337 |  |
| gst_rcm_001 | gst | basic | 3.0 | 6.0 | None | 25927 |  |
| gst_rcm_002 | gst | intermediate | 3.0 | 6.0 | None | 19552 |  |
| gst_ref_001 | gst | intermediate | 5.0 | 6.0 | None | 28119 |  |
| gst_ewb_001 | gst | basic | 4.0 | 6.0 | None | 24367 |  |
| gst_exp_001 | gst | intermediate | 3.0 | 7.0 | None | 28531 |  |
| cus_val_001 | customs | basic | 2.0 | 5.0 | None | 24300 |  |
| cus_val_002 | customs | intermediate | 1.0 | 7.0 | None | 25169 |  |
| cus_cls_001 | customs | basic | 0.0 | 7.0 | None | 30000 |  |
| cus_wh_001 | customs | intermediate | 1.0 | 7.0 | None | 23430 |  |
| cus_db_001 | customs | basic | 3.0 | 6.0 | None | 26168 |  |
| cus_db_002 | customs | intermediate | 0.0 | 5.0 | None | 25692 |  |
| cus_svb_001 | customs | intermediate | 3.0 | 5.0 | None | 22600 |  |
| cus_ar_001 | customs | basic | 4.0 | 8.0 | None | 23710 |  |
| cus_cls_002 | customs | complex | 1.0 | 8.0 | None | 44893 |  |
| cus_ig_001 | customs | basic | 2.0 | 8.0 | None | 28050 |  |
| exc_val_001 | central_excise | basic | 3.0 | 6.0 | None | 26312 |  |
| exc_val_002 | central_excise | intermediate | 2.0 | 6.0 | None | 25683 |  |
| exc_cen_001 | central_excise | basic | 3.0 | 6.0 | None | 26057 |  |
| exc_cen_002 | central_excise | intermediate | 5.0 | 6.0 | None | 26149 |  |
| exc_man_001 | central_excise | basic | 1.0 | 6.0 | None | 26549 |  |
| exc_man_002 | central_excise | intermediate | 2.0 | 7.0 | None | 25580 |  |
| exc_ssi_001 | central_excise | basic | 0.0 | 5.0 | None | 23612 |  |
| exc_ssi_002 | central_excise | complex | 1.0 | 5.0 | None | 30784 |  |
| st_val_001 | service_tax | basic | 4.0 | 6.0 | None | 29828 |  |
| st_nl_001 | service_tax | basic | 3.0 | 8.0 | None | 31503 |  |
| st_exp_001 | service_tax | complex | 2.0 | 7.0 | None | 28227 |  |
| st_pos_001 | service_tax | basic | 1.0 | 4.0 | None | 25987 |  |
| st_rcm_001 | service_tax | intermediate | 2.0 | 7.0 | None | 24501 |  |
| st_lev_001 | service_tax | basic | 1.0 | 7.0 | None | 24231 |  |
| oth_ap_001 | others | basic | 2.0 | 7.0 | None | 32341 |  |
| oth_ap_002 | others | basic | 2.0 | 7.0 | None | 23696 |  |
| oth_ap_003 | others | intermediate | 1.0 | 7.0 | None | 24672 |  |
| oth_gaar_001 | others | basic | 1.0 | 5.0 | None | 26138 |  |
| oth_pen_001 | others | complex | 2.0 | 10.0 | None | 30306 |  |
| oth_cls_001 | others | intermediate | 1.0 | 7.0 | None | 22485 |  |