# CBIC RAG Eval Run

Run dir: `D:\_gpu_rig_ai\eval\runs\a3on_20260421_222520`
Score: **69.0 / 319.0** (21.63%)
Items: 50  |  Errors: 0
Latency ms — median 55529.0  p95 79809  max 92704

## By category

| Category | N | Points | Max | % | Errors |
|---|---:|---:|---:|---:|---:|
| central_excise | 8 | 13.0 | 47.0 | 27.66 | 0 |
| customs | 10 | 14.0 | 66.0 | 21.21 | 0 |
| gst | 20 | 26.0 | 124.0 | 20.97 | 0 |
| others | 6 | 9.0 | 43.0 | 20.93 | 0 |
| service_tax | 6 | 7.0 | 39.0 | 17.95 | 0 |

## Per-item

| id | cat | diff | pts | max | judge | lat_ms | err |
|---|---|---|---:|---:|---:|---:|---|
| gst_pos_001 | gst | basic | 0.0 | 6.0 | None | 29223 |  |
| gst_pos_002 | gst | intermediate | 2.0 | 6.0 | None | 69580 |  |
| gst_pos_003 | gst | basic | 0.0 | 5.0 | None | 63605 |  |
| gst_pos_004 | gst | intermediate | 0.0 | 5.0 | None | 47211 |  |
| gst_pos_005 | gst | complex | 1.0 | 10.0 | None | 63579 |  |
| gst_cs_001 | gst | intermediate | 0.0 | 7.0 | None | 61917 |  |
| gst_cs_002 | gst | basic | 2.0 | 5.0 | None | 52507 |  |
| gst_itc_001 | gst | basic | 1.0 | 5.0 | None | 78990 |  |
| gst_itc_002 | gst | intermediate | 2.0 | 7.0 | None | 92704 |  |
| gst_itc_003 | gst | intermediate | 1.0 | 7.0 | None | 79809 |  |
| gst_itc_004 | gst | basic | 1.0 | 7.0 | None | 78506 |  |
| gst_tos_001 | gst | basic | 2.0 | 6.0 | None | 69437 |  |
| gst_tos_002 | gst | basic | 2.0 | 6.0 | None | 52420 |  |
| gst_inv_001 | gst | basic | 0.0 | 5.0 | None | 48872 |  |
| gst_inv_002 | gst | intermediate | 0.0 | 6.0 | None | 40544 |  |
| gst_rcm_001 | gst | basic | 4.0 | 6.0 | None | 45185 |  |
| gst_rcm_002 | gst | intermediate | 0.0 | 6.0 | None | 47112 |  |
| gst_ref_001 | gst | intermediate | 4.0 | 6.0 | None | 55592 |  |
| gst_ewb_001 | gst | basic | 3.0 | 6.0 | None | 62675 |  |
| gst_exp_001 | gst | intermediate | 1.0 | 7.0 | None | 59874 |  |
| cus_val_001 | customs | basic | 0.0 | 5.0 | None | 54150 |  |
| cus_val_002 | customs | intermediate | 0.0 | 7.0 | None | 57931 |  |
| cus_cls_001 | customs | basic | 1.0 | 7.0 | None | 65338 |  |
| cus_wh_001 | customs | intermediate | 2.0 | 7.0 | None | 58597 |  |
| cus_db_001 | customs | basic | 3.0 | 6.0 | None | 57963 |  |
| cus_db_002 | customs | intermediate | 0.0 | 5.0 | None | 52711 |  |
| cus_svb_001 | customs | intermediate | 3.0 | 5.0 | None | 49520 |  |
| cus_ar_001 | customs | basic | 4.0 | 8.0 | None | 59758 |  |
| cus_cls_002 | customs | complex | 0.0 | 8.0 | None | 54864 |  |
| cus_ig_001 | customs | basic | 1.0 | 8.0 | None | 50456 |  |
| exc_val_001 | central_excise | basic | 0.0 | 6.0 | None | 53102 |  |
| exc_val_002 | central_excise | intermediate | 2.0 | 6.0 | None | 47863 |  |
| exc_cen_001 | central_excise | basic | 0.0 | 6.0 | None | 49813 |  |
| exc_cen_002 | central_excise | intermediate | 5.0 | 6.0 | None | 56588 |  |
| exc_man_001 | central_excise | basic | 1.0 | 6.0 | None | 55466 |  |
| exc_man_002 | central_excise | intermediate | 3.0 | 7.0 | None | 47286 |  |
| exc_ssi_001 | central_excise | basic | 0.0 | 5.0 | None | 39593 |  |
| exc_ssi_002 | central_excise | complex | 2.0 | 5.0 | None | 45923 |  |
| st_val_001 | service_tax | basic | 0.0 | 6.0 | None | 57938 |  |
| st_nl_001 | service_tax | basic | 0.0 | 8.0 | None | 55225 |  |
| st_exp_001 | service_tax | complex | 0.0 | 7.0 | None | 54922 |  |
| st_pos_001 | service_tax | basic | 2.0 | 4.0 | None | 56972 |  |
| st_rcm_001 | service_tax | intermediate | 3.0 | 7.0 | None | 56259 |  |
| st_lev_001 | service_tax | basic | 2.0 | 7.0 | None | 54442 |  |
| oth_ap_001 | others | basic | 1.0 | 7.0 | None | 62961 |  |
| oth_ap_002 | others | basic | 3.0 | 7.0 | None | 59202 |  |
| oth_ap_003 | others | intermediate | 2.0 | 7.0 | None | 51375 |  |
| oth_gaar_001 | others | basic | 0.0 | 5.0 | None | 44269 |  |
| oth_pen_001 | others | complex | 2.0 | 10.0 | None | 77833 |  |
| oth_cls_001 | others | intermediate | 1.0 | 7.0 | None | 90082 |  |