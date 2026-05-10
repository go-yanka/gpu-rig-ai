# CBIC RAG Eval Run

Run dir: `D:\_gpu_rig_ai\eval\runs\baseline_a3off_20260421_220214`
Score: **106.0 / 319.0** (33.23%)
Items: 50  |  Errors: 0
Latency ms — median 51300.5  p95 55789  max 59639

## By category

| Category | N | Points | Max | % | Errors |
|---|---:|---:|---:|---:|---:|
| central_excise | 8 | 18.0 | 47.0 | 38.3 | 0 |
| customs | 10 | 17.0 | 66.0 | 25.76 | 0 |
| gst | 20 | 48.0 | 124.0 | 38.71 | 0 |
| others | 6 | 9.0 | 43.0 | 20.93 | 0 |
| service_tax | 6 | 14.0 | 39.0 | 35.9 | 0 |

## Per-item

| id | cat | diff | pts | max | judge | lat_ms | err |
|---|---|---|---:|---:|---:|---:|---|
| gst_pos_001 | gst | basic | 3.0 | 6.0 | None | 52035 |  |
| gst_pos_002 | gst | intermediate | 3.0 | 6.0 | None | 52036 |  |
| gst_pos_003 | gst | basic | 2.0 | 5.0 | None | 23464 |  |
| gst_pos_004 | gst | intermediate | 2.0 | 5.0 | None | 48124 |  |
| gst_pos_005 | gst | complex | 2.0 | 10.0 | None | 52981 |  |
| gst_cs_001 | gst | intermediate | 3.0 | 7.0 | None | 54941 |  |
| gst_cs_002 | gst | basic | 1.0 | 5.0 | None | 50739 |  |
| gst_itc_001 | gst | basic | 1.0 | 5.0 | None | 49531 |  |
| gst_itc_002 | gst | intermediate | 2.0 | 7.0 | None | 51032 |  |
| gst_itc_003 | gst | intermediate | 0.0 | 7.0 | None | 50840 |  |
| gst_itc_004 | gst | basic | 1.0 | 7.0 | None | 47745 |  |
| gst_tos_001 | gst | basic | 3.0 | 6.0 | None | 49073 |  |
| gst_tos_002 | gst | basic | 3.0 | 6.0 | None | 54503 |  |
| gst_inv_001 | gst | basic | 3.0 | 5.0 | None | 51786 |  |
| gst_inv_002 | gst | intermediate | 1.0 | 6.0 | None | 51663 |  |
| gst_rcm_001 | gst | basic | 3.0 | 6.0 | None | 54303 |  |
| gst_rcm_002 | gst | intermediate | 3.0 | 6.0 | None | 44712 |  |
| gst_ref_001 | gst | intermediate | 5.0 | 6.0 | None | 47255 |  |
| gst_ewb_001 | gst | basic | 4.0 | 6.0 | None | 50887 |  |
| gst_exp_001 | gst | intermediate | 3.0 | 7.0 | None | 48557 |  |
| cus_val_001 | customs | basic | 2.0 | 5.0 | None | 49712 |  |
| cus_val_002 | customs | intermediate | 1.0 | 7.0 | None | 48783 |  |
| cus_cls_001 | customs | basic | 0.0 | 7.0 | None | 54551 |  |
| cus_wh_001 | customs | intermediate | 1.0 | 7.0 | None | 52708 |  |
| cus_db_001 | customs | basic | 3.0 | 6.0 | None | 48504 |  |
| cus_db_002 | customs | intermediate | 0.0 | 5.0 | None | 54215 |  |
| cus_svb_001 | customs | intermediate | 3.0 | 5.0 | None | 50668 |  |
| cus_ar_001 | customs | basic | 4.0 | 8.0 | None | 45378 |  |
| cus_cls_002 | customs | complex | 1.0 | 8.0 | None | 47153 |  |
| cus_ig_001 | customs | basic | 2.0 | 8.0 | None | 51430 |  |
| exc_val_001 | central_excise | basic | 3.0 | 6.0 | None | 53471 |  |
| exc_val_002 | central_excise | intermediate | 2.0 | 6.0 | None | 51171 |  |
| exc_cen_001 | central_excise | basic | 4.0 | 6.0 | None | 54394 |  |
| exc_cen_002 | central_excise | intermediate | 5.0 | 6.0 | None | 54804 |  |
| exc_man_001 | central_excise | basic | 1.0 | 6.0 | None | 51492 |  |
| exc_man_002 | central_excise | intermediate | 2.0 | 7.0 | None | 50903 |  |
| exc_ssi_001 | central_excise | basic | 0.0 | 5.0 | None | 48048 |  |
| exc_ssi_002 | central_excise | complex | 1.0 | 5.0 | None | 53371 |  |
| st_val_001 | service_tax | basic | 4.0 | 6.0 | None | 59639 |  |
| st_nl_001 | service_tax | basic | 4.0 | 8.0 | None | 57285 |  |
| st_exp_001 | service_tax | complex | 2.0 | 7.0 | None | 55680 |  |
| st_pos_001 | service_tax | basic | 1.0 | 4.0 | None | 53079 |  |
| st_rcm_001 | service_tax | intermediate | 2.0 | 7.0 | None | 49179 |  |
| st_lev_001 | service_tax | basic | 1.0 | 7.0 | None | 47704 |  |
| oth_ap_001 | others | basic | 2.0 | 7.0 | None | 55789 |  |
| oth_ap_002 | others | basic | 2.0 | 7.0 | None | 55248 |  |
| oth_ap_003 | others | intermediate | 1.0 | 7.0 | None | 46844 |  |
| oth_gaar_001 | others | basic | 1.0 | 5.0 | None | 49962 |  |
| oth_pen_001 | others | complex | 2.0 | 10.0 | None | 55263 |  |
| oth_cls_001 | others | intermediate | 1.0 | 7.0 | None | 51682 |  |