# CBIC RAG Eval Run

Run dir: `D:\_gpu_rig_ai\eval\runs\baseline_20260421_211827`
Score: **0.0 / 319.0** (0.0%)
Items: 50  |  Errors: 50
Latency ms — median 10.0  p95 32  max 38

## By category

| Category | N | Points | Max | % | Errors |
|---|---:|---:|---:|---:|---:|
| central_excise | 8 | 0.0 | 47.0 | 0.0 | 8 |
| customs | 10 | 0.0 | 66.0 | 0.0 | 10 |
| gst | 20 | 0.0 | 124.0 | 0.0 | 20 |
| others | 6 | 0.0 | 43.0 | 0.0 | 6 |
| service_tax | 6 | 0.0 | 39.0 | 0.0 | 6 |

## Per-item

| id | cat | diff | pts | max | judge | lat_ms | err |
|---|---|---|---:|---:|---:|---:|---|
| gst_pos_001 | gst | basic | 0.0 | 6.0 | None | 33 | HTTP 404: {"detail":"Not Found"} |
| gst_pos_002 | gst | intermediate | 0.0 | 6.0 | None | 32 | HTTP 404: {"detail":"Not Found"} |
| gst_pos_003 | gst | basic | 0.0 | 5.0 | None | 10 | HTTP 404: {"detail":"Not Found"} |
| gst_pos_004 | gst | intermediate | 0.0 | 5.0 | None | 10 | HTTP 404: {"detail":"Not Found"} |
| gst_pos_005 | gst | complex | 0.0 | 10.0 | None | 10 | HTTP 404: {"detail":"Not Found"} |
| gst_cs_001 | gst | intermediate | 0.0 | 7.0 | None | 11 | HTTP 404: {"detail":"Not Found"} |
| gst_cs_002 | gst | basic | 0.0 | 5.0 | None | 10 | HTTP 404: {"detail":"Not Found"} |
| gst_itc_001 | gst | basic | 0.0 | 5.0 | None | 11 | HTTP 404: {"detail":"Not Found"} |
| gst_itc_002 | gst | intermediate | 0.0 | 7.0 | None | 8 | HTTP 404: {"detail":"Not Found"} |
| gst_itc_003 | gst | intermediate | 0.0 | 7.0 | None | 11 | HTTP 404: {"detail":"Not Found"} |
| gst_itc_004 | gst | basic | 0.0 | 7.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| gst_tos_001 | gst | basic | 0.0 | 6.0 | None | 8 | HTTP 404: {"detail":"Not Found"} |
| gst_tos_002 | gst | basic | 0.0 | 6.0 | None | 10 | HTTP 404: {"detail":"Not Found"} |
| gst_inv_001 | gst | basic | 0.0 | 5.0 | None | 13 | HTTP 404: {"detail":"Not Found"} |
| gst_inv_002 | gst | intermediate | 0.0 | 6.0 | None | 8 | HTTP 404: {"detail":"Not Found"} |
| gst_rcm_001 | gst | basic | 0.0 | 6.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| gst_rcm_002 | gst | intermediate | 0.0 | 6.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| gst_ref_001 | gst | intermediate | 0.0 | 6.0 | None | 22 | HTTP 404: {"detail":"Not Found"} |
| gst_ewb_001 | gst | basic | 0.0 | 6.0 | None | 8 | HTTP 404: {"detail":"Not Found"} |
| gst_exp_001 | gst | intermediate | 0.0 | 7.0 | None | 7 | HTTP 404: {"detail":"Not Found"} |
| cus_val_001 | customs | basic | 0.0 | 5.0 | None | 38 | HTTP 404: {"detail":"Not Found"} |
| cus_val_002 | customs | intermediate | 0.0 | 7.0 | None | 7 | HTTP 404: {"detail":"Not Found"} |
| cus_cls_001 | customs | basic | 0.0 | 7.0 | None | 14 | HTTP 404: {"detail":"Not Found"} |
| cus_wh_001 | customs | intermediate | 0.0 | 7.0 | None | 14 | HTTP 404: {"detail":"Not Found"} |
| cus_db_001 | customs | basic | 0.0 | 6.0 | None | 8 | HTTP 404: {"detail":"Not Found"} |
| cus_db_002 | customs | intermediate | 0.0 | 5.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| cus_svb_001 | customs | intermediate | 0.0 | 5.0 | None | 8 | HTTP 404: {"detail":"Not Found"} |
| cus_ar_001 | customs | basic | 0.0 | 8.0 | None | 10 | HTTP 404: {"detail":"Not Found"} |
| cus_cls_002 | customs | complex | 0.0 | 8.0 | None | 7 | HTTP 404: {"detail":"Not Found"} |
| cus_ig_001 | customs | basic | 0.0 | 8.0 | None | 7 | HTTP 404: {"detail":"Not Found"} |
| exc_val_001 | central_excise | basic | 0.0 | 6.0 | None | 7 | HTTP 404: {"detail":"Not Found"} |
| exc_val_002 | central_excise | intermediate | 0.0 | 6.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| exc_cen_001 | central_excise | basic | 0.0 | 6.0 | None | 10 | HTTP 404: {"detail":"Not Found"} |
| exc_cen_002 | central_excise | intermediate | 0.0 | 6.0 | None | 7 | HTTP 404: {"detail":"Not Found"} |
| exc_man_001 | central_excise | basic | 0.0 | 6.0 | None | 32 | HTTP 404: {"detail":"Not Found"} |
| exc_man_002 | central_excise | intermediate | 0.0 | 7.0 | None | 30 | HTTP 404: {"detail":"Not Found"} |
| exc_ssi_001 | central_excise | basic | 0.0 | 5.0 | None | 12 | HTTP 404: {"detail":"Not Found"} |
| exc_ssi_002 | central_excise | complex | 0.0 | 5.0 | None | 11 | HTTP 404: {"detail":"Not Found"} |
| st_val_001 | service_tax | basic | 0.0 | 6.0 | None | 10 | HTTP 404: {"detail":"Not Found"} |
| st_nl_001 | service_tax | basic | 0.0 | 8.0 | None | 11 | HTTP 404: {"detail":"Not Found"} |
| st_exp_001 | service_tax | complex | 0.0 | 7.0 | None | 10 | HTTP 404: {"detail":"Not Found"} |
| st_pos_001 | service_tax | basic | 0.0 | 4.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| st_rcm_001 | service_tax | intermediate | 0.0 | 7.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| st_lev_001 | service_tax | basic | 0.0 | 7.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| oth_ap_001 | others | basic | 0.0 | 7.0 | None | 26 | HTTP 404: {"detail":"Not Found"} |
| oth_ap_002 | others | basic | 0.0 | 7.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| oth_ap_003 | others | intermediate | 0.0 | 7.0 | None | 18 | HTTP 404: {"detail":"Not Found"} |
| oth_gaar_001 | others | basic | 0.0 | 5.0 | None | 24 | HTTP 404: {"detail":"Not Found"} |
| oth_pen_001 | others | complex | 0.0 | 10.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |
| oth_cls_001 | others | intermediate | 0.0 | 7.0 | None | 9 | HTTP 404: {"detail":"Not Found"} |