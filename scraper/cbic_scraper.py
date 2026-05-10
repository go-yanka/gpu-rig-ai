"""CBIC Tax Information Portal scraper (taxinformation.cbic.gov.in).

Strategy (discovered by API reverse-engineering):
  - /api/cbic-tax-msts              -> 5 taxes (GST, Customs, Central Excise,
                                        Service Tax, HSNS Cess)
  - /api/cbic-act-msts              -> 15 Acts          (contentFilePath)
  - /api/cbic-rule-msts             -> 95 Rules         (+ contentFilePathHi)
  - /api/cbic-regulation-msts       -> 583 Regulations  (+ contentFilePathHi, docFilePath)
  - /api/cbic-regulation-doc-msts   -> 71 Reg. docs
  - /api/cbic-form-msts             -> 406 Forms
  - /api/cbic-instruction-msts      -> 571 Instructions (docFilePath + docFilePathHi)
  - /api/cbic-order-msts            -> 360 Orders       (docFilePath + docFilePathHi)
  - /api/cbic-allied-act-msts       -> 28 Allied Acts
  - /api/cbic-allied-act-dtls       -> 486 Allied Act documents
  - /api/cbic-service-tax-msts      -> 10 Service-tax items
  - /api/cbic-others-document-msts  -> 45 Misc (docFilePath + docFilePathHi)
  - /api/cbic-attachment-dtls       -> 19 enclosures    (docFilePath)

  Bulk circulars:
  - /api/cbic-circular-msts/fetchAllCircularsByTaxId/{taxId}
      GST 270  | Customs 1677 | CE 1061 | ST 212 | HSN 0   (~3,220 total)

  Notifications (bulk endpoint is server-broken; individual ID lookup works):
  - /api/cbic-notification-msts/{id}   for id in 1_000_001 .. ~1_010_628

  Download endpoint:
  - /content/pdf/<path>  -> {"data":"<base64>","fileName":"..."}
"""
from __future__ import annotations
import sys, os, re, json, time, base64, logging, hashlib
from typing import Iterable, Optional, List, Dict, Any, Tuple
from pathlib import Path
from urllib.parse import quote
import requests
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from base_scraper import BaseScraper, Document, ThreadSafeTokenBucket, CircuitBreaker

log = logging.getLogger("cbic")

BASE = "https://taxinformation.cbic.gov.in"

TAX_ID_TO_CATEGORY = {
    1000001: "gst",
    1000002: "customs",
    1000003: "central_excise",
    1000004: "service_tax",
    1000005: "hsn_cess",  # 2026-04-26 fix: was 100005 (6 digits) — broke API, silently skipped HSN/Cess scrape
}

CATEGORY_MAP = {
    "gst": "gst",
    "customs": "customs",
    "ce": "central_excise",
    "central-excise": "central_excise",
    "central_excise": "central_excise",
    "centralexcise": "central_excise",
    "excise": "central_excise",
    "st": "service_tax",
    "service-tax": "service_tax",
    "service_tax": "service_tax",
    "servicetax": "service_tax",
    "hsn": "hsn_cess",
    "hsns": "hsn_cess",
    "hsns-cess": "hsn_cess",
    "hsns_cess": "hsn_cess",
    "finance": "finance_acts",
    "finance-acts": "finance_acts",
    "finance_acts": "finance_acts",
}

_SUBCAT_CANON = {
    "acts": "acts",
    "rules": "rules",
    "regulations": "regulations",
    "circulars": "circulars",
    "notifications": "notifications",
    "instructions": "instructions",
    "orders": "orders",
    "forms": "forms",
    "allied_acts": "allied_acts",
    "allied-acts": "allied_acts",
    "others": "others",
    "service_tax": "service_tax",
}

BULK_ENDPOINTS: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = [
    ("cbic-act-msts",               ("contentFilePath",),                        ("contentFilePathHi",)),
    ("cbic-rule-msts",              ("contentFilePath",),                        ("contentFilePathHi",)),
    ("cbic-regulation-msts",        ("contentFilePath", "docFilePath"),          ("contentFilePathHi", "docFilePathHi")),
    ("cbic-regulation-doc-msts",    ("contentFilePath",),                        ("contentFilePathHi",)),
    ("cbic-form-msts",              ("contentFilePath",),                        ("contentFilePathHi",)),
    ("cbic-instruction-msts",       ("docFilePath",),                            ("docFilePathHi",)),
    ("cbic-order-msts",             ("docFilePath",),                            ("docFilePathHi",)),
    ("cbic-service-tax-msts",       ("contentFilePath",),                        ("contentFilePathHi",)),
    ("cbic-others-document-msts",   ("docFilePath",),                            ("docFilePathHi",)),
    ("cbic-attachment-dtls",        ("docFilePath",),                            ()),
    ("cbic-allied-act-dtls",        ("doc_file_path",),                          ("doc_file_path_hi",)),
    ("cbic-allied-act-msts",        ("doc_file_path",),                          ("doc_file_path_hi",)),
]

NOTIF_ID_MIN = 1_000_001
NOTIF_ID_MAX = 1_015_000


class CBICScraper(BaseScraper):
    source = "cbic"
    base_url = BASE
    rate_limit_rps = 3.0
    user_agent = ("Mozilla/5.0 IndianLegalAI-research-scrape/1.0 "
                  "(academic corpus; Rahul / 192.168.1.107)")

    def __init__(self, root_dir: str):
        super().__init__(root_dir)
        self._primed = False
        self.session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # === THREADING INFRA ===
        self.session_lock = threading.Lock()
        self.token_bucket = ThreadSafeTokenBucket(rps=5.0)
        self.circuit_breaker = CircuitBreaker(pause_seconds=45)
        self.result_queue = queue.Queue()
        self.counter_lock = threading.Lock()
        self.writer_thread = None
        self.completion_count = 0

    def _prime(self):
        if self._primed:
            return
        log.info("priming session (GET / for cookies)...")
        with self.session_lock:
            self.session.headers.update({
                "User-Agent": self.user_agent,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
                "Referer": BASE + "/",
                "Origin": BASE,
            })
            r = self.session.get(BASE + "/", timeout=60)
            r.raise_for_status()
        self._primed = True
        log.info(f"session primed; {len(self.session.cookies)} cookies")

    def _api_get(self, path: str, params: Optional[Dict] = None,
                 retries: int = 2) -> Any:
        self._prime()
        p = path.lstrip("/")
        if not p.startswith("api/"):
            p = "api/" + p
        last_err = None
        for attempt in range(retries + 1):
            self.rl.wait()
            try:
                r = self.session.get(BASE + "/" + p, params=params, timeout=60)
                if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
                    return r.json()
                if r.status_code == 404:
                    return None
                last_err = f"http {r.status_code} ctype={r.headers.get('content-type','')[:30]}"
            except Exception as e:
                last_err = str(e)
            time.sleep(0.5 * (attempt + 1))
        log.debug(f"_api_get {path} {params}: {last_err}")
        return None

    @staticmethod
    def _normalise(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        p = path.replace("\\", "/").strip().lstrip("/")
        return p or None

    @staticmethod
    def _tax_id(item: Dict[str, Any]) -> Optional[int]:
        for k in ("taxId", "cbicTaxMst", "tax", "tax_id"):
            v = item.get(k)
            if isinstance(v, dict):
                v = v.get("id")
            if isinstance(v, (int, str)) and str(v).isdigit():
                return int(v)
        return None

    @classmethod
    def _category_from_path(cls, pdf_path: str) -> Tuple[str, str, Optional[str], List[str]]:
        parts = pdf_path.split("/")
        if parts and parts[0].lower() == "tax_repository":
            parts = parts[1:]
        if not parts:
            return "unknown", "misc", None, []
        cat_folder = parts[0].lower()
        category = CATEGORY_MAP.get(cat_folder, cat_folder)
        subcat_raw = parts[1].lower() if len(parts) > 1 else "misc"
        subcat = _SUBCAT_CANON.get(subcat_raw, subcat_raw)
        year = None
        for seg in parts:
            m = re.search(r"(?:^|[-_/])(20\d{2}|19\d{2})(?:$|[-_/\.])", seg)
            if m:
                year = m.group(1)
                break
        return category, subcat, year, parts

    @staticmethod
    def _iso_date(v: Any) -> Optional[str]:
        if not v:
            return None
        s = str(v)
        m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
        return m.group(1) if m else None

    @staticmethod
    def _title_of(item: Dict[str, Any]) -> str:
        for k in ("actName", "ruleDocName", "regulationName", "formName",
                  "circularName", "notificationName", "instructionName",
                  "orderName", "documentName", "regulationDocName",
                  "contentName", "allied_act_name", "allied_act_subject"):
            v = item.get(k)
            if v:
                return str(v)[:240]
        return ""

    @staticmethod
    def _number_of(item: Dict[str, Any]) -> Optional[str]:
        for k in ("actNo", "ruleDocNo", "regulationNo", "formNo",
                  "circularNo", "notificationNo", "instructionNo", "orderNo",
                  "regulationDocNo", "allied_act_ref_no", "allied_act_ccn_no"):
            v = item.get(k)
            if v:
                return str(v)[:120]
        return None

    @staticmethod
    def _date_of(item: Dict[str, Any]) -> Optional[str]:
        for k in ("issueDt", "notificationDt", "instructionDt", "orderDt",
                  "amendDt", "allied_act_dt", "issue_dt"):
            iso = CBICScraper._iso_date(item.get(k))
            if iso:
                return iso
        return None

    def discover(self, scope: Optional[List[str]] = None) -> Iterable[Document]:
        self._prime()
        scope_set = set(scope) if scope else None

        for ep, en_keys, hi_keys in BULK_ENDPOINTS:
            log.info(f"fetching /{ep} ...")
            items = self._api_get(ep)
            if not isinstance(items, list):
                log.warning(f"  {ep}: not a list; got {type(items).__name__}")
                continue
            log.info(f"  {ep}: {len(items):,} rows")
            for a in items:
                doc = self._row_to_doc(a, en_keys, hi_keys, ep_slug=ep)
                if doc is None:
                    continue
                if scope_set and doc.category not in scope_set:
                    continue
                yield doc

        for tax_id, cat in TAX_ID_TO_CATEGORY.items():
            if scope_set and cat not in scope_set:
                continue
            log.info(f"fetching circulars for tax={tax_id} ({cat}) ...")
            items = self._api_get(f"cbic-circular-msts/fetchAllCircularsByTaxId/{tax_id}")
            if not isinstance(items, list):
                log.warning(f"  circulars {cat}: not a list; got {type(items).__name__}")
                continue
            log.info(f"  circulars {cat}: {len(items):,} rows")
            for a in items:
                doc = self._row_to_doc(a, ("docFilePath",), ("docFilePathHi",),
                                        ep_slug="cbic-circular-msts")
                if doc is None:
                    continue
                if scope_set and doc.category not in scope_set:
                    continue
                yield doc

        if (scope_set is None) or scope_set.intersection(
                {"gst", "customs", "central_excise", "service_tax", "hsn_cess"}):
            log.info(f"scanning notifications id={NOTIF_ID_MIN}..{NOTIF_ID_MAX} ...")
            misses = 0
            hits = 0
            for nid in range(NOTIF_ID_MIN, NOTIF_ID_MAX + 1):
                item = self._api_get(f"cbic-notification-msts/{nid}")
                if not isinstance(item, dict) or not item.get("id"):
                    misses += 1
                    if misses >= 300:
                        log.info(f"  notifications: {hits} hits, early exit "
                                 f"after {misses} consecutive misses at id={nid}")
                        break
                    continue
                misses = 0
                hits += 1
                doc = self._row_to_doc(item, ("docFilePath",), ("docFilePathHi",),
                                        ep_slug="cbic-notification-msts")
                if doc is None:
                    continue
                if scope_set and doc.category not in scope_set:
                    continue
                yield doc
                if hits % 500 == 0:
                    log.info(f"  notifications: {hits:,} hits so far (id={nid})")
            log.info(f"notifications scan complete: {hits:,} hits")

    def _row_to_doc(self,
                    item: Dict[str, Any],
                    en_keys: Tuple[str, ...],
                    hi_keys: Tuple[str, ...],
                    ep_slug: str) -> Optional[Document]:
        en_path = next((self._normalise(item.get(k)) for k in en_keys
                        if self._normalise(item.get(k))), None)
        hi_path = next((self._normalise(item.get(k)) for k in hi_keys
                        if self._normalise(item.get(k))), None)
        primary = en_path or hi_path
        if not primary or not primary.lower().endswith(".pdf"):
            return None

        category, subcategory, year, parts = self._category_from_path(primary)
        tax_id = self._tax_id(item)
        if tax_id and tax_id in TAX_ID_TO_CATEGORY:
            tax_cat = TAX_ID_TO_CATEGORY[tax_id]
            if category == "unknown":
                category = tax_cat

        item_id = item.get("id") or item.get("contentId")
        doc_id = f"{ep_slug}:{item_id}" if item_id else f"{ep_slug}:{primary}"

        url_en = f"{BASE}/content/pdf/{quote(en_path)}" if en_path else None
        url_hi = f"{BASE}/content/pdf/{quote(hi_path)}" if hi_path else None

        return Document(
            source="cbic",
            category=category,
            subcategory=subcategory,
            doc_id=doc_id,
            title=self._title_of(item),
            number=self._number_of(item),
            date=self._date_of(item),
            year=year,
            url_en=url_en,
            url_hi=url_hi,
            extra={
                "endpoint": ep_slug,
                "item_id": item_id,
                "content_id": item.get("contentId") or item.get("content_id"),
                "path_parts": parts,
                "en_path": en_path,
                "hi_path": hi_path,
                "category_hint": item.get("notificationCategory")
                                  or item.get("circularCategory")
                                  or item.get("ruleCategory")
                                  or item.get("instructionCategory")
                                  or item.get("orderCategory")
                                  or item.get("formCategory")
                                  or item.get("regulationCategory"),
                "tax_id": tax_id,
            },
        )

    def stage_download(self, scope=None, languages=("en", "hi"),
                       max_docs: Optional[int] = None) -> Dict[str, Any]:
        self._prime()
        started = time.strftime("%Y-%m-%d %H:%M:%S")

        def manifest_writer():
            while True:
                item = self.result_queue.get()
                if item is None:
                    break
                try:
                    if item.get("error"):
                        self.manifest.mark_error(**item["error"])
                    else:
                        self.manifest.mark_downloaded(**item["success"])
                except Exception as e:
                    log.error(f"Manifest writer error: {e}")
                self.result_queue.task_done()

                with self.counter_lock:
                    self.completion_count += 1
                    if self.completion_count % 50 == 0:
                        log.info(f"download progress: {self.completion_count} docs processed")

        self.writer_thread = threading.Thread(target=manifest_writer, daemon=True, name="writer")
        self.writer_thread.start()

        pending = self.manifest.pending_downloads(
            self.source,
            category=(scope[0] if scope and len(scope)==1 else None))
        if scope and len(scope) > 1:
            pending = [r for r in pending if r["category"] in scope]
        if max_docs:
            pending = pending[:max_docs]

        total = len(pending)
        log.info(f"download: {total} pending for scope={scope} langs={languages}")

        downloaded = failed = 0

        def _worker(row):
            nonlocal downloaded, failed
            cat = row["category"]
            sub = row["subcategory"]
            did = row["doc_id"]
            try:
                extra = json.loads(row["extra_json"] or "{}")
                row_d = dict(row)

                paths = {"en": None, "hi": None}
                shas  = {"en": None, "hi": None}
                sizes = {"en": None, "hi": None}
                last_err = None

                for lang in languages:
                    url = row_d.get(f"url_{lang}")
                    if not url:
                        continue

                    self.circuit_breaker.wait_if_tripped()
                    self.token_bucket.wait()

                    lp = extra.get("en_path") if lang == "en" else extra.get("hi_path")
                    parts = extra.get("path_parts") or []
                    fname = (lp.split("/")[-1] if lp else None) or (parts[-1] if parts else f"{did}.pdf")
                    if lang == "hi":
                        if "_hi" not in fname.lower() and "hindi" not in fname.lower():
                            fname = re.sub(r"\.pdf$", "_hi.pdf", fname, flags=re.I)

                    year_seg = row["year"] or "unknown"
                    outdir = self.root / cat / sub / year_seg
                    dest = outdir / fname

                    if dest.exists() and dest.stat().st_size > 0:
                        paths[lang] = str(dest)
                        sizes[lang] = dest.stat().st_size
                        shas[lang] = self._sha256_of(dest)
                        continue

                    t_start = time.time()
                    try:
                        r = self.session.get(url, timeout=(15, 45))

                        if r.status_code in (429, 503):
                            self.circuit_breaker.trip()
                            raise RuntimeError(f"http {r.status_code} — circuit breaker tripped")

                        if r.status_code != 200:
                            raise RuntimeError(f"http {r.status_code}")
                        if len(r.content) == 0:
                            raise RuntimeError("empty body (server has no file)")

                        ct = r.headers.get("content-type", "")
                        if "json" in ct:
                            payload = r.json()
                            b64 = payload.get("data")
                            if not b64:
                                raise RuntimeError("no data field in JSON payload")
                            pdf_bytes = base64.b64decode(b64)
                        elif "pdf" in ct or r.content[:4] == b"%PDF":
                            pdf_bytes = r.content
                        else:
                            raise RuntimeError(f"unexpected ctype {ct[:30]}")

                        if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
                            raise RuntimeError("not a PDF (no %PDF header)")

                        dest.parent.mkdir(parents=True, exist_ok=True)
                        tmp = dest.with_suffix(dest.suffix + ".part")
                        with open(tmp, "wb") as f:
                            f.write(pdf_bytes)
                        os.replace(tmp, dest)

                        h = hashlib.sha256(pdf_bytes).hexdigest()
                        paths[lang] = str(dest)
                        shas[lang] = h
                        sizes[lang] = len(pdf_bytes)
                        log.info(f"[{cat}/{sub}/{did}] {lang} OK {len(pdf_bytes)}B in {time.time()-t_start:.2f}s")

                    except Exception as e:
                        elapsed = time.time() - t_start
                        last_err = f"{lang}: {e}"
                        log.warning(f"[{cat}/{sub}/{did}] {lang} FAIL in {elapsed:.2f}s: {e}")

                if any(paths.values()):
                    self.result_queue.put({
                        "success": {
                            "source": self.source, "category": cat, "subcategory": sub, "doc_id": did,
                            "path_en": paths["en"], "path_hi": paths["hi"],
                            "sha256_en": shas["en"], "sha256_hi": shas["hi"],
                            "bytes_en": sizes["en"], "bytes_hi": sizes["hi"]
                        }
                    })
                    with self.counter_lock:
                        downloaded += 1
                else:
                    self.result_queue.put({
                        "error": {
                            "source": self.source, "category": cat, "subcategory": sub, "doc_id": did,
                            "err": last_err or "no urls"
                        }
                    })
                    with self.counter_lock:
                        failed += 1
            except Exception as e:
                log.error(f"[{cat}/{sub}/{did}] worker crashed: {e}")
                try:
                    self.result_queue.put({
                        "error": {
                            "source": self.source,
                            "category": cat,
                            "subcategory": sub,
                            "doc_id": did,
                            "err": f"worker crash: {e}"
                        }
                    })
                except Exception:
                    pass
                with self.counter_lock:
                    failed += 1

        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="dl") as executor:
            futures = [executor.submit(_worker, row) for row in pending]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Worker exception: {e}")
                    with self.counter_lock:
                        failed += 1

        self.result_queue.put(None)
        self.writer_thread.join()

        ended = time.strftime("%Y-%m-%d %H:%M:%S")
        self.manifest.record_run(self.source, "download", ",".join(scope or ["all"]),
                                 started, ended, total, downloaded, failed, "ok")

        log.info(f"download complete: {downloaded} ok, {failed} failed")
        return {"total": total, "downloaded": downloaded, "failed": failed}


SCOPE_MAP = {
    "gst": "GST",
    "customs": "Customs",
    "central_excise": "Central Excise",
    "service_tax": "Service Tax",
    "hsn_cess": "HSNS Cess",
    "finance_acts": "Finance Acts",
}
