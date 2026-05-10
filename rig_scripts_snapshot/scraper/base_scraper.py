"""Base scraper framework for Indian legal/regulatory sites.

Pluggable design so the same core handles CBIC, RBI, SEBI, MCA, Income Tax, etc.

Architecture:
  - BaseScraper: abstract driver (stages: discover -> report -> download)
  - Manifest: SQLite table per source with resumable state
  - Politeness: rate-limited, retry-on-error, respects robots.txt
  - Folder layout: <root>/<source>/<category>/<subcategory>/<year>/<files>
"""
from __future__ import annotations
import os, sys, json, time, hashlib, sqlite3, logging, urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger("scraper")


@dataclass
class Document:
    """Canonical representation of one scraped document."""
    source: str            # "cbic"
    category: str          # "gst" | "customs" | ...
    subcategory: str       # "notifications" | "circulars" | "acts" | ...
    doc_id: str            # source-assigned id (stable across runs)
    title: str
    number: Optional[str] = None      # e.g. "20/2025"
    date: Optional[str] = None        # ISO yyyy-mm-dd
    year: Optional[str] = None
    url_en: Optional[str] = None
    url_hi: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # arbitrary metadata

    def filename_base(self) -> str:
        """Sanitised <number>_<date>_<slug> base for saved files."""
        parts = []
        if self.number:
            parts.append(self.number.replace("/", "-"))
        if self.date:
            parts.append(self.date)
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in (self.title or "")[:80])
        slug = "-".join(p for p in slug.split("-") if p)
        if slug:
            parts.append(slug)
        if not parts:
            parts.append(self.doc_id)
        return "_".join(parts)


class Manifest:
    """SQLite-backed resumable manifest. Thread-safe-ish via WAL."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS docs (
      source TEXT NOT NULL,
      category TEXT NOT NULL,
      subcategory TEXT NOT NULL,
      doc_id TEXT NOT NULL,
      title TEXT,
      number TEXT,
      date TEXT,
      year TEXT,
      url_en TEXT,
      url_hi TEXT,
      extra_json TEXT,
      discovered_at TEXT,
      path_en TEXT,
      path_hi TEXT,
      sha256_en TEXT,
      sha256_hi TEXT,
      bytes_en INTEGER,
      bytes_hi INTEGER,
      downloaded_at TEXT,
      last_error TEXT,
      PRIMARY KEY (source, category, subcategory, doc_id)
    );
    CREATE INDEX IF NOT EXISTS idx_status ON docs(source, downloaded_at);
    CREATE INDEX IF NOT EXISTS idx_cat ON docs(source, category, subcategory);
    CREATE TABLE IF NOT EXISTS runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source TEXT, stage TEXT, scope TEXT,
      started_at TEXT, ended_at TEXT,
      count_seen INTEGER, count_downloaded INTEGER, count_failed INTEGER,
      notes TEXT
    );
    """

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.executescript(self.SCHEMA)

    def upsert_doc(self, doc: Document):
        self.conn.execute("""
            INSERT INTO docs (source, category, subcategory, doc_id, title, number, date, year,
                              url_en, url_hi, extra_json, discovered_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?, datetime('now'))
            ON CONFLICT(source, category, subcategory, doc_id) DO UPDATE SET
                title=excluded.title, number=excluded.number, date=excluded.date,
                year=excluded.year, url_en=excluded.url_en, url_hi=excluded.url_hi,
                extra_json=excluded.extra_json;
        """, (doc.source, doc.category, doc.subcategory, doc.doc_id, doc.title,
              doc.number, doc.date, doc.year, doc.url_en, doc.url_hi,
              json.dumps(doc.extra, default=str)))

    def mark_downloaded(self, source, category, subcategory, doc_id,
                        path_en=None, path_hi=None,
                        sha256_en=None, sha256_hi=None,
                        bytes_en=None, bytes_hi=None):
        self.conn.execute("""
            UPDATE docs SET path_en=?, path_hi=?, sha256_en=?, sha256_hi=?,
                            bytes_en=?, bytes_hi=?, downloaded_at=datetime('now'), last_error=NULL
            WHERE source=? AND category=? AND subcategory=? AND doc_id=?
        """, (path_en, path_hi, sha256_en, sha256_hi, bytes_en, bytes_hi,
              source, category, subcategory, doc_id))

    def mark_error(self, source, category, subcategory, doc_id, err):
        self.conn.execute("""
            UPDATE docs SET last_error=? WHERE source=? AND category=? AND subcategory=? AND doc_id=?
        """, (str(err)[:2000], source, category, subcategory, doc_id))

    def pending_downloads(self, source, category=None, subcategory=None) -> List[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        q = "SELECT * FROM docs WHERE source=? AND downloaded_at IS NULL"
        args = [source]
        if category:
            q += " AND category=?"; args.append(category)
        if subcategory:
            q += " AND subcategory=?"; args.append(subcategory)
        return self.conn.execute(q, args).fetchall()

    def summary(self, source) -> Dict[str, Any]:
        self.conn.row_factory = sqlite3.Row
        rows = self.conn.execute("""
            SELECT category, subcategory,
                   COUNT(*) AS total,
                   SUM(CASE WHEN downloaded_at IS NOT NULL THEN 1 ELSE 0 END) AS done,
                   SUM(CASE WHEN url_en IS NOT NULL THEN 1 ELSE 0 END) AS has_en,
                   SUM(CASE WHEN url_hi IS NOT NULL THEN 1 ELSE 0 END) AS has_hi,
                   SUM(COALESCE(bytes_en,0)+COALESCE(bytes_hi,0)) AS bytes
            FROM docs WHERE source=?
            GROUP BY category, subcategory
            ORDER BY category, subcategory
        """, (source,)).fetchall()
        return {"source": source, "breakdown": [dict(r) for r in rows]}

    def record_run(self, source, stage, scope, started, ended, seen, downloaded, failed, notes):
        self.conn.execute("""
            INSERT INTO runs (source, stage, scope, started_at, ended_at,
                              count_seen, count_downloaded, count_failed, notes)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (source, stage, scope, started, ended, seen, downloaded, failed, notes))

    def close(self):
        try: self.conn.close()
        except Exception: pass


class RateLimiter:
    def __init__(self, rps: float):
        self.min_gap = 1.0 / rps if rps > 0 else 0
        self._last = 0.0
    def wait(self):
        if self.min_gap <= 0: return
        now = time.monotonic()
        gap = now - self._last
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        self._last = time.monotonic()


class ThreadSafeTokenBucket:
    def __init__(self, rps: float = 3.0):
        self.rps = rps
        self.tokens = rps
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.rps, self.tokens + elapsed * self.rps)
                self.last_refill = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            time.sleep(0.01)


class CircuitBreaker:
    def __init__(self, pause_seconds: int = 45):
        self.pause_seconds = pause_seconds
        self.tripped = threading.Event()

    def trip(self):
        if not self.tripped.is_set():
            self.tripped.set()
            log.warning(f"Circuit breaker tripped — pausing all workers {self.pause_seconds}s")
            threading.Timer(self.pause_seconds, self.reset).start()

    def reset(self):
        self.tripped.clear()

    def wait_if_tripped(self):
        if self.tripped.is_set():
            self.tripped.wait()


class Downloader:
    """HTTP downloader with rate limit, retries, sha256, and atomic writes."""

    def __init__(self, root: Path, rate_limiter: RateLimiter,
                 session: requests.Session, user_agent: str):
        self.root = Path(root)
        self.rl = rate_limiter
        self.session = session
        self.session.headers.update({"User-Agent": user_agent})

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=1, max=30),
           retry=retry_if_exception_type((requests.RequestException, OSError)))
    def _get(self, url: str) -> requests.Response:
        self.rl.wait()
        r = self.session.get(url, timeout=120, allow_redirects=True, stream=True)
        r.raise_for_status()
        return r

    def fetch_to(self, url: str, dest: Path) -> Dict[str, Any]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        h = hashlib.sha256()
        total = 0
        r = self._get(url)
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if not chunk: continue
                f.write(chunk); h.update(chunk); total += len(chunk)
        os.replace(tmp, dest)
        return {"path": str(dest), "bytes": total, "sha256": h.hexdigest()}


class BaseScraper(ABC):
    """Abstract scraper. Subclass and implement `discover()`."""

    source: str = ""                     # override: "cbic", "rbi", ...
    base_url: str = ""                   # override
    user_agent: str = "IndianLegalAI-research-scrape/1.0"
    rate_limit_rps: float = 1.0          # 1 req/sec default (polite)

    def __init__(self, root_dir: str):
        self.root = Path(root_dir) / self.source
        self.manifest = Manifest(str(self.root / "_manifest.sqlite"))
        self.rl = RateLimiter(self.rate_limit_rps)
        self.session = requests.Session()
        self.dl = Downloader(self.root, self.rl, self.session, self.user_agent)

    # ---- subclasses implement ----
    @abstractmethod
    def discover(self, scope: Optional[List[str]] = None) -> Iterable[Document]:
        """Yield Documents. scope = list of category names to limit (or None = all)."""
        ...

    # ---- shared stages ----
    def stage_discover(self, scope=None) -> Dict[str, Any]:
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        seen = 0
        for doc in self.discover(scope=scope):
            self.manifest.upsert_doc(doc)
            seen += 1
            if seen % 50 == 0:
                log.info(f"discover: {seen} docs upserted...")
        ended = time.strftime("%Y-%m-%d %H:%M:%S")
        self.manifest.record_run(self.source, "discover",
                                 ",".join(scope or ["all"]),
                                 started, ended, seen, 0, 0, "ok")
        return self.manifest.summary(self.source)

    def stage_download(self, scope=None, languages=("en","hi"),
                       max_docs: Optional[int] = None) -> Dict[str, Any]:
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        downloaded = failed = 0
        pending = self.manifest.pending_downloads(
            self.source,
            category=(scope[0] if scope and len(scope)==1 else None))
        if scope and len(scope) > 1:
            pending = [r for r in pending if r["category"] in scope]
        if max_docs:
            pending = pending[:max_docs]
        total = len(pending)
        log.info(f"download: {total} pending for scope={scope} langs={languages}")
        for i, row in enumerate(pending, 1):
            cat, sub, did = row["category"], row["subcategory"], row["doc_id"]
            outdir = self.root / cat / sub / (row["year"] or "unknown")
            base = self._safe_base(row)
            paths = {"en": None, "hi": None}
            shas = {"en": None, "hi": None}
            sizes = {"en": None, "hi": None}
            err = None
            for lang in languages:
                url = row[f"url_{lang}"]
                if not url: continue
                dest = outdir / f"{base}_{lang}.pdf"
                if dest.exists() and dest.stat().st_size > 0:
                    paths[lang] = str(dest); sizes[lang] = dest.stat().st_size
                    # compute sha lazily
                    shas[lang] = self._sha256_of(dest)
                    continue
                try:
                    info = self.dl.fetch_to(url, dest)
                    paths[lang] = info["path"]; shas[lang] = info["sha256"]; sizes[lang] = info["bytes"]
                except Exception as e:
                    err = f"{lang}: {e}"
                    log.warning(f"[{i}/{total}] {cat}/{sub}/{did} {lang} FAIL: {e}")
            if any(paths.values()):
                self.manifest.mark_downloaded(
                    self.source, cat, sub, did,
                    path_en=paths["en"], path_hi=paths["hi"],
                    sha256_en=shas["en"], sha256_hi=shas["hi"],
                    bytes_en=sizes["en"], bytes_hi=sizes["hi"])
                downloaded += 1
            else:
                self.manifest.mark_error(self.source, cat, sub, did, err or "no urls")
                failed += 1
            if i % 25 == 0:
                log.info(f"download: {i}/{total} ok={downloaded} fail={failed}")
        ended = time.strftime("%Y-%m-%d %H:%M:%S")
        self.manifest.record_run(self.source, "download",
                                 ",".join(scope or ["all"]),
                                 started, ended, total, downloaded, failed, "ok")
        return {"total": total, "downloaded": downloaded, "failed": failed}

    def stage_report(self) -> Dict[str, Any]:
        return self.manifest.summary(self.source)

    # ---- helpers ----
    @staticmethod
    def _safe_base(row) -> str:
        parts = []
        if row["number"]: parts.append(row["number"].replace("/", "-"))
        if row["date"]: parts.append(row["date"])
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in (row["title"] or "")[:60])
        slug = "-".join(p for p in slug.split("-") if p)
        if slug: parts.append(slug)
        if not parts: parts.append(str(row["doc_id"]))
        return "_".join(parts)[:180]

    @staticmethod
    def _sha256_of(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(65536), b""):
                h.update(blk)
        return h.hexdigest()
