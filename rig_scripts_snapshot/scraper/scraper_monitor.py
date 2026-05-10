#!/usr/bin/env python3
"""scraper_monitor.py — READ-ONLY live dashboard for the CBIC scraper.

Reads from (never writes):
  - /opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite   (SQLite, opened ?mode=ro)
  - /opt/indian-legal-ai/scraper/logs/cbic_download_*.log     (tail)
  - /opt/indian-legal-ai/scraper/logs/cbic_pipeline.log       (tail)
  - /opt/indian-legal-ai/scraper/logs/watchdog.log            (tail)

Usage:
  python3 scraper_monitor.py
  python3 scraper_monitor.py --refresh 2
  python3 scraper_monitor.py --dump-schema    # one-shot; prints DB tables+cols and exits

Dependencies: `rich` (pip install rich).
"""
from __future__ import annotations
import argparse
import os
import sqlite3
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("ERROR: rich not installed. Run: pip install rich", file=sys.stderr)
    sys.exit(1)


DEFAULT_DB = "/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite"
DEFAULT_QA_DB = "/opt/indian-legal-ai/data/scraped/cbic/_qa.sqlite"
DEFAULT_LOGDIR = "/opt/indian-legal-ai/scraper/logs"
SCOPES = ["gst", "customs", "central_excise", "service_tax", "hsn_cess", "finance_acts"]

QA_STATUS_STYLE = {
    "ok":          "green",
    "image_only":  "yellow",
    "error_page":  "red",
    "corrupt":     "red bold",
    "missing":     "red bold",
}


# ---------- DB introspection ----------

def open_ro(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return con


def detect_table(con):
    tables = [r["name"] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    for c in ("manifest", "documents", "docs", "files"):
        if c in tables:
            return c
    sizes = []
    for t in tables:
        if t.startswith("sqlite_"):
            continue
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            sizes.append((n, t))
        except sqlite3.Error:
            pass
    sizes.sort(reverse=True)
    return sizes[0][1] if sizes else None


def detect_cols(con, table):
    return [r["name"] for r in con.execute(f"PRAGMA table_info({table})")]


def dump_schema(db_path):
    con = open_ro(db_path)
    print(f"DB: {db_path}\n")
    for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        t = r["name"]
        print(f"TABLE: {t}")
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  rows: {n:,}")
        except sqlite3.Error:
            pass
        for c in con.execute(f"PRAGMA table_info({t})"):
            print(f"  - {c['name']} ({c['type']})")
        print()
    con.close()


# ---------- Data queries (read-only) ----------

def get_stats(con, table, cols):
    has_en = "path_en" in cols
    has_hi = "path_hi" in cols
    has_err = "last_error" in cols or "error" in cols
    has_status = "status" in cols

    err_col = "last_error" if "last_error" in cols else ("error" if "error" in cols else None)

    dl_parts = []
    if has_en:
        dl_parts.append("path_en IS NOT NULL AND path_en != ''")
    if has_hi:
        dl_parts.append("path_hi IS NOT NULL AND path_hi != ''")
    if not dl_parts and has_status:
        dl_parts.append("status='downloaded'")
    if not dl_parts:
        dl_parts.append("0")
    dl = "(" + " OR ".join(dl_parts) + ")"

    fail = f"({err_col} IS NOT NULL AND {err_col} != '')" if err_col else "0"

    sql = f"""
        SELECT category,
               COUNT(*) AS total,
               SUM(CASE WHEN {dl} THEN 1 ELSE 0 END) AS downloaded,
               SUM(CASE WHEN {fail} AND NOT {dl} THEN 1 ELSE 0 END) AS failed
        FROM {table}
        WHERE source='cbic'
        GROUP BY category
    """
    stats = {}
    for r in con.execute(sql):
        cat = r["category"] or "unknown"
        total = r["total"] or 0
        done = r["downloaded"] or 0
        failed = r["failed"] or 0
        stats[cat] = {
            "total": total,
            "downloaded": done,
            "failed": failed,
            "pending": max(0, total - done - failed),
        }
    return stats


def get_recent(con, table, cols, limit=10):
    order = "rowid DESC"
    for c in ("downloaded_at", "updated_at", "modified_at"):
        if c in cols:
            order = f"{c} DESC"
            break
    path_expr = []
    if "path_en" in cols:
        path_expr.append("path_en")
    if "path_hi" in cols:
        path_expr.append("path_hi")
    if not path_expr:
        return []
    pe = f"COALESCE({', '.join(path_expr)})"
    where_dl = " OR ".join(f"{p} IS NOT NULL AND {p} != ''" for p in path_expr)
    sql = f"""
        SELECT category, subcategory, doc_id, {pe} AS path
        FROM {table}
        WHERE source='cbic' AND ({where_dl})
        ORDER BY {order}
        LIMIT ?
    """
    try:
        return list(con.execute(sql, (limit,)))
    except sqlite3.Error:
        return []


def get_recent_failures(con, table, cols, limit=5):
    err_col = "last_error" if "last_error" in cols else ("error" if "error" in cols else None)
    if not err_col:
        return []
    order = "rowid DESC"
    for c in ("updated_at", "modified_at", "downloaded_at"):
        if c in cols:
            order = f"{c} DESC"
            break
    sql = f"""
        SELECT category, subcategory, doc_id, {err_col} AS err
        FROM {table}
        WHERE source='cbic' AND {err_col} IS NOT NULL AND {err_col} != ''
        ORDER BY {order}
        LIMIT ?
    """
    try:
        return list(con.execute(sql, (limit,)))
    except sqlite3.Error:
        return []


# ---------- QA DB queries (separate _qa.sqlite) ----------

def get_qa_stats(qa_db):
    """Return (total_audited, by_status_counter, by_cat_issues) or (0, {}, {})."""
    if not Path(qa_db).exists():
        return 0, {}, {}
    try:
        con = open_ro(qa_db)
    except sqlite3.Error:
        return 0, {}, {}
    by_status = {}
    by_cat = {}
    total = 0
    try:
        for r in con.execute("SELECT status, COUNT(*) AS n FROM qa GROUP BY status"):
            by_status[r["status"]] = r["n"]
            total += r["n"]
        for r in con.execute("""
            SELECT category, status, COUNT(*) AS n FROM qa
            WHERE status != 'ok'
            GROUP BY category, status
        """):
            by_cat.setdefault(r["category"], {})[r["status"]] = r["n"]
    except sqlite3.Error:
        pass
    con.close()
    return total, by_status, by_cat


def get_recent_qa_issues(qa_db, limit=8):
    if not Path(qa_db).exists():
        return []
    try:
        con = open_ro(qa_db)
    except sqlite3.Error:
        return []
    rows = []
    try:
        rows = list(con.execute("""
            SELECT category, subcategory, doc_id, lang, status, reason, checked_at
            FROM qa
            WHERE status != 'ok'
            ORDER BY checked_at DESC
            LIMIT ?
        """, (limit,)))
    except sqlite3.Error:
        pass
    con.close()
    return rows


# ---------- Log / process probes ----------

def find_latest_log(logdir, pattern="cbic_download_*.log"):
    p = Path(logdir)
    if not p.exists():
        return None
    files = sorted(p.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0] if files else None


def tail_file(path, n=12):
    if not path or not Path(path).exists():
        return []
    try:
        out = subprocess.check_output(["tail", "-n", str(n), str(path)],
                                      text=True, timeout=2)
        return out.rstrip("\n").split("\n")
    except Exception:
        return []


def is_running(pattern):
    try:
        subprocess.check_output(["pgrep", "-f", pattern], timeout=2)
        return True
    except Exception:
        return False


# ---------- UI builders ----------

def fmt_eta(pending, rate_per_min):
    if rate_per_min <= 0 or pending <= 0:
        return "—"
    m = pending / rate_per_min
    if m < 60:
        return f"{m:.0f}m"
    if m < 60 * 48:
        return f"{m/60:.1f}h"
    return f"{m/1440:.1f}d"


def overall_panel(stats, rate_per_min, started_at):
    total = sum(s["total"] for s in stats.values())
    done = sum(s["downloaded"] for s in stats.values())
    failed = sum(s["failed"] for s in stats.values())
    pending = sum(s["pending"] for s in stats.values())
    pct = (done / total * 100) if total else 0

    up = int(time.time() - started_at)
    hh, mm, ss = up // 3600, (up % 3600) // 60, up % 60

    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="dim")
    t.add_column(style="bold")
    t.add_row("Total:", f"{total:,}")
    t.add_row("Downloaded:", f"[green]{done:,}[/] ({pct:.1f}%)")
    t.add_row("Failed:", f"[red]{failed:,}[/]")
    t.add_row("Pending:", f"[yellow]{pending:,}[/]")
    t.add_row("Rate:", f"[cyan]{rate_per_min:.1f}[/] files/min")
    t.add_row("ETA:", f"[cyan]{fmt_eta(pending, rate_per_min)}[/]")
    t.add_row("Monitor up:", f"{hh:02d}:{mm:02d}:{ss:02d}")
    return Panel(t, title="[bold cyan]CBIC Scraper — Overall[/]", border_style="cyan")


def scope_table(stats):
    t = Table(expand=True, show_lines=False, header_style="bold",
              title_style="bold", padding=(0, 1))
    t.add_column("Scope", style="bold")
    t.add_column("Total", justify="right")
    t.add_column("Done", justify="right", style="green")
    t.add_column("Fail", justify="right", style="red")
    t.add_column("Pend", justify="right", style="yellow")
    t.add_column("Progress", ratio=2)

    ordered = [c for c in SCOPES if c in stats] + \
              [c for c in stats if c not in SCOPES]
    for cat in ordered:
        s = stats[cat]
        total = s["total"]
        done = s["downloaded"]
        pct = (done / total * 100) if total else 0
        width = 30
        filled = int(width * pct / 100) if total else 0
        bar = f"[green]{'█'*filled}[/][dim]{'░'*(width-filled)}[/] {pct:5.1f}%"
        t.add_row(cat, f"{total:,}", f"{done:,}",
                  f"{s['failed']:,}", f"{s['pending']:,}", bar)
    return Panel(t, title="[bold]Per-scope progress[/]", border_style="cyan")


def process_panel():
    dl = is_running("run_scrape.py cbic --stage download")
    pipe = is_running("pipeline_all.sh")
    dog = is_running("watchdog.sh")
    qa = is_running("qa_watchdog.sh")
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="dim")
    t.add_column()
    t.add_row("Downloader:", "[green]● running[/]" if dl else "[dim]○ idle[/]")
    t.add_row("Pipeline:", "[green]● running[/]" if pipe else "[dim]○ idle[/]")
    t.add_row("Watchdog:", "[green]● running[/]" if dog else "[red]○ stopped[/]")
    t.add_row("QA Watchdog:", "[green]● running[/]" if qa else "[dim]○ idle[/]")
    return Panel(t, title="[bold magenta]Processes[/]", border_style="magenta")


def qa_panel(total, by_status, by_cat):
    """QA status panel. Green border if clean, red if issues detected."""
    ok = by_status.get("ok", 0)
    image_only = by_status.get("image_only", 0)
    corrupt = by_status.get("corrupt", 0)
    error_page = by_status.get("error_page", 0)
    missing = by_status.get("missing", 0)
    bad = corrupt + error_page + missing
    pct_ok = (ok / total * 100) if total else 0

    # border color signals severity
    if total == 0:
        border = "dim"
        title_prefix = "[dim]○"
    elif bad > 0:
        border = "red"
        title_prefix = "[red]⚠"
    elif image_only > 0:
        border = "yellow"
        title_prefix = "[yellow]⚠"
    else:
        border = "green"
        title_prefix = "[green]✓"

    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="dim")
    t.add_column(style="bold")
    t.add_row("Audited:", f"{total:,}")
    if total > 0:
        t.add_row("OK:", f"[green]{ok:,}[/] ({pct_ok:.1f}%)")
        if image_only:
            t.add_row("Image-only:", f"[yellow]{image_only:,}[/] (need OCR)")
        if error_page:
            t.add_row("Error page:", f"[red]{error_page:,}[/]")
        if corrupt:
            t.add_row("Corrupt:", f"[red bold]{corrupt:,}[/]")
        if missing:
            t.add_row("Missing:", f"[red bold]{missing:,}[/]")
        if bad == 0 and image_only == 0:
            t.add_row("", "[green]all files clean[/]")
    else:
        t.add_row("", "[dim]no QA data yet[/]")
    return Panel(t, title=f"{title_prefix} Quality Audit[/]", border_style=border)


def qa_issues_panel(issues):
    """Recent non-OK QA entries."""
    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=22, no_wrap=True)
    t.add_column(width=12, no_wrap=True)
    t.add_column(overflow="ellipsis", no_wrap=True)
    if not issues:
        t.add_row("", "", "[dim]no issues detected[/]")
        return Panel(t, title="[bold]Recent QA issues[/]", border_style="green")
    for r in issues:
        tag = f"{r['category']}/{r['subcategory']}"
        lang = f"[{r['lang']}]"
        status = r['status']
        style = QA_STATUS_STYLE.get(status, "white")
        reason = (r['reason'] or "")[:60]
        t.add_row(tag[:22], f"[{style}]{status}[/] {lang}", f"[{style}]{reason}[/]")
    border = "red" if any(r['status'] in ("corrupt","error_page","missing") for r in issues) else "yellow"
    return Panel(t, title="[bold]Recent QA issues[/]", border_style=border)


def recent_panel(rows, failures):
    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=24, no_wrap=True)
    t.add_column(overflow="ellipsis", no_wrap=True)
    t.add_row("[bold green]✓ recent downloads[/]", "")
    if not rows:
        t.add_row("", "[dim]none yet[/]")
    for r in rows:
        path = r["path"] or ""
        fname = os.path.basename(path) if path else (r["doc_id"] or "")
        tag = f"{r['category']}/{r['subcategory']}"
        t.add_row(tag[:24], fname[:80])
    if failures:
        t.add_row("", "")
        t.add_row("[bold red]✗ recent failures[/]", "")
        for r in failures:
            tag = f"{r['category']}/{r['subcategory']}"
            err = (r["err"] or "").replace("\n", " ")[:80]
            t.add_row(tag[:24], f"[red]{err}[/]")
    return Panel(t, title="[bold]Recent activity[/]", border_style="green")


def log_panel(logfile, n=14):
    lines = tail_file(logfile, n)
    text = Text()
    for line in lines:
        style = None
        low = line.lower()
        if "fail" in low or "error" in low or "traceback" in low:
            style = "red"
        elif "circuit breaker" in low:
            style = "yellow bold"
        elif "progress:" in low:
            style = "cyan"
        elif "complete" in low or "all done" in low:
            style = "green bold"
        text.append(line + "\n", style=style)
    name = logfile.name if logfile else "(no log)"
    return Panel(text, title=f"[bold]Log tail[/] — {name}", border_style="blue")


# ---------- Main loop ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--qa-db", default=DEFAULT_QA_DB)
    ap.add_argument("--logdir", default=DEFAULT_LOGDIR)
    ap.add_argument("--refresh", type=float, default=2.0)
    ap.add_argument("--dump-schema", action="store_true")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: manifest DB not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    if args.dump_schema:
        dump_schema(args.db)
        return

    con = open_ro(args.db)
    table = detect_table(con)
    if not table:
        print("ERROR: could not detect manifest table", file=sys.stderr)
        sys.exit(1)
    cols = detect_cols(con, table)
    con.close()

    console = Console()
    started = time.time()
    history = deque(maxlen=60)

    layout = Layout()
    layout.split_column(
        Layout(name="top", size=12),
        Layout(name="middle", size=14),
        Layout(name="qa_row", size=11),
        Layout(name="bottom", size=18),
    )
    layout["top"].split_row(
        Layout(name="overall"),
        Layout(name="procs", ratio=1),
    )
    layout["middle"].split_row(
        Layout(name="scopes", ratio=3),
        Layout(name="recent", ratio=2),
    )
    layout["qa_row"].split_row(
        Layout(name="qa_stats", ratio=2),
        Layout(name="qa_issues", ratio=3),
    )

    with Live(layout, console=console,
              refresh_per_second=max(1, int(1/args.refresh)),
              screen=True):
        while True:
            try:
                con = open_ro(args.db)
                stats = get_stats(con, table, cols)
                recent = get_recent(con, table, cols, limit=8)
                failures = get_recent_failures(con, table, cols, limit=5)
                con.close()

                now = time.time()
                done_now = sum(s["downloaded"] for s in stats.values())
                history.append((now, done_now))
                if len(history) >= 2:
                    t0, c0 = history[0]
                    t1, c1 = history[-1]
                    dt = max(t1 - t0, 1)
                    rate = (c1 - c0) / dt * 60
                else:
                    rate = 0.0

                logfile = find_latest_log(args.logdir)

                qa_total, qa_by_status, qa_by_cat = get_qa_stats(args.qa_db)
                qa_issues = get_recent_qa_issues(args.qa_db, limit=7)

                layout["overall"].update(overall_panel(stats, rate, started))
                layout["procs"].update(process_panel())
                layout["scopes"].update(scope_table(stats))
                layout["recent"].update(recent_panel(recent, failures))
                layout["qa_stats"].update(qa_panel(qa_total, qa_by_status, qa_by_cat))
                layout["qa_issues"].update(qa_issues_panel(qa_issues))
                layout["bottom"].update(log_panel(logfile, n=16))
            except Exception as e:
                console.log(f"[red]monitor loop error:[/] {e}")
            time.sleep(args.refresh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
