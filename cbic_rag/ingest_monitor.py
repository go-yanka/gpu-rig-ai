#!/usr/bin/env python3
"""ingest_monitor.py - live dashboard for CBIC RAG ingestion.

Read-only. Reads from:
  - Qdrant collection cbic_v1 via HTTP
  - /opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite
  - /tmp/cbic-ingest.log
  - /tmp/cbic-progress.log
  - ps (process health)
  - rocm-smi (GPU power)

Usage: python3 ingest_monitor.py [--refresh 2]
"""
from __future__ import annotations
import argparse, os, sqlite3, subprocess, sys, time, json, re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen, Request

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("pip install rich"); sys.exit(1)

MANIFEST = "/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite"
QURL = "http://127.0.0.1:6343"
COLL = "cbic_v1"
INGEST_LOG = "/tmp/cbic-ingest.log"
PROGRESS_LOG = "/tmp/cbic-progress.log"
EMBED_PORTS = [11434, 11440, 11441, 11443, 11444, 11446]
EMBED_GPU = {11434: 2, 11440: 0, 11441: 1, 11443: 3, 11444: 4, 11446: 6}


def q_get(path):
    try:
        with urlopen(f"{QURL}{path}", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def collection_info():
    d = q_get(f"/collections/{COLL}")
    if not d: return None
    return d.get("result", {})


def total_docs():
    try:
        c = sqlite3.connect(f"file:{MANIFEST}?mode=ro", uri=True, timeout=3)
        n = c.execute("SELECT COUNT(*) FROM docs WHERE path_en IS NOT NULL OR path_hi IS NOT NULL").fetchone()[0]
        c.close()
        return n
    except Exception:
        return 0


def docs_per_category():
    try:
        c = sqlite3.connect(f"file:{MANIFEST}?mode=ro", uri=True, timeout=3)
        r = {}
        for row in c.execute("SELECT category, COUNT(*) FROM docs WHERE path_en IS NOT NULL OR path_hi IS NOT NULL GROUP BY category"):
            r[row[0] or "unknown"] = row[1]
        c.close()
        return r
    except Exception:
        return {}


def qdrant_category_counts(max_sample=40000):
    counts = Counter()
    offset = None
    sampled = 0
    try:
        while sampled < max_sample:
            body = {"limit": 2000, "with_payload": ["category"], "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            req = Request(f"{QURL}/collections/{COLL}/points/scroll",
                          data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=5) as r:
                d = json.loads(r.read())
            pts = d.get("result", {}).get("points", [])
            if not pts: break
            for p in pts:
                cat = (p.get("payload") or {}).get("category", "?")
                counts[cat] += 1
            sampled += len(pts)
            offset = d.get("result", {}).get("next_page_offset")
            if offset is None:
                break
    except Exception:
        pass
    return counts, sampled


def proc_info():
    try:
        out = subprocess.check_output(["pgrep", "-fa", "python3 -u ingest.py"], text=True).strip().splitlines()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    pid = int(out[0].split()[0])
    try:
        elapsed = subprocess.check_output(["ps","-o","etime=","-p",str(pid)], text=True).strip()
    except Exception:
        elapsed = "?"
    workers = len(out) - 1
    return {"pid": pid, "elapsed": elapsed, "workers": workers}


def gpu_watts():
    try:
        out = subprocess.check_output(["/opt/rocm/bin/rocm-smi","--showpower","--json"], text=True, timeout=3)
        d = json.loads(out)
        w = {}
        for k, v in d.items():
            m = re.search(r"card(\d+)", k)
            if not m: continue
            gid = int(m.group(1))
            for kk, vv in v.items():
                if "Average Graphics Package Power" in kk:
                    try: w[gid] = float(vv)
                    except Exception: pass
        return w
    except Exception:
        return {}


def gpu_vram():
    try:
        out = subprocess.check_output(["/opt/rocm/bin/rocm-smi","--showmeminfo","vram","--json"], text=True, timeout=3)
        d = json.loads(out)
        r = {}
        for k, v in d.items():
            m = re.search(r"card(\d+)", k)
            if not m: continue
            gid = int(m.group(1))
            used = total = 0
            for kk, vv in v.items():
                if "Used Memory" in kk:
                    try: used = int(vv) / (1024*1024)
                    except: pass
                if "Total Memory" in kk:
                    try: total = int(vv) / (1024*1024)
                    except: pass
            r[gid] = (used, total)
        return r
    except Exception:
        return {}


_EMBED_RE = re.compile(r"\|\s+200\s+\|\s+([\d.]+)(us|ms|s)\s+\|.*POST\s+\"/api/embed\"")
_UNIT = {"us": 1e-3, "ms": 1.0, "s": 1000.0}


def embed_stats_for_service(unit, since_sec=60):
    """Return (count, avg_ms) of /api/embed requests in last `since_sec` seconds.
    `unit` is systemd unit name (e.g. 'ollama.service', 'ollama-embed@3.service')."""
    try:
        out = subprocess.check_output(
            ["sudo","-n","journalctl","-u",unit,"--since",f"{since_sec} sec ago","--no-pager","-o","cat"],
            text=True, timeout=5, stderr=subprocess.DEVNULL)
    except Exception:
        return 0, 0.0
    latencies = []
    for line in out.splitlines():
        m = _EMBED_RE.search(line)
        if m:
            val = float(m.group(1)); u = m.group(2)
            latencies.append(val * _UNIT[u])
    if not latencies:
        return 0, 0.0
    return len(latencies), sum(latencies)/len(latencies)


def all_embed_stats():
    """Return dict[gpu_id] -> (reqs_per_min, avg_ms)."""
    # port 11434 is ollama.service on GPU 2; 11440+g is ollama-embed@g
    mapping = [(2, "ollama.service"),
               (0, "ollama-embed@0.service"),
               (1, "ollama-embed@1.service"),
               (3, "ollama-embed@3.service"),
               (4, "ollama-embed@4.service"),
               (6, "ollama-embed@6.service")]
    out = {}
    for g, u in mapping:
        c, avg = embed_stats_for_service(u, since_sec=60)
        out[g] = (c, avg)
    return out


def port_busy(port):
    try:
        out = subprocess.check_output(["ss","-tln"], text=True)
        return f":{port} " in out
    except Exception:
        return False


def tail(path, n=10):
    try:
        return subprocess.check_output(["tail","-n",str(n),path], text=True).strip().splitlines()
    except Exception:
        return []


def fmt(n):
    return f"{n:,}"


def header_panel(info, total, proc, rate_pts_per_s):
    points = info.get("points_count", 0) if info else 0
    status = info.get("status", "?") if info else "?"
    est_docs = points // 12 if points else 0
    pct = (est_docs / total * 100) if total else 0.0
    eta_min = 0
    if rate_pts_per_s > 0 and total:
        rem = max(0, total*12 - points)
        eta_min = int(rem / rate_pts_per_s / 60)
    tbl = Table.grid(expand=True, padding=(0,2))
    for _ in range(4):
        tbl.add_column(ratio=1)
    tbl.add_row(
        Text.from_markup(f"Points\n[bold cyan]{fmt(points)}[/]", justify="center"),
        Text.from_markup(f"~Docs\n[bold green]{fmt(est_docs)}/{fmt(total)}[/]  ({pct:.1f}%)", justify="center"),
        Text.from_markup(f"Rate\n[bold yellow]{rate_pts_per_s*60:.0f} pts/min[/]", justify="center"),
        Text.from_markup(f"ETA\n[bold magenta]{eta_min} min[/]", justify="center"),
    )
    sub = f"Status: [bold]{status}[/]   "
    if proc:
        sub += f"PID {proc['pid']}  workers={proc['workers']}  elapsed={proc['elapsed']}"
    else:
        sub += "[red bold]ingest NOT RUNNING[/]"
    return Panel(tbl, title=" CBIC RAG Ingestion ", subtitle=sub, border_style="cyan")


def category_panel(total_by_cat, qd_counts, sampled):
    tbl = Table(expand=True, show_header=True, header_style="bold")
    tbl.add_column("Category", style="white")
    tbl.add_column("Docs", justify="right")
    tbl.add_column("Pts sampled", justify="right")
    tbl.add_column("~Docs done", justify="right")
    tbl.add_column("Coverage", justify="right")
    for cat in sorted(total_by_cat.keys()):
        corpus = total_by_cat[cat]
        pts = qd_counts.get(cat, 0)
        est = pts // 12
        cov = (est / corpus * 100) if corpus else 0.0
        color = "green" if cov >= 95 else ("yellow" if cov >= 20 else "white")
        tbl.add_row(cat, fmt(corpus), fmt(pts), fmt(est), f"[{color}]{cov:.1f}%[/]")
    return Panel(tbl, title=f" Coverage (sampled {fmt(sampled)} pts) ", border_style="green")


def gpu_panel(watts, vram, embed_stats):
    tbl = Table(expand=True, show_header=True, header_style="bold")
    tbl.add_column("GPU", style="cyan", justify="center")
    tbl.add_column("Port", justify="center")
    tbl.add_column("State", justify="center")
    tbl.add_column("Req/min", justify="right")
    tbl.add_column("Avg lat", justify="right")
    tbl.add_column("Power", justify="right")
    tbl.add_column("VRAM", justify="right")
    for p in EMBED_PORTS:
        g = EMBED_GPU[p]
        up = port_busy(p)
        w = watts.get(g, 0.0)
        reqs, lat_ms = embed_stats.get(g, (0, 0.0))
        used, total = vram.get(g, (0, 0))
        vram_str = f"{used:.0f}/{total:.0f}MB" if total else "-"
        state = "[green]LIVE[/]" if up else "[red]down[/]"
        rate_color = "green" if reqs > 20 else ("yellow" if reqs > 0 else "red")
        tbl.add_row(f"GPU {g}", str(p), state,
                    f"[{rate_color}]{reqs}[/]",
                    f"{lat_ms:.0f} ms" if lat_ms else "-",
                    f"{w:.0f} W", vram_str)
    total_reqs = sum(c for c,_ in embed_stats.values())
    return Panel(tbl, title=f" Embed Fleet  (total {total_reqs} reqs/min) ",
                 border_style="magenta")


def log_panel(path, title, n=10, color="white"):
    lines = tail(path, n)
    body = "\n".join(lines) if lines else "(empty)"
    return Panel(Text(body, style=color), title=f" {title} ", border_style="blue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", type=float, default=2.0)
    args = ap.parse_args()
    console = Console()
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=6),
        Layout(name="middle", ratio=2),
        Layout(name="bottom", ratio=1),
    )
    layout["middle"].split_row(
        Layout(name="coverage", ratio=3),
        Layout(name="gpus", ratio=2),
    )
    layout["bottom"].split_row(
        Layout(name="progress_log", ratio=1),
        Layout(name="ingest_log", ratio=1),
    )
    total_cat = docs_per_category()
    total = sum(total_cat.values()) or total_docs()
    prev_pts = 0
    prev_t = time.time()
    rate = 0.0
    with Live(layout, console=console, refresh_per_second=1.0/args.refresh, screen=True) as live:
        while True:
            info = collection_info() or {}
            qd_counts, sampled = qdrant_category_counts()
            proc = proc_info()
            watts = gpu_watts()
            vram = gpu_vram()
            embed_stats = all_embed_stats()
            pts = info.get("points_count", 0)
            now = time.time()
            dt = max(1e-3, now - prev_t)
            inst = (pts - prev_pts) / dt
            rate = (0.6*rate + 0.4*inst) if rate > 0 else inst
            prev_pts = pts
            prev_t = now
            layout["header"].update(header_panel(info, total, proc, rate))
            layout["coverage"].update(category_panel(total_cat, qd_counts, sampled))
            layout["gpus"].update(gpu_panel(watts, vram, embed_stats))
            layout["progress_log"].update(log_panel(PROGRESS_LOG, "Progress (10s poll)", 8, "cyan"))
            layout["ingest_log"].update(log_panel(INGEST_LOG, "Ingest stdout", 8, "yellow"))
            time.sleep(args.refresh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
