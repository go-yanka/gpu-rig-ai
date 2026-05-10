#!/usr/bin/env python3
"""ingest_monitor.py - live dashboard for CBIC RAG ingestion (direct in-process Vulkan).

Read-only. Reads from:
  - Qdrant collection cbic_v1 via HTTP :6343
  - /opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite (corpus total)
  - systemd journal (cbic-ingest.service) for flush events and recent log
  - ps / pgrep for ingest PID + GPU worker children
  - rocm-smi for per-GPU power and VRAM

Usage: python3 ingest_monitor.py [--refresh 2]
"""
from __future__ import annotations
import argparse, os, sqlite3, subprocess, sys, time, json, re, collections
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
    print('pip install rich'); sys.exit(1)

MANIFEST = '/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite'
QURL = 'http://127.0.0.1:6343'
COLL = 'cbic_v1'
SERVICE = 'cbic-ingest.service'
GPU_IDS = [int(x) for x in os.environ.get('EMBED_GPUS', '0,1,3,4,5,6').split(',')]


def q_get(path):
    try:
        with urlopen(f'{QURL}{path}', timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def collection_info():
    d = q_get(f'/collections/{COLL}')
    return (d or {}).get('result', {}) or {}


def total_docs():
    try:
        c = sqlite3.connect(f'file:{MANIFEST}?mode=ro', uri=True, timeout=3)
        n = c.execute('SELECT COUNT(*) FROM docs WHERE path_en IS NOT NULL OR path_hi IS NOT NULL').fetchone()[0]
        c.close()
        return n
    except Exception:
        return 0


def docs_per_category():
    try:
        c = sqlite3.connect(f'file:{MANIFEST}?mode=ro', uri=True, timeout=3)
        r = {}
        for row in c.execute('SELECT category, COUNT(*) FROM docs WHERE path_en IS NOT NULL OR path_hi IS NOT NULL GROUP BY category'):
            r[row[0] or 'unknown'] = row[1]
        c.close()
        return r
    except Exception:
        return {}


def qdrant_category_counts(max_sample=40000):
    counts = collections.Counter()
    offset = None
    sampled = 0
    try:
        while sampled < max_sample:
            body = {'limit': 2000, 'with_payload': ['category'], 'with_vector': False}
            if offset is not None:
                body['offset'] = offset
            req = Request(f'{QURL}/collections/{COLL}/points/scroll',
                          data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
            with urlopen(req, timeout=5) as r:
                d = json.loads(r.read())
            pts = d.get('result', {}).get('points', [])
            if not pts: break
            for p in pts:
                cat = (p.get('payload') or {}).get('category', '?')
                counts[cat] += 1
            sampled += len(pts)
            offset = d.get('result', {}).get('next_page_offset')
            if offset is None:
                break
    except Exception:
        pass
    return counts, sampled


def proc_info():
    """Return dict with main ingest pid + child GPU worker pids."""
    try:
        out = subprocess.check_output(['pgrep', '-fa', 'python3 -u ingest.py'], text=True).strip().splitlines()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    pid = int(out[0].split()[0])
    try:
        elapsed = subprocess.check_output(['ps', '-o', 'etime=', '-p', str(pid)], text=True).strip()
    except Exception:
        elapsed = '?'
    worker_pids = []
    try:
        tree = subprocess.check_output(['pgrep', '-P', str(pid)], text=True).strip().splitlines()
        worker_pids = [int(x) for x in tree if x.strip().isdigit()]
        deeper = []
        for wp in worker_pids:
            try:
                sub = subprocess.check_output(['pgrep', '-P', str(wp)], text=True).strip().splitlines()
                deeper += [int(x) for x in sub if x.strip().isdigit()]
            except Exception:
                pass
        worker_pids = worker_pids + deeper
    except Exception:
        pass
    return {'pid': pid, 'elapsed': elapsed, 'workers': len(worker_pids), 'worker_pids': worker_pids}


def gpu_watts():
    try:
        out = subprocess.check_output(['/opt/rocm/bin/rocm-smi', '--showpower', '--json'], text=True, timeout=3)
        d = json.loads(out)
        w = {}
        for k, v in d.items():
            m = re.search(r'card(\d+)', k)
            if not m: continue
            gid = int(m.group(1))
            for kk, vv in v.items():
                if 'Average Graphics Package Power' in kk:
                    try: w[gid] = float(vv)
                    except Exception: pass
        return w
    except Exception:
        return {}


def gpu_vram():
    try:
        out = subprocess.check_output(['/opt/rocm/bin/rocm-smi', '--showmeminfo', 'vram', '--json'], text=True, timeout=3)
        d = json.loads(out)
        r = {}
        for k, v in d.items():
            m = re.search(r'card(\d+)', k)
            if not m: continue
            gid = int(m.group(1))
            used = total = 0
            for kk, vv in v.items():
                if 'Used Memory' in kk:
                    try: used = int(vv) / (1024 * 1024)
                    except Exception: pass
                if 'Total Memory' in kk:
                    try: total = int(vv) / (1024 * 1024)
                    except Exception: pass
            r[gid] = (used, total)
        return r
    except Exception:
        return {}


_FLUSH_RE = re.compile(r'\[flush #\d+\]\s+batch=(\d+)\s+upserted=(\d+)\s+total=(\d+)\s+took=([\d.]+)s')


def journal_lines(since_sec=120, n=400):
    try:
        out = subprocess.check_output(
            ['sudo', '-n', 'journalctl', '-u', SERVICE, '--since', f'{since_sec} sec ago', '--no-pager', '-o', 'cat', '-q'],
            text=True, timeout=5, stderr=subprocess.DEVNULL)
        lines = out.splitlines()
        return lines[-n:]
    except Exception:
        return []


def flush_stats(since_sec=60):
    """Parse recent flush lines from journal. Returns (flushes, total_items, avg_ms, last_total)."""
    lines = journal_lines(since_sec=since_sec, n=4000)
    flushes = 0
    items = 0
    dt_sum = 0.0
    last_total = 0
    for ln in lines:
        m = _FLUSH_RE.search(ln)
        if m:
            flushes += 1
            items += int(m.group(2))
            dt_sum += float(m.group(4))
            last_total = max(last_total, int(m.group(3)))
    avg = (dt_sum / flushes * 1000) if flushes else 0.0
    return flushes, items, avg, last_total


def tail_journal(n=10):
    lines = journal_lines(since_sec=120, n=n * 4)
    keep = [ln for ln in lines if 'embeddings required but some input' not in ln]
    return keep[-n:]


def fmt(n):
    return f'{n:,}'


def header_panel(info, total, proc, rate_pts_per_s, flushes_1m, items_1m):
    points = info.get('points_count', 0)
    status = info.get('status', '?')
    est_docs = points // 12 if points else 0
    pct = (est_docs / total * 100) if total else 0.0
    eta_min = 0
    if rate_pts_per_s > 0 and total:
        rem = max(0, total * 12 - points)
        eta_min = int(rem / rate_pts_per_s / 60)
    tbl = Table.grid(expand=True, padding=(0, 2))
    for _ in range(5):
        tbl.add_column(ratio=1)
    tbl.add_row(
        Text.from_markup(f'Points\n[bold cyan]{fmt(points)}[/]', justify='center'),
        Text.from_markup(f'~Docs\n[bold green]{fmt(est_docs)}/{fmt(total)}[/]  ({pct:.1f}%)', justify='center'),
        Text.from_markup(f'Items/s\n[bold yellow]{rate_pts_per_s:.1f}[/]', justify='center'),
        Text.from_markup(f'Flushes/min\n[bold blue]{flushes_1m}[/]  ({items_1m} pts)', justify='center'),
        Text.from_markup(f'ETA\n[bold magenta]{eta_min} min[/]', justify='center'),
    )
    sub = f'Status: [bold]{status}[/]   '
    if proc:
        sub += f"PID {proc['pid']}  GPU workers={proc['workers']}  elapsed={proc['elapsed']}"
    else:
        sub += '[red bold]ingest NOT RUNNING[/]'
    return Panel(tbl, title=' CBIC RAG Ingestion - direct Vulkan ', subtitle=sub, border_style='cyan')


def category_panel(total_by_cat, qd_counts, sampled):
    tbl = Table(expand=True, show_header=True, header_style='bold')
    tbl.add_column('Category', style='white')
    tbl.add_column('Docs', justify='right')
    tbl.add_column('Pts sampled', justify='right')
    tbl.add_column('~Docs done', justify='right')
    tbl.add_column('Coverage', justify='right')
    for cat in sorted(total_by_cat.keys()):
        corpus = total_by_cat[cat]
        pts = qd_counts.get(cat, 0)
        est = pts // 12
        cov = (est / corpus * 100) if corpus else 0.0
        color = 'green' if cov >= 95 else ('yellow' if cov >= 20 else 'white')
        tbl.add_row(cat, fmt(corpus), fmt(pts), fmt(est), f'[{color}]{cov:.1f}%[/]')
    return Panel(tbl, title=f' Coverage (sampled {fmt(sampled)} pts) ', border_style='green')


def gpu_panel(watts, vram, worker_count):
    tbl = Table(expand=True, show_header=True, header_style='bold')
    tbl.add_column('GPU', style='cyan', justify='center')
    tbl.add_column('State', justify='center')
    tbl.add_column('Power', justify='right')
    tbl.add_column('VRAM used', justify='right')
    tbl.add_column('VRAM total', justify='right')
    tbl.add_column('Util%', justify='right')
    for g in GPU_IDS:
        w = watts.get(g, 0.0)
        used, total = vram.get(g, (0, 0))
        util = (used / total * 100) if total else 0.0
        loaded = used > 400
        state = '[green]LOADED[/]' if loaded else '[yellow]idle[/]'
        power_color = 'green' if w > 40 else ('yellow' if w > 15 else 'white')
        tbl.add_row(f'GPU {g}', state,
                    f'[{power_color}]{w:.0f} W[/]',
                    f'{used:.0f} MB', f'{total:.0f} MB',
                    f'{util:.0f}%')
    title = f' GPU Fleet - {len(GPU_IDS)} cards, {worker_count} workers alive '
    return Panel(tbl, title=title, border_style='magenta')


def log_panel(lines, title, color='yellow'):
    body = '\n'.join(lines) if lines else '(waiting for log...)'
    return Panel(Text(body, style=color), title=f' {title} ', border_style='blue')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', type=float, default=2.0)
    args = ap.parse_args()
    console = Console()
    layout = Layout()
    layout.split_column(
        Layout(name='header', size=6),
        Layout(name='middle', ratio=2),
        Layout(name='bottom', ratio=1),
    )
    layout['middle'].split_row(
        Layout(name='coverage', ratio=3),
        Layout(name='gpus', ratio=2),
    )
    total_cat = docs_per_category()
    total = sum(total_cat.values()) or total_docs()
    prev_pts = 0
    prev_t = time.time()
    rate = 0.0
    with Live(layout, console=console, refresh_per_second=1.0 / args.refresh, screen=True) as live:
        while True:
            info = collection_info() or {}
            qd_counts, sampled = qdrant_category_counts()
            proc = proc_info()
            watts = gpu_watts()
            vram = gpu_vram()
            flushes_1m, items_1m, avg_ms, last_total = flush_stats(since_sec=60)
            last_lines = tail_journal(n=10)
            pts = info.get('points_count', 0)
            now = time.time()
            dt = max(1e-3, now - prev_t)
            inst = (pts - prev_pts) / dt
            rate = (0.6 * rate + 0.4 * inst) if rate > 0 else inst
            prev_pts = pts
            prev_t = now
            workers = proc['workers'] if proc else 0
            layout['header'].update(header_panel(info, total, proc, rate, flushes_1m, items_1m))
            layout['coverage'].update(category_panel(total_cat, qd_counts, sampled))
            layout['gpus'].update(gpu_panel(watts, vram, workers))
            layout['bottom'].update(log_panel(last_lines, f'Recent ingest log (avg flush latency {avg_ms:.0f} ms)', 'yellow'))
            time.sleep(args.refresh)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
