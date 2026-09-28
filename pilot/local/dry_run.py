#!/usr/bin/env python3
"""Run every test-pack mission through a running pilot and write a quick report.

    python pilot/local/dry_run.py --url http://localhost:9600 --token <owner key>

Checks the go-live bar in pilot/README.md: no errors, trap questions refused or
"not found", notification-status answers present, median answer time. It does NOT
judge correctness — rate the answers yourself in the app (Test missions tab).
"""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:9600')
    ap.add_argument('--token', required=True)
    ap.add_argument('--out', default='dry_run_report.csv')
    a = ap.parse_args()
    pack = json.loads((ROOT / 'pilot' / 'test_pack.json').read_text(encoding='utf-8'))
    rows, times, errors = [], [], []
    for m in pack['missions']:
        while True:
            r = httpx.post(f'{a.url}/api/ask', headers={'X-Tester-Token': a.token}, timeout=600,
                           json={'question': m['question'], 'mission_id': m['id']})
            if r.status_code != 429:
                break
            time.sleep(15)
        if r.status_code != 200:
            errors.append(m['id'])
            rows.append([m['id'], m['expected_behaviour'], f'ERROR {r.status_code}', '', '', '', r.text[:200]])
            print(f"{m['id']:<28} ERROR {r.status_code}")
            continue
        d = r.json()
        times.append(d['timings']['total_ms'] / 1000)
        chains = '; '.join(f"{c['key']}: {c['status']}" for c in d['chains'])
        rows.append([m['id'], m['expected_behaviour'], d['status'], d['confidence'],
                     round(d['timings']['total_ms'] / 1000, 1), chains, d['answer_markdown'][:500]])
        print(f"{m['id']:<28} {d['status']:<13} {d['confidence']:<7} {times[-1]:6.1f}s  {chains[:70]}")
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['mission', 'expected', 'status', 'confidence', 'seconds', 'notification_status', 'answer'])
        w.writerows(rows)
    traps = [r for r in rows if r[1] == 'refuse']
    traps_ok = sum(r[2] in ('not_found', 'out_of_scope') for r in traps)
    print(f'\nerrors: {len(errors)}   trap questions refused/not-found: {traps_ok}/{len(traps)}   '
          f'median time: {statistics.median(times):.0f}s' if times else '\nno successful answers')
    print(f'report: {a.out}  (now rate the answers in the app — this script does not judge correctness)')


if __name__ == '__main__':
    main()
