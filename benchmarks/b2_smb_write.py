"""
B2: SMB concurrent write correctness test.

Windows (192.168.1.222) shares D:\ as 'projects'.
Rig (192.168.1.107) mounts it as /mnt/d (cifs vers=3.0).

Test dir (both views of the same directory):
  Windows: D:\_gpu_rig_ai\benchmarks\b2_test\
  Rig:     /mnt/d/_gpu_rig_ai/benchmarks/b2_test/

Pattern under test: one file per host, no shared-append.
  parsed_win.jsonl  <- written by Windows
  parsed_rig.jsonl  <- written by rig

Pass: both files have exactly the intended line counts, every line parses
as JSON, no file contains the other host's tag string (interleave check),
and per-host i counter is strictly monotonically increasing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

DURATION_SEC = 60
INTERVAL_SEC = 0.060   # ~60 ms -> ~1000 lines / host
PAD = "x" * 500

WIN_DIR = Path(r"D:\_gpu_rig_ai\benchmarks\b2_test")
WIN_FILE = WIN_DIR / "parsed_win.jsonl"
RIG_FILE_WIN_VIEW = WIN_DIR / "parsed_rig.jsonl"
RIG_DIR_RIG_VIEW = "/mnt/d/_gpu_rig_ai/benchmarks/b2_test"
RIG_FILE_RIG_VIEW = f"{RIG_DIR_RIG_VIEW}/parsed_rig.jsonl"

RESULTS = Path(r"D:\_gpu_rig_ai\benchmarks\b2_results.json")


def win_writer(start_at: float, stop_at: float) -> int:
    """Append JSON lines from Windows until stop_at wall clock."""
    n = 0
    # Sleep until barrier
    while time.time() < start_at:
        time.sleep(0.001)
    with open(WIN_FILE, "a", encoding="utf-8") as f:
        next_tick = start_at
        while time.time() < stop_at:
            rec = {"host": "win", "i": n, "ts": time.time(), "pad": PAD}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            next_tick += INTERVAL_SEC
            slack = next_tick - time.time()
            if slack > 0:
                time.sleep(slack)
    return n


# Script run on the rig. Uses /mnt/d CIFS mount.
RIG_SCRIPT = r'''
import json, time, sys
PATH = "{path}"
DURATION = {dur}
INTERVAL = {iv}
START_AT = {start}
PAD = "y" * 500
n = 0
while time.time() < START_AT:
    time.sleep(0.001)
stop_at = START_AT + DURATION
with open(PATH, "a", encoding="utf-8") as f:
    next_tick = START_AT
    while time.time() < stop_at:
        rec = {{"host": "rig", "i": n, "ts": time.time(), "pad": PAD}}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        n += 1
        next_tick += INTERVAL
        slack = next_tick - time.time()
        if slack > 0:
            time.sleep(slack)
print("RIG_WROTE", n)
'''


def sync_clock_offset() -> float:
    """Return (rig_time - win_time) in seconds, roughly. Not super precise
    but enough to pick a common start moment."""
    t0 = time.time()
    out = subprocess.run(
        ["ssh", "-o", "ControlMaster=no", "user@192.168.1.107",
         "python3 -c 'import time;print(repr(time.time()))'"],
        capture_output=True, text=True, timeout=15,
    )
    t1 = time.time()
    if out.returncode != 0:
        raise RuntimeError(f"ssh clock probe failed: {out.stderr}")
    rig_t = float(out.stdout.strip())
    win_mid = (t0 + t1) / 2.0
    return rig_t - win_mid


def main() -> int:
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh test: remove any old files first
    for p in (WIN_FILE, RIG_FILE_WIN_VIEW):
        if p.exists():
            p.unlink()

    offset = sync_clock_offset()
    print(f"[b2] rig clock offset vs win: {offset:+.3f}s")

    # Pick a start moment 5s in the future on Windows wall clock
    win_start = time.time() + 5.0
    rig_start = win_start + offset  # same wall instant, expressed in rig clock
    stop_at = win_start + DURATION_SEC

    script = RIG_SCRIPT.format(
        path=RIG_FILE_RIG_VIEW,
        dur=DURATION_SEC,
        iv=INTERVAL_SEC,
        start=repr(rig_start),
    )
    print(f"[b2] launching rig writer; start in ~5s, duration {DURATION_SEC}s")
    rig_proc = subprocess.Popen(
        ["ssh", "-o", "ControlMaster=no", "user@192.168.1.107",
         "python3 -u -c " + _shq(script)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    win_count_box = {"n": 0}

    def _run_win():
        win_count_box["n"] = win_writer(win_start, stop_at)

    th = threading.Thread(target=_run_win)
    th.start()
    th.join(timeout=DURATION_SEC + 30)

    try:
        rig_stdout, rig_stderr = rig_proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        rig_proc.kill()
        rig_stdout, rig_stderr = rig_proc.communicate()

    print(f"[b2] rig stdout: {rig_stdout.strip()}")
    if rig_stderr.strip():
        print(f"[b2] rig stderr: {rig_stderr.strip()}")

    rig_reported = 0
    for line in rig_stdout.splitlines():
        if line.startswith("RIG_WROTE"):
            rig_reported = int(line.split()[1])

    win_n = win_count_box["n"]
    print(f"[b2] win wrote {win_n} lines, rig reported {rig_reported}")

    # Give CIFS a moment to flush
    time.sleep(2)

    report = verify(win_n, rig_reported)
    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


def _shq(s: str) -> str:
    # Single-quote-escape for ssh remote shell
    return "'" + s.replace("'", "'\"'\"'") + "'"


def verify(win_wrote: int, rig_wrote: int) -> dict:
    notes: list[str] = []
    win_lines = WIN_FILE.read_text(encoding="utf-8").splitlines() if WIN_FILE.exists() else []
    rig_lines = RIG_FILE_WIN_VIEW.read_text(encoding="utf-8").splitlines() if RIG_FILE_WIN_VIEW.exists() else []

    parse_fail = 0
    interleaves = 0

    def _check(lines, expect_host, other_host):
        nonlocal parse_fail, interleaves
        prev_i = -1
        mono_ok = True
        for ln in lines:
            # Interleave check: does this line contain the other host's tag?
            if f'"host":"{other_host}"' in ln or f'"host": "{other_host}"' in ln:
                interleaves += 1
            try:
                obj = json.loads(ln)
            except Exception:
                parse_fail += 1
                continue
            if obj.get("host") != expect_host:
                interleaves += 1
            i = obj.get("i", -1)
            if i <= prev_i:
                mono_ok = False
            prev_i = i
        return mono_ok

    win_mono = _check(win_lines, "win", "rig")
    rig_mono = _check(rig_lines, "rig", "win")

    win_ok = len(win_lines) == win_wrote and win_wrote > 0
    rig_ok = len(rig_lines) == rig_wrote and rig_wrote > 0

    if not win_ok:
        notes.append(f"win line count mismatch: file={len(win_lines)} wrote={win_wrote}")
    if not rig_ok:
        notes.append(f"rig line count mismatch: file={len(rig_lines)} wrote={rig_wrote}")
    if not win_mono:
        notes.append("win i counter not strictly increasing")
    if not rig_mono:
        notes.append("rig i counter not strictly increasing")
    if interleaves:
        notes.append(f"{interleaves} interleave markers detected")
    if parse_fail:
        notes.append(f"{parse_fail} JSON parse failures")

    overall = win_ok and rig_ok and win_mono and rig_mono and interleaves == 0 and parse_fail == 0

    return {
        "pass": overall,
        "win_lines": len(win_lines),
        "rig_lines": len(rig_lines),
        "win_wrote": win_wrote,
        "rig_wrote": rig_wrote,
        "interleaves": interleaves,
        "json_parse_failures": parse_fail,
        "win_monotonic": win_mono,
        "rig_monotonic": rig_mono,
        "share_path": r"\\192.168.1.222\projects  (D:\ on Windows, /mnt/d CIFS vers=3.0 on rig)",
        "test_dir": r"D:\_gpu_rig_ai\benchmarks\b2_test (== /mnt/d/_gpu_rig_ai/benchmarks/b2_test)",
        "duration_sec": DURATION_SEC,
        "interval_sec": INTERVAL_SEC,
        "notes": "; ".join(notes) if notes else "clean",
    }


if __name__ == "__main__":
    sys.exit(main())
