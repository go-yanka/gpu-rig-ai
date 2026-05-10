#!/usr/bin/env python3
"""test_b5_b6_b7.py — T14-T19 for shadow/θ/snapshot components.

Covers:
  T14 divergence math (Jaccard)
  T15 kill-switch trip on sustained divergence
  T16 endpoint contract + /shadow/status shape
  T17 manual kill/resume
  T18 theta_tune.pick_theta feasibility + infeasibility
  T19 snapshot_v2.sh size-sanity + rotation logic (static shell-lint + manifest format)

Gate: all must pass before Stage B exit.
"""
from __future__ import annotations
import sys, os, json, subprocess, tempfile, re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cbic_rag"))

PASS, FAIL = 0, 0
def _t(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"[ ok ] {name}")
    else:
        FAIL += 1; print(f"[FAIL] {name} -- {msg}")


# --- T14 divergence math ----------------------------------------------------
def t14_divergence():
    # Stub imports to avoid FastAPI + retriever deps at test time
    os.environ.setdefault("SHADOW_LOG", str(HERE / "_shadow_test.jsonl"))
    os.environ.setdefault("THETA_V2_PATH", str(HERE / "_theta_test.json"))
    try:
        import api_v2_shadow as shadow
    except Exception as e:
        _t("T14 import api_v2_shadow", False, str(e))
        return

    d = shadow._jaccard_divergence(["a","b","c"], ["a","b","c"])
    _t("T14.a identical → divergence 0", d == 0.0)

    d = shadow._jaccard_divergence(["a","b","c"], ["x","y","z"])
    _t("T14.b disjoint → divergence 1.0", d == 1.0)

    d = shadow._jaccard_divergence(["a","b","c","d"], ["a","b","e","f"])
    # 2 overlap / max(4,4)=4 → 1 - 0.5 = 0.5
    _t("T14.c half-overlap → 0.5", abs(d - 0.5) < 1e-6, f"got {d}")

    d = shadow._jaccard_divergence([], [])
    _t("T14.d both empty → 0.0", d == 0.0)


# --- T15 kill-switch trip ---------------------------------------------------
def t15_killswitch():
    import api_v2_shadow as shadow
    # Reset state
    shadow._recent.clear()
    shadow._kill_switch.update({"tripped": False, "reason": None, "ts": None})
    # simulate KILL_WINDOW+1 high-divergence requests
    for _ in range(shadow.KILL_WINDOW):
        shadow._record_divergence(0.5, {"q":"test"})
    _t("T15.a kill-switch tripped after window of high divergence",
       shadow._kill_switch["tripped"] is True,
       f"state={shadow._kill_switch}")

    # Reset: low divergence should NOT trip
    shadow._recent.clear()
    shadow._kill_switch.update({"tripped": False, "reason": None, "ts": None})
    for _ in range(shadow.KILL_WINDOW):
        shadow._record_divergence(0.01, {"q":"test"})  # below 2% threshold
    _t("T15.b kill-switch NOT tripped on low divergence",
       shadow._kill_switch["tripped"] is False)


# --- T16 endpoint contract / status shape -----------------------------------
def t16_status_shape():
    import api_v2_shadow as shadow
    r = shadow.shadow_status()
    expected = {"kill_switch", "window_size", "window_capacity",
                "rolling_avg_divergence", "threshold", "stats", "theta_v2"}
    missing = expected - set(r)
    _t("T16.a shadow_status has all keys", not missing, f"missing {missing}")
    _t("T16.b threshold matches env default",
       abs(r["threshold"] - shadow.DIVERGENCE_THRESHOLD) < 1e-9)


# --- T17 manual kill/resume -------------------------------------------------
def t17_manual_kill():
    import api_v2_shadow as shadow
    shadow.shadow_resume()
    r = shadow.shadow_kill(reason="unit-test")
    _t("T17.a manual kill tripped", r["tripped"] is True)
    r = shadow.shadow_resume()
    _t("T17.b resume clears",
       r["tripped"] is False and not shadow._recent)


# --- T18 theta_tune.pick_theta ---------------------------------------------
def t18_pick_theta():
    from theta_tune import pick_theta

    # Well-separated distributions → feasible θ exists
    gold = [0.8, 0.82, 0.85, 0.88, 0.9] * 20   # n=100, all high
    adv = [0.2, 0.25, 0.3, 0.35, 0.4] * 10     # n=50, all low
    best, diag = pick_theta(gold, adv, 0.95, 0.9)
    _t("T18.a separated distros → feasible", best is not None)
    if best:
        _t("T18.b θ between adv_max and gold_min",
           max(adv) <= best["theta"] <= min(gold) + 1e-6,
           f"θ={best['theta']} adv_max={max(adv)} gold_min={min(gold)}")
        _t("T18.c achieved gold recall >= target", best["gold_recall"] >= 0.95)
        _t("T18.d achieved adv refuse >= target", best["adv_refuse"] >= 0.9)

    # Overlapping distributions → infeasible
    gold = [0.5, 0.55, 0.6, 0.65] * 20
    adv = [0.55, 0.6, 0.65, 0.7] * 10
    best, diag = pick_theta(gold, adv, 0.95, 0.9)
    _t("T18.e overlapping distros → infeasible", best is None)

    # Empty input
    best, diag = pick_theta([], [], 0.95, 0.9)
    _t("T18.f empty input → infeasible", best is None)


# --- T19 snapshot_v2.sh static checks ---------------------------------------
def t19_snapshot_script():
    sh = HERE / "snapshot_v2.sh"
    src = sh.read_text(encoding="utf-8")

    # Must have rotation
    _t("T19.a has RETAIN_DAYS rotation",
       "RETAIN_DAYS" in src and "-mtime" in src)

    # Must have size sanity
    _t("T19.b has size sanity (MIN_BYTES)",
       "MIN_BYTES" in src and "exit 3" in src)

    # Must compare to previous
    _t("T19.c has prev-size 50% corruption check",
       "prev_size" in src and "50%" in src.lower() or "prev_size / 2" in src)

    # Must write manifest row
    _t("T19.d writes manifest.jsonl", "manifest.jsonl" in src)

    # Exit codes documented
    _t("T19.e exit codes documented in header",
       "exit 2" in src and "exit 3" in src and "set -eu" in src)

    # Quick shellcheck-lite: no obvious unquoted $COLL
    bad = re.findall(r'rm\s+-rf\s+\$[A-Z_]+/', src)
    _t("T19.f no unquoted rm -rf $VAR/", not bad, f"bad lines: {bad}")


if __name__ == "__main__":
    t14_divergence()
    t15_killswitch()
    t16_status_shape()
    t17_manual_kill()
    t18_pick_theta()
    t19_snapshot_script()
    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(0 if FAIL == 0 else 1)
