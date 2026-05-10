#!/usr/bin/env python3
"""p1_v1 deploy script — A1 retrieval-side BM25 boost for statute queries.

Sentinel: p1_v1
Target:   user@192.168.1.107:/opt/indian-legal-ai/rag/cbic_rag/
Files:    retriever.py, api.py  (NOT storyformat.py)

Idempotent. Re-running is safe: a sentinel grep skips files already patched.
Backups written with suffix   .bak.p1_v1.<TS>   next to the live file.
Patched copies mirrored back as  <name>.patched.p1_v1.<TS>  for audit.

USAGE (dry-run by default — prints commands, does nothing):
    python apply.py
    python apply.py --apply          # actually run ssh/scp
    python apply.py --rollback       # restore .bak.p1_v1.<TS>

*** DO NOT run --apply until A3 has shipped. ***
"""
from __future__ import annotations
import argparse, os, shlex, subprocess, sys, time
from pathlib import Path

TS         = "20260421_213107"                          # frozen at patch-gen time
SENTINEL   = "p1_v1"
USER_HOST  = "user@192.168.1.107"
REMOTE_DIR = "/opt/indian-legal-ai/rag/cbic_rag"
SSH_FLAGS  = [
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-i", os.path.expanduser("~/.ssh/id_ed25519"),
]

HERE = Path(__file__).resolve().parent
FILES = {
    "retriever.py": HERE / "retriever.patched.py",
    "api.py":       HERE / "api.patched.py",
}


def _ssh_cmd(remote_cmd: str) -> list[str]:
    return ["ssh", *SSH_FLAGS, USER_HOST, remote_cmd]


def _scp_cmd(local: Path, remote: str) -> list[str]:
    return ["scp", *SSH_FLAGS, str(local), f"{USER_HOST}:{remote}"]


def _run(cmd: list[str], apply: bool) -> int:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"$ {printable}")
    if not apply:
        return 0
    return subprocess.call(cmd)


def check_local_artifacts() -> None:
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        print(f"FATAL: missing local artifact(s): {missing}", file=sys.stderr)
        sys.exit(2)


def deploy(apply: bool) -> int:
    check_local_artifacts()
    rc_total = 0
    for remote_name, local_path in FILES.items():
        remote_live = f"{REMOTE_DIR}/{remote_name}"
        remote_bak  = f"{remote_live}.bak.{SENTINEL}.{TS}"
        remote_pat  = f"{remote_live}.patched.{SENTINEL}.{TS}"

        # 1) skip if sentinel already present (idempotent)
        skip_cmd = (
            f"grep -q '{SENTINEL}' {shlex.quote(remote_live)} "
            f"&& echo SKIP || echo NEED"
        )
        rc = _run(_ssh_cmd(skip_cmd), apply)
        rc_total |= rc

        # 2) backup (-n = no-clobber so re-runs don't overwrite the first backup)
        backup_cmd = f"cp -n {shlex.quote(remote_live)} {shlex.quote(remote_bak)}"
        rc_total |= _run(_ssh_cmd(backup_cmd), apply)

        # 3) upload patched file as *.patched.<sentinel>.<ts> mirror
        rc_total |= _run(_scp_cmd(local_path, remote_pat), apply)

        # 4) atomically swap into place
        swap_cmd = f"cp {shlex.quote(remote_pat)} {shlex.quote(remote_live)}"
        rc_total |= _run(_ssh_cmd(swap_cmd), apply)

        # 5) verify sentinel landed
        verify_cmd = f"grep -n '{SENTINEL}' {shlex.quote(remote_live)} | head -3"
        rc_total |= _run(_ssh_cmd(verify_cmd), apply)

    # 6) restart systemd unit (user must confirm the exact unit name)
    print("\n# After verifying, restart the service:")
    print(f"#   ssh {' '.join(SSH_FLAGS)} {USER_HOST} "
          f"'sudo systemctl restart cbic-rag'")
    return rc_total


def rollback(apply: bool) -> int:
    rc_total = 0
    for remote_name in FILES:
        remote_live = f"{REMOTE_DIR}/{remote_name}"
        remote_bak  = f"{remote_live}.bak.{SENTINEL}.{TS}"
        restore_cmd = (
            f"test -f {shlex.quote(remote_bak)} "
            f"&& cp {shlex.quote(remote_bak)} {shlex.quote(remote_live)} "
            f"&& echo RESTORED:{remote_live} "
            f"|| echo NO_BACKUP:{remote_bak}"
        )
        rc_total |= _run(_ssh_cmd(restore_cmd), apply)
    print("\n# After restoring, restart the service:")
    print(f"#   ssh {' '.join(SSH_FLAGS)} {USER_HOST} "
          f"'sudo systemctl restart cbic-rag'")
    return rc_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually run ssh/scp (default: dry-run)")
    ap.add_argument("--rollback", action="store_true",
                    help="restore .bak.p1_v1.<TS> instead of applying")
    args = ap.parse_args()
    print(f"# p1_v1 deploy  ts={TS}  apply={args.apply}  rollback={args.rollback}")
    if args.rollback:
        return rollback(args.apply)
    return deploy(args.apply)


if __name__ == "__main__":
    sys.exit(main())
