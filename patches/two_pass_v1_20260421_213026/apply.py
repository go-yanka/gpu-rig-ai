#!/usr/bin/env python3
"""Idempotent deploy script for A3 two-pass (sentinel: two_pass_v1).

DO NOT RUN automatically — user deploys manually after baseline eval lands.

What it does (in order, all via ssh to 192.168.1.107):
  1. Sanity-check that rig paths exist.
  2. For each of api.py, storyformat.py:
       - sha256 the current rig file
       - copy it to <file>.bak.two_pass_v1.<TS>   (unless that backup already exists)
  3. Upload api.patched.py    -> <rig>/api.py
     Upload storyformat.patched.py -> <rig>/storyformat.py
     Upload validator.py      -> <rig>/validator.py   (new file)
     Also leaves <file>.patched.two_pass_v1.<TS> sidecars on the rig so you can
     diff/restore at any time.
  4. Prints the exact systemctl restart command + smoke-test curl — but does
     NOT restart the service. Operator restarts manually.

Re-running is safe: backups are only created if missing; upload is straight
overwrite (content is identical on a repeat run). The feature flag
TWO_PASS_ENABLED defaults to 0 on the live box, so this deploy is a no-op in
terms of behavior until the operator exports TWO_PASS_ENABLED=1 and restarts.

SSH notes:
  - Uses ControlMaster=no / ControlPath=none to avoid the rig's shared-socket
    quirks.
  - Key: ~/.ssh/id_ed25519. User: user. Host: 192.168.1.107.
"""
from __future__ import annotations
import os, sys, subprocess, shlex, hashlib, pathlib, time

TS         = "20260421_213026"
SENTINEL   = "two_pass_v1"
RIG_HOST   = "user@192.168.1.107"
RIG_DIR    = "/opt/indian-legal-ai/rag/cbic_rag"
SSH_KEY    = os.path.expanduser("~/.ssh/id_ed25519")
SSH_OPTS   = [
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-i", SSH_KEY,
]

HERE = pathlib.Path(__file__).resolve().parent

# Map of local-patch-file -> rig-basename
UPLOADS = {
    "api.patched.py":         "api.py",
    "storyformat.patched.py": "storyformat.py",
    "validator.py":           "validator.py",   # new file on rig
}
# Files we back up (existing files being overwritten)
BACKUP_TARGETS = ["api.py", "storyformat.py"]


def _ssh(cmd: str, check: bool = True) -> tuple[int, str, str]:
    full = ["ssh", *SSH_OPTS, RIG_HOST, cmd]
    p = subprocess.run(full, capture_output=True, text=True)
    if check and p.returncode != 0:
        print(f"[ssh FAIL] {cmd}\n  stderr: {p.stderr.strip()}")
        sys.exit(2)
    return p.returncode, p.stdout, p.stderr


def _scp(local: pathlib.Path, remote: str) -> None:
    full = ["scp", *SSH_OPTS, str(local), f"{RIG_HOST}:{remote}"]
    p = subprocess.run(full, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"[scp FAIL] {local} -> {remote}\n  stderr: {p.stderr.strip()}")
        sys.exit(3)


def main() -> None:
    print(f"[two_pass_v1] deploy script  ts={TS}  dir={HERE}")

    # 0) verify local patch files exist
    for local_name in UPLOADS:
        p = HERE / local_name
        if not p.is_file():
            print(f"[FATAL] missing local patch file: {p}")
            sys.exit(1)

    # 1) sanity-check rig dir
    rc, out, _ = _ssh(f"test -d {shlex.quote(RIG_DIR)} && echo OK")
    if "OK" not in out:
        print(f"[FATAL] rig dir missing: {RIG_DIR}")
        sys.exit(4)

    # 2) backups (idempotent — only create if not already present)
    for base in BACKUP_TARGETS:
        rig_path = f"{RIG_DIR}/{base}"
        bak_path = f"{rig_path}.bak.{SENTINEL}.{TS}"
        rc, out, _ = _ssh(f"test -f {shlex.quote(bak_path)} && echo EXISTS || echo MISSING")
        if "EXISTS" in out:
            print(f"[backup] already exists, skipping: {bak_path}")
        else:
            # also ensure the source exists before copying
            rc2, out2, _ = _ssh(f"test -f {shlex.quote(rig_path)} && echo OK || echo NO")
            if "NO" in out2:
                print(f"[WARN] rig file missing, cannot back up: {rig_path}")
                continue
            _ssh(f"cp -p {shlex.quote(rig_path)} {shlex.quote(bak_path)}")
            print(f"[backup] {rig_path} -> {bak_path}")

    # 3) uploads
    for local_name, rig_base in UPLOADS.items():
        local_path = HERE / local_name
        rig_path   = f"{RIG_DIR}/{rig_base}"
        sidecar    = f"{rig_path}.patched.{SENTINEL}.{TS}"
        # sha256 local
        h = hashlib.sha256(local_path.read_bytes()).hexdigest()
        print(f"[upload] {local_name}  sha256={h[:16]}  -> {rig_path}")
        _scp(local_path, rig_path)
        # also drop a sidecar copy so you can diff later
        _ssh(f"cp -p {shlex.quote(rig_path)} {shlex.quote(sidecar)}")

    # 4) verify sentinel present in the uploaded api.py
    rc, out, _ = _ssh(f"grep -c {shlex.quote(SENTINEL)} {shlex.quote(RIG_DIR + '/api.py')}")
    if out.strip().isdigit() and int(out.strip()) >= 1:
        print(f"[verify] sentinel {SENTINEL} present in rig api.py ({out.strip()} occurrences)")
    else:
        print(f"[verify WARN] sentinel not found — upload may have failed silently")

    print()
    print("=" * 72)
    print("DEPLOY WRITE PHASE COMPLETE. Nothing restarted. Nothing enabled.")
    print("=" * 72)
    print("Next steps (run manually on the rig):")
    print("  1. Start with flag OFF to confirm zero regression:")
    print("       sudo systemctl restart cbic-rag")
    print("       # hit /query — behavior must be byte-identical to pre-deploy")
    print()
    print("  2. Flip the flag ON:")
    print("       sudo systemctl edit cbic-rag   # add Environment=TWO_PASS_ENABLED=1")
    print("       sudo systemctl daemon-reload && sudo systemctl restart cbic-rag")
    print()
    print("  3. Rollback (any time):")
    print(f"       ssh {RIG_HOST} 'cp {RIG_DIR}/api.py.bak.{SENTINEL}.{TS} {RIG_DIR}/api.py && "
          f"cp {RIG_DIR}/storyformat.py.bak.{SENTINEL}.{TS} {RIG_DIR}/storyformat.py && "
          f"sudo systemctl restart cbic-rag'")
    print()


if __name__ == "__main__":
    main()
