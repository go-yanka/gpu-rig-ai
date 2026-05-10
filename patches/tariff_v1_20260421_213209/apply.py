"""
apply.py — deployment script for tariff_v1 patch. DO NOT EXECUTE from this file blindly;
read through and run steps manually, OR run `python apply.py --yes` once A3+A1 have shipped.

Effects on the rig (192.168.1.107):
  1. scp tariff_schema.sql, tariff_ingest.py, tariff_query.py, tariff_endpoint.py
     and gold_set_tariff.yaml to /opt/indian-legal-ai/rag/cbic_rag/
  2. sqlite3 /opt/indian-legal-ai/tariff.db < /opt/indian-legal-ai/rag/cbic_rag/tariff_schema.sql
  3. Patch api.py to register the tariff router (idempotent via grep guard).
  4. systemctl restart cbic-rag.service  (LEFT COMMENTED; user restarts manually)

SSH/SCP flags: -o ControlMaster=no -o ControlPath=none -i ~/.ssh/id_ed25519
Remote user@host: user@192.168.1.107
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = 'user@192.168.1.107'
REMOTE_DIR = '/opt/indian-legal-ai/rag/cbic_rag'
REMOTE_DB = '/opt/indian-legal-ai/tariff.db'
REMOTE_API = f'{REMOTE_DIR}/api.py'
SSH_FLAGS = '-o ControlMaster=no -o ControlPath=none -i ~/.ssh/id_ed25519'

FILES_TO_COPY = [
    'tariff_schema.sql',
    'tariff_ingest.py',
    'tariff_query.py',
    'tariff_endpoint.py',
    'gold_set_tariff.yaml',
]

# Idempotent import + include_router block appended to api.py.
# Guarded by `# BEGIN tariff_v1` sentinel; grep skips re-apply.
API_PATCH_SENTINEL = '# BEGIN tariff_v1'
API_PATCH_BLOCK = f"""\

{API_PATCH_SENTINEL}
# Added by tariff_v1 patch (A4). Bypasses RAG for HSN/SAC/rate queries.
from tariff_endpoint import router as tariff_router
app.include_router(tariff_router)
# END tariff_v1
"""

# Remote sed command to append the block if sentinel not present.
# Runs via ssh; uses a heredoc to avoid quote-hell.
REMOTE_APPLY_API_PATCH = (
    f"grep -q '{API_PATCH_SENTINEL}' {REMOTE_API} "
    f"|| cat >> {REMOTE_API} <<'PYEOF'\n{API_PATCH_BLOCK}PYEOF"
)


def sh(cmd: str, dry: bool) -> int:
    print(f"$ {cmd}")
    if dry:
        return 0
    return subprocess.call(cmd, shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yes', action='store_true',
                    help='Actually execute (default: dry-run / print only)')
    args = ap.parse_args()
    dry = not args.yes

    if dry:
        print('[apply.py] DRY-RUN — commands will be printed only. Pass --yes to execute.\n')

    # 1. scp files
    for f in FILES_TO_COPY:
        local = HERE / f
        if not local.exists():
            print(f"!! missing local file: {local}", file=sys.stderr)
            return 2
        sh(f"scp {SSH_FLAGS} {shlex.quote(str(local))} {REMOTE}:{REMOTE_DIR}/{f}", dry)

    # 2. create / migrate SQLite DB
    sh(
        f"ssh {SSH_FLAGS} {REMOTE} "
        f"'sqlite3 {REMOTE_DB} < {REMOTE_DIR}/tariff_schema.sql && "
        f"sqlite3 {REMOTE_DB} \"SELECT value FROM tariff_meta WHERE key=\\\"sentinel\\\";\"'",
        dry,
    )

    # 3. register router in api.py (idempotent)
    sh(f"ssh {SSH_FLAGS} {REMOTE} {shlex.quote(REMOTE_APPLY_API_PATCH)}", dry)

    # 4. restart service — LEFT COMMENTED, user decides when
    print('\n# Manual step (not executed): restart the RAG service')
    print(f"# ssh {SSH_FLAGS} {REMOTE} 'sudo systemctl restart cbic-rag.service'")

    # 5. smoke test
    print('\n# Smoke test (after restart):')
    print(
        f"# curl -s http://192.168.1.107:9500/v1/rate-query/health "
        f"# expect: {{\"sentinel\":\"tariff_v1\",\"status\":\"ok\"}}"
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
