#!/usr/bin/env bash
# smb_install_helper.sh — convenience wrapper for the SMB-only install workflow
#
# Codified rule (2026-05-09): rig has silent bit-flip corruption on local-disk
# writes >150MB. All large package installs MUST go through Windows-side
# download + SMB-mounted install. See `RULES_INDEX.md [TRIGGER: SMB workflow]`
# and `CLAUDE.md "RIG HARDWARE FAULT"` for full context.
#
# This script automates the verified install pattern. Run on Windows side first
# to download/extract, then on rig to install.
#
# Usage:
#   ./smb_install_helper.sh download <pkg> [<pkg2> ...]
#       (run on Windows) — pip download into D:/_gpu_rig_ai/tmp_downloads/wheels/
#   ./smb_install_helper.sh extract
#       (run on Windows) — unzip all wheels into wheels_extracted/ + sha256 manifest
#   ./smb_install_helper.sh install <venv-path>
#       (run on rig)     — copy_wheels_verified.py against the venv
#   ./smb_install_helper.sh verify <venv-path>
#       (run on rig)     — re-check sha256 of all installed files; report any corrupt

set -u

WIN_BASE="D:/_gpu_rig_ai/tmp_downloads"
RIG_BASE="/mnt/d/_gpu_rig_ai/tmp_downloads"
WHEELS_DIR="$WIN_BASE/wheels"
EXTRACTED_DIR="$WIN_BASE/wheels_extracted"
MANIFEST="$WIN_BASE/wheels_manifest.sha256"
COPY_SCRIPT="$WIN_BASE/copy_wheels_verified.py"

action="${1:-}"
shift || true

case "$action" in
  download)
    if [ $# -eq 0 ]; then
      echo "usage: $0 download <pkg1> [<pkg2> ...]" >&2; exit 1
    fi
    mkdir -p "$WHEELS_DIR"
    cd "$WHEELS_DIR" || exit 2
    # Windows pip path — assumes Python 3.12 default install
    PIP=$(ls /c/Users/*/AppData/Local/Programs/Python/Python312/Scripts/pip.exe 2>/dev/null | head -1)
    if [ -z "$PIP" ]; then echo "ERROR: pip not found"; exit 3; fi
    "$PIP" download --no-deps --platform manylinux2014_x86_64 \
      --python-version 310 --abi cp310 --implementation cp \
      --only-binary=:all: "$@"
    echo "downloaded to $WHEELS_DIR:"
    ls -lh *.whl | tail -20
    ;;

  extract)
    if [ ! -d "$WHEELS_DIR" ]; then echo "ERROR: $WHEELS_DIR missing"; exit 2; fi
    rm -rf "$EXTRACTED_DIR"
    mkdir -p "$EXTRACTED_DIR"
    cd "$EXTRACTED_DIR" || exit 2
    for w in "$WHEELS_DIR"/*.whl; do
      echo "extracting $(basename "$w")"
      python -c "import zipfile; zipfile.ZipFile('$w').extractall('.')"
    done
    echo "generating sha256 manifest..."
    find . -type f -exec sha256sum {} + > "$MANIFEST"
    echo "manifest: $(wc -l < "$MANIFEST") files in $EXTRACTED_DIR"
    ;;

  install)
    venv="${1:-}"
    if [ -z "$venv" ]; then echo "usage: $0 install <venv-path>"; exit 1; fi
    if [ ! -d "$RIG_BASE/wheels_extracted" ]; then
      echo "ERROR: $RIG_BASE/wheels_extracted not found — did you run 'extract' on Windows side?"; exit 2
    fi
    if [ ! -f "$RIG_BASE/wheels_manifest.sha256" ]; then
      echo "ERROR: $RIG_BASE/wheels_manifest.sha256 not found"; exit 2
    fi
    SITE_PKGS="$venv/lib/python3.10/site-packages"
    if [ ! -d "$SITE_PKGS" ]; then echo "ERROR: $SITE_PKGS missing"; exit 2; fi
    DST_BASE="$SITE_PKGS" /usr/bin/python3 "$COPY_SCRIPT"
    ;;

  verify)
    venv="${1:-}"
    if [ -z "$venv" ]; then echo "usage: $0 verify <venv-path>"; exit 1; fi
    /usr/bin/python3 - <<EOF
import sys, hashlib
from pathlib import Path
manifest = Path("$RIG_BASE/wheels_manifest.sha256")
venv_pkgs = Path("$venv/lib/python3.10/site-packages")
n_ok = n_bad = n_missing = 0
with manifest.open() as f:
    for line in f:
        parts = line.strip().split(None, 1)
        if len(parts) != 2: continue
        sha, rel = parts
        rel = rel.lstrip("./").lstrip("*")
        dst = venv_pkgs / rel
        if not dst.exists(): n_missing += 1; continue
        h = hashlib.sha256()
        with dst.open("rb", buffering=1024*1024) as fp:
            while True:
                b = fp.read(1024*1024)
                if not b: break
                h.update(b)
        if h.hexdigest() == sha: n_ok += 1
        else: n_bad += 1; print(f"BAD: {rel}")
print(f"\nok={n_ok} bad={n_bad} missing={n_missing}")
EOF
    ;;

  *)
    echo "Usage: $0 {download|extract|install|verify} [args]" >&2
    echo ""
    echo "Workflow:"
    echo "  Step 1 (Windows): $0 download torch sentence-transformers ..."
    echo "  Step 2 (Windows): $0 extract"
    echo "  Step 3 (rig):     $0 install /opt/training-venv-cuda"
    echo "  Step 4 (rig):     $0 verify /opt/training-venv-cuda"
    exit 1
    ;;
esac
