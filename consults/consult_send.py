#!/usr/bin/env python3
"""Send CBIC consult package to Gemini 2.5 Pro (and stub for GPT-5).

Usage:
    python consult_send.py [path_to_package.md]

Reads GEMINI_API_KEY from .env or environment. Writes responses to
./responses/<model>_<timestamp>.md next to the package.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = ROOT / "consult_package_20260422.md"

SYSTEM_PROMPT = (
    "You are a senior ML engineer reviewing a retrieval-system improvement "
    "plan. Give sharp, opinionated critique. Challenge assumptions. Suggest "
    "concrete alternatives with numbers. Under 2000 words."
)


def load_env() -> None:
    """Load .env from project root if present."""
    for candidate in (ROOT / ".env", ROOT.parent / ".env", Path("D:/_gpu_rig_ai/.env")):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def send_gemini(package_text: str, api_key: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-pro:generateContent?key={api_key}"
    )
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": package_text}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 30000,
            "responseMimeType": "text/plain",
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        return f"<ERROR parsing Gemini response: {e}>\n\n{json.dumps(payload, indent=2)}"


def send_openai(package_text: str, api_key: str) -> str:
    """TODO — requires OPENAI_API_KEY. Stub.

    When implementing:
      - model: gpt-5
      - endpoint: https://api.openai.com/v1/chat/completions (or /v1/responses)
      - system role = SYSTEM_PROMPT, user role = package_text
      - temperature 0.2, max_tokens 30000
    """
    raise NotImplementedError("OpenAI path not wired yet — set OPENAI_API_KEY and implement.")


def main() -> int:
    load_env()
    pkg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PACKAGE
    if not pkg_path.exists():
        print(f"Package not found: {pkg_path}", file=sys.stderr)
        return 2
    package_text = pkg_path.read_text(encoding="utf-8")
    print(f"Package: {pkg_path} ({len(package_text):,} chars, ~{len(package_text.split()):,} words)")

    out_dir = ROOT / "responses"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # --- Gemini 2.5 Pro ---
    gem_key = os.environ.get("GEMINI_API_KEY")
    if not gem_key:
        print("GEMINI_API_KEY not set — skipping Gemini.", file=sys.stderr)
    else:
        print("Sending to Gemini 2.5 Pro...")
        try:
            reply = send_gemini(package_text, gem_key)
            out = out_dir / f"gemini-2.5-pro_{ts}.md"
            out.write_text(reply, encoding="utf-8")
            print(f"  -> {out} ({len(reply.split()):,} words)")
        except Exception as e:  # noqa: BLE001
            print(f"  !! Gemini failed: {e}", file=sys.stderr)

    # --- GPT-5 (stub) ---
    oai_key = os.environ.get("OPENAI_API_KEY")
    if not oai_key:
        print("OPENAI_API_KEY not set — GPT-5 path is a TODO stub.")
    else:
        try:
            reply = send_openai(package_text, oai_key)
            out = out_dir / f"gpt-5_{ts}.md"
            out.write_text(reply, encoding="utf-8")
            print(f"  -> {out}")
        except NotImplementedError as e:
            print(f"  !! GPT-5: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
