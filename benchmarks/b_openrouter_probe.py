#!/usr/bin/env python3
"""Probe OpenRouter keys: credits, available vision models, basic completion."""
import json, urllib.request
from pathlib import Path

ENV = Path("/mnt/d/_gpu_rig_ai/.env")
keys = {}
for line in ENV.read_text().splitlines():
    if "=" in line and line.startswith("OPENROUTER"):
        k, v = line.split("=", 1)
        keys[k] = v.strip()

def get(url, key):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def post(url, key, body):
    req = urllib.request.Request(url, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps(body).encode())
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

for name, key in keys.items():
    print(f"\n=== {name} ({key[:20]}...) ===")
    try:
        # Credits
        cred = get("https://openrouter.ai/api/v1/credits", key)
        print(f"  credits: {cred.get('data', cred)}")
    except Exception as e:
        print(f"  credits ERR: {e}")
    try:
        # Simple chat
        resp = post("https://openrouter.ai/api/v1/chat/completions", key, {
            "model": "google/gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 10,
        })
        msg = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        print(f"  chat OK: reply={msg!r} usage={usage}")
    except Exception as e:
        print(f"  chat ERR: {e}")
