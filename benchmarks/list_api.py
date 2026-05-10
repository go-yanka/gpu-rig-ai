import urllib.request, json
d = json.loads(urllib.request.urlopen("http://127.0.0.1:9500/openapi.json", timeout=10).read())
for p, methods in d.get("paths", {}).items():
    for m, spec in methods.items():
        if m in ("get","post","put","delete"):
            print(f"  {m.upper():5} {p:42} {spec.get('summary','')}")
