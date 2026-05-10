# CBIC RAG — Public Access via Cloudflare Quick-Tunnel

Set up 2026-04-21. Night-shift agent was concurrently patching `cbic-rag-api`;
this setup installed tunnel + auth scaffolding only — api.py was NOT modified.

## Current tunnel URL

    https://mileage-demographic-duplicate-oven.trycloudflare.com

Verified: `/health` -> 200, `/ui` -> 307 redirect (UI reachable).
Note: `/status` is not an endpoint on this build — use `/health` for probes.

## Tester tokens

| Identity          | Token                              |
|-------------------|------------------------------------|
| tester1@cbic.ai   | `sknnZTvmwiuAucxTMzzQqGz6uft4wIIB` |
| tester2@cbic.ai   | `npVHxN4t8f8gxQyMGPGnvWuwOuEljv_z` |
| tester3@cbic.ai   | `b1FcoMoF04KEH5TRuLXErSaOQUA0-gAM` |

Stored at `/opt/cbic-auth/tokens.json` (chmod 600, root-owned).

## Sample tester URLs (after auth middleware is wired)

    https://mileage-demographic-duplicate-oven.trycloudflare.com/ui?t=sknnZTvmwiuAucxTMzzQqGz6uft4wIIB
    https://mileage-demographic-duplicate-oven.trycloudflare.com/ui?t=npVHxN4t8f8gxQyMGPGnvWuwOuEljv_z
    https://mileage-demographic-duplicate-oven.trycloudflare.com/ui?t=b1FcoMoF04KEH5TRuLXErSaOQUA0-gAM

Testers can also pass `X-Tester-Token: <token>` header.

## Task status

| # | Task                                              | Status                    |
|---|---------------------------------------------------|---------------------------|
| 1 | Install cloudflared on rig                        | DONE (v2026.3.0)          |
| 2 | cloudflared-cbic.service (Restart=always)         | DONE, active              |
| 3 | Generate 3 tester tokens -> /opt/cbic-auth/tokens.json | DONE (chmod 600)     |
| 4 | /opt/cbic-auth/middleware.py (auth)               | WRITTEN, not yet wired    |
| 4 | /opt/cbic-auth/HOW_TO_WIRE.md                     | DONE                      |
| 5 | /opt/cbic-auth/ratelimit_middleware.py (10/min)   | WRITTEN, not yet wired    |
| 6 | /opt/cbic-auth/CONCURRENCY_CAP.md                 | DONE (docs only)          |
| 7 | Verify tunnel reaches api (/health=200)           | PASS                      |
| 7 | Confirm api.py untouched                          | PASS (grep: 0 matches)    |
| 7 | cbic-rag-api still running                        | active (no restart done)  |

## Morning TO-DO — wire auth middleware (single service restart)

1. SSH to rig: `ssh rig`
2. Expose /opt/cbic-auth on PYTHONPATH:
       sudo ln -s /opt/cbic-auth /opt/indian-legal-ai/rag/cbic_auth
3. Edit `/opt/indian-legal-ai/rag/cbic_rag/api.py`, add near the other
   `app.add_middleware(...)` calls:

       from cbic_auth.middleware import TesterAuthMiddleware
       from cbic_auth.ratelimit_middleware import RateLimitMiddleware

       app.add_middleware(RateLimitMiddleware)
       app.add_middleware(TesterAuthMiddleware)   # outermost = evaluated first

4. Restart once:
       sudo systemctl restart cbic-rag-api
5. Smoke test:
       curl -s -o /dev/null -w '%{http_code}\n' https://mileage-demographic-duplicate-oven.trycloudflare.com/health   # expect 200
       curl -s -o /dev/null -w '%{http_code}\n' https://mileage-demographic-duplicate-oven.trycloudflare.com/ui       # expect 401
       curl -s -o /dev/null -w '%{http_code}\n' 'https://mileage-demographic-duplicate-oven.trycloudflare.com/ui?t=sknnZTvmwiuAucxTMzzQqGz6uft4wIIB'  # expect 200/307
6. Tail the audit log:
       sudo tail -f /var/log/cbic-auth.log

See `/opt/cbic-auth/HOW_TO_WIRE.md` on the rig for the canonical copy.

## How to rotate a token

    ssh rig
    NEW=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')
    sudo python3 -c "import json,sys; p='/opt/cbic-auth/tokens.json'; d=json.load(open(p)); d['tester2@cbic.ai']='$NEW'; json.dump(d,open(p,'w'),indent=2)"
    sudo systemctl restart cbic-rag-api   # middleware re-reads tokens on startup

Then hand the new token to that tester. Old token is instantly invalid after
restart.

## How to revoke tester access

    ssh rig
    sudo python3 -c "import json; p='/opt/cbic-auth/tokens.json'; d=json.load(open(p)); d.pop('tester3@cbic.ai',None); json.dump(d,open(p,'w'),indent=2)"
    sudo systemctl restart cbic-rag-api

## Complete teardown (reversible)

    sudo systemctl disable --now cloudflared-cbic
    sudo rm /etc/systemd/system/cloudflared-cbic.service
    sudo systemctl daemon-reload
    sudo rm -rf /opt/cbic-auth
    sudo apt remove -y cloudflared

## Caveats

- **URL changes on every cloudflared-cbic restart.** Quick-tunnels are
  ephemeral. If the rig reboots or the service restarts, re-run:
      ssh rig "sudo journalctl -u cloudflared-cbic -n 100 --no-pager | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1"
  and redistribute the new URL.
- For a **stable URL**, buy a domain (~Rs 600/yr for a .com at Namecheap,
  GoDaddy, or BigRock — Cloudflare's 1.1.1.1 is a DNS resolver, not a
  registrar, though Cloudflare Registrar sells at wholesale cost once you
  have a zone on their free plan). Then switch to a **named tunnel** via
  `cloudflared tunnel create` + `cloudflared tunnel route dns`. That also
  unlocks real rate-limiting and Cloudflare Access (SSO).
- Current setup has **no rate limit active** until middleware is wired in
  the morning. 3 testers is low risk for one night.
- Audit log lives at `/var/log/cbic-auth.log` once middleware is active.
