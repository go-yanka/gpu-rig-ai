#!/usr/bin/env python3
"""Playwright v2: real page navigation + search, with response-capture.
- Navigate to indiacode simple-search, get first result
- Navigate to handle page, find PDF bitstream link
- Use response listener to capture PDF bytes when browser follows JS challenge
- Inline QC before save
"""
import os, asyncio, fitz, re, urllib.parse
from playwright.async_api import async_playwright

STAGE = "/opt/indian-legal-ai/gst_stage"

JOBS = [
    ("general_clauses_1897", "general clauses act 1897", "general clauses act", "t1_interpretation/general_clauses_1897.pdf"),
    ("indian_succession_1925", "indian succession act 1925", "indian succession act", "t1_personal_law/indian_succession_1925.pdf"),
    ("special_marriage_1954", "special marriage act 1954", "special marriage act", "t1_personal_law/special_marriage_1954.pdf"),
    ("muslim_dissolution_1939", "dissolution of muslim marriages act 1939", "dissolution of muslim marriages", "t1_personal_law/muslim_dissolution_1939.pdf"),
    ("shariat_1937", "muslim personal law shariat application act 1937", "shariat", "t1_personal_law/shariat_application_1937.pdf"),
    ("indian_divorce_1869", "indian divorce act 1869", "indian divorce act", "t1_personal_law/indian_divorce_1869.pdf"),
    ("indian_trusts_1882", "indian trusts act 1882", "indian trusts act", "t1_other_bare_acts/indian_trusts_1882.pdf"),
    ("indian_easements_1882", "indian easements act 1882", "indian easements act", "t1_other_bare_acts/indian_easements_1882.pdf"),
    ("insurance_1938", "insurance act 1938", "insurance act, 1938", "t1_banking/insurance_act_1938.pdf"),
    ("benami_1988", "benami transactions prohibition 1988", "benami", "t1_banking/benami_1988.pdf"),
    ("customs_1962", "customs act 1962", "customs act, 1962", "t1_customs_excise/customs_act_1962.pdf"),
    ("central_excise_1944", "central excise act 1944", "central excise act", "t1_customs_excise/central_excise_1944.pdf"),
    ("trade_marks_1999", "trade marks act 1999", "trade marks act", "t1_ip_acts/trade_marks_1999.pdf"),
    ("pc_act_1988", "prevention of corruption act 1988", "prevention of corruption", "t1_criminal_special/pc_act_1988.pdf"),
    ("ndps_1985", "narcotic drugs psychotropic substances 1985", "narcotic drugs", "t1_criminal_special/ndps_1985.pdf"),
    ("arms_1959", "arms act 1959", "arms act", "t1_criminal_special/arms_1959.pdf"),
    ("dpdp_2023", "digital personal data protection act 2023", "digital personal data protection", "t1_data_privacy/dpdp_2023.pdf"),
    ("competition_2002", "competition act 2002", "competition act", "t1_other_bare_acts/competition_2002.pdf"),
    ("environment_1986", "environment protection act 1986", "environment (protection) act", "t1_environment/environment_1986.pdf"),
    ("wildlife_1972", "wild life protection act 1972", "wild life", "t1_environment/wildlife_1972.pdf"),
    ("industrial_disputes_1947", "industrial disputes act 1947", "industrial disputes act", "t1_pre_codified_labour/industrial_disputes_1947.pdf"),
    ("factories_1948", "factories act 1948", "factories act", "t1_pre_codified_labour/factories_1948.pdf"),
    ("min_wages_1948", "minimum wages act 1948", "minimum wages act", "t1_pre_codified_labour/minimum_wages_1948.pdf"),
    ("gratuity_1972", "payment of gratuity act 1972", "payment of gratuity", "t1_pre_codified_labour/gratuity_1972.pdf"),
]

BAD = ["amendment bill","written answer","jstor","research paper","papers laid",
       "digital copy of a book","cornell university","all india christian council",
       "leave of absence"]

def qc(content, kw):
    if len(content) < 1024: return False, f"too-small ({len(content)}B)", {}
    if content[:4] != b"%PDF": return False, f"not-pdf ({content[:20]!r})", {}
    try:
        d = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        return False, f"parse-err: {e}", {}
    pg = d.page_count
    if pg < 4:
        d.close(); return False, f"stub ({pg}p)", {"pg": pg}
    tot = sum(len(d.load_page(i).get_text()) for i in range(pg))
    cpp = tot // max(pg,1)
    if cpp < 200:
        d.close(); return False, f"scanned ({cpp}c/p)", {"pg": pg, "cpp": cpp}
    first = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
    d.close()
    tl = first.lower()[:8000]
    title = first[:160].replace("\n"," ").strip()
    if kw.lower() not in tl:
        return False, f"kw-miss '{kw}'", {"pg": pg, "cpp": cpp, "title": title[:80]}
    for b in BAD:
        if b in tl: return False, f"bad '{b}'", {"pg": pg}
    return True, "ok", {"pg": pg, "cpp": cpp, "size_kb": len(content)//1024, "title": title[:100]}


async def fetch_one(browser, label, search, kw, rel):
    """Returns True if saved."""
    ctx = await browser.new_context(
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36",
        accept_downloads=True,
    )
    page = await ctx.new_page()

    pdf_bytes = {"data": None}

    async def on_resp(resp):
        if pdf_bytes["data"]: return
        try:
            ct = (resp.headers or {}).get("content-type","").lower()
            url = resp.url
            if (".pdf" in url or "pdf" in ct) and resp.status == 200:
                body = await resp.body()
                if body and body[:4] == b"%PDF":
                    pdf_bytes["data"] = body
        except: pass

    page.on("response", on_resp)

    try:
        # Step 1: search
        q = urllib.parse.quote_plus(search)
        search_url = f"https://www.indiacode.nic.in/simple-search?query={q}&btngo=Go"
        print(f"  [{label}] search...")
        await page.goto(search_url, timeout=40000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2500)

        # Step 2: find first result link - usually within a table of acts
        # Results are <a href="/handle/123456789/NNN">...</a>
        handles = await page.evaluate("""
            () => {
                const links = [...document.querySelectorAll('a[href*="/handle/123456789/"]')];
                return links.slice(0, 5).map(a => ({
                    href: a.href, text: a.innerText.trim()
                }));
            }
        """)
        if not handles:
            print(f"  [{label}] no search results")
            await ctx.close(); return False
        print(f"  [{label}] found {len(handles)} handles, top: {handles[0]['text'][:70]}")

        # Step 3: pick first and visit handle page
        await page.goto(handles[0]["href"], timeout=40000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2500)

        # Step 4: find bitstream link on handle page
        pdf_links = await page.evaluate("""
            () => {
                const links = [...document.querySelectorAll('a[href*="/bitstream/"]')];
                return links.map(a => ({
                    href: a.href, text: a.innerText.trim(),
                    size: (a.closest('tr')?.innerText || '').match(/\\d+(\\.\\d+)?\\s*(KB|MB)/i)?.[0] || ''
                }));
            }
        """)
        if not pdf_links:
            print(f"  [{label}] handle page has no bitstream links")
            await ctx.close(); return False
        # prefer pdf that looks like main act (largest / first)
        pdf_url = pdf_links[0]["href"]
        print(f"  [{label}] bitstream -> {pdf_url[-60:]}")

        # Step 5: navigate to bitstream URL - browser follows JS challenge, response listener catches
        try:
            await page.goto(pdf_url, timeout=45000, wait_until="networkidle")
        except Exception as e:
            pass  # networkidle may time out; response listener may have bytes

        await page.wait_for_timeout(2000)

        if not pdf_bytes["data"]:
            # Try context.request to fetch with browser cookies
            try:
                resp = await ctx.request.get(pdf_url, timeout=45000)
                if resp.status == 200:
                    body = await resp.body()
                    if body and body[:4] == b"%PDF":
                        pdf_bytes["data"] = body
            except: pass

        if not pdf_bytes["data"]:
            print(f"  [{label}] no PDF captured")
            await ctx.close(); return False

        # Step 6: QC
        ok, reason, meta = qc(pdf_bytes["data"], kw)
        if not ok:
            print(f"  [{label}] QC-REJECT: {reason} {meta}")
            await ctx.close(); return False

        dest = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f: f.write(pdf_bytes["data"])
        print(f"  [{label}] SAVED {meta['size_kb']}KB {meta['pg']}p c/p={meta['cpp']}")
        print(f"           title: {meta['title']}")
        await ctx.close()
        return True

    except Exception as e:
        print(f"  [{label}] EXC {type(e).__name__}: {e}")
        await ctx.close()
        return False


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])

        saved = 0
        failed = []
        for label, search, kw, rel in JOBS:
            ok = await fetch_one(browser, label, search, kw, rel)
            if ok: saved += 1
            else: failed.append(label)

        await browser.close()

        print("\n" + "="*70)
        print(f"PLAYWRIGHT v2: {saved}/{len(JOBS)}")
        print("\nFAILED:")
        for f in failed: print(f"  - {f}")

        print("\n=== FINAL STAGING ===")
        tp=0; tk=0
        for d in sorted(os.listdir(STAGE)):
            full = os.path.join(STAGE, d)
            if not os.path.isdir(full) or not d.startswith("t1_"): continue
            pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
            kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
            if pdfs:
                tp+=len(pdfs); tk+=kb
                print(f"  {d:<32} {len(pdfs):>3} pdfs  {kb:>7} KB")
        print(f"TOTAL: {tp} pdfs, {tk/1024:.1f} MB")

asyncio.run(main())
