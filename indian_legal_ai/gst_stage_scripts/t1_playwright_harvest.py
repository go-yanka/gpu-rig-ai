#!/usr/bin/env python3
"""Playwright harvest of 24 failing acts from indiacode.nic.in with inline QC.

Strategy: launch real Chromium, navigate to homepage to solve JS challenge + set cookies,
then navigate to each bitstream URL. Browser handles redirect chain. Capture PDF
via page.expect_download() or raw response. Run inline QC before saving.
"""
import os, asyncio, fitz
from playwright.async_api import async_playwright

STAGE = "/opt/indian-legal-ai/gst_stage"

# (label, act_title_for_search, title_kw, dest_rel, [candidate_bitstream_urls_or_handle_urls])
JOBS = [
    ("general_clauses_1897", "general clauses act 1897", "general clauses act",
     "t1_interpretation/general_clauses_1897.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2309/1/A1897-10.pdf",
     ]),
    ("indian_succession_1925", "indian succession act 1925", "indian succession act",
     "t1_personal_law/indian_succession_1925.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2362/1/A1925-39.pdf",
     ]),
    ("special_marriage_1954", "special marriage act 1954", "special marriage act",
     "t1_personal_law/special_marriage_1954.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1650/1/A1954-43.pdf",
     ]),
    ("muslim_dissolution_1939", "dissolution of muslim marriages act 1939", "dissolution of muslim marriages",
     "t1_personal_law/muslim_dissolution_1939.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2401/1/A1939-08.pdf",
     ]),
    ("shariat_1937", "muslim personal law shariat application act 1937", "shariat",
     "t1_personal_law/shariat_application_1937.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2402/1/A1937-26.pdf",
     ]),
    ("indian_divorce_1869", "indian divorce act 1869", "indian divorce act",
     "t1_personal_law/indian_divorce_1869.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2357/1/A1869-04.pdf",
     ]),
    ("indian_trusts_1882", "indian trusts act 1882", "indian trusts act",
     "t1_other_bare_acts/indian_trusts_1882.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2340/1/A1882-02.pdf",
     ]),
    ("indian_easements_1882", "indian easements act 1882", "indian easements act",
     "t1_other_bare_acts/indian_easements_1882.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2341/1/A1882-05.pdf",
     ]),
    ("insurance_1938", "insurance act 1938", "insurance act, 1938",
     "t1_banking/insurance_act_1938.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2397/1/A1938-04.pdf",
     ]),
    ("benami_1988", "benami transactions prohibition act 1988", "benami",
     "t1_banking/benami_1988.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1782/1/A1988-45.pdf",
     ]),
    ("customs_1962", "customs act 1962", "customs act, 1962",
     "t1_customs_excise/customs_act_1962.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1587/1/A1962-52.pdf",
     ]),
    ("central_excise_1944", "central excise act 1944", "central excise act",
     "t1_customs_excise/central_excise_1944.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1367/1/A1944-01.pdf",
     ]),
    ("trade_marks_1999", "trade marks act 1999", "trade marks act",
     "t1_ip_acts/trade_marks_1999.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1993/1/A1999-47.pdf",
     ]),
    ("pc_act_1988", "prevention of corruption act 1988", "prevention of corruption",
     "t1_criminal_special/pc_act_1988.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1793/1/A1988-49.pdf",
     ]),
    ("ndps_1985", "narcotic drugs psychotropic substances act 1985", "narcotic drugs",
     "t1_criminal_special/ndps_1985.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1791/1/A1985-61.pdf",
     ]),
    ("arms_1959", "arms act 1959", "arms act",
     "t1_criminal_special/arms_1959.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1435/1/A1959-54.pdf",
     ]),
    ("dpdp_2023", "digital personal data protection act 2023", "digital personal data protection",
     "t1_data_privacy/dpdp_2023.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/19865/1/dpdp_act_2023.pdf",
     ]),
    ("competition_2002", "competition act 2002", "competition act",
     "t1_other_bare_acts/competition_2002.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2004/1/A2002-12.pdf",
     ]),
    ("environment_1986", "environment protection act 1986", "environment (protection) act",
     "t1_environment/environment_1986.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1767/1/A1986-29.pdf",
     ]),
    ("wildlife_1972", "wildlife protection act 1972", "wild life",
     "t1_environment/wildlife_1972.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1726/1/A1972-53.pdf",
     ]),
    ("industrial_disputes_1947", "industrial disputes act 1947", "industrial disputes act",
     "t1_pre_codified_labour/industrial_disputes_1947.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1404/1/A1947-14.pdf",
     ]),
    ("factories_1948", "factories act 1948", "factories act",
     "t1_pre_codified_labour/factories_1948.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1552/1/A1948-63.pdf",
     ]),
    ("min_wages_1948", "minimum wages act 1948", "minimum wages act",
     "t1_pre_codified_labour/minimum_wages_1948.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1551/1/A1948-11.pdf",
     ]),
    ("gratuity_1972", "payment of gratuity act 1972", "payment of gratuity",
     "t1_pre_codified_labour/gratuity_1972.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1622/1/A1972-39.pdf",
     ]),
]

BAD = ["amendment bill","written answer","jstor","research paper","papers laid",
       "digital copy of a book","cornell university","all india christian council",
       "leave of absence"]

def qc_pdf_bytes(content: bytes, kw: str):
    """Inline QC - returns (ok: bool, reason: str, meta: dict)"""
    if len(content) < 1024:
        return False, f"too-small ({len(content)}B)", {}
    if content[:4] != b"%PDF":
        return False, f"not-pdf-magic ({content[:20]!r})", {}
    try:
        d = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        return False, f"parse-err: {e}", {}
    pg = d.page_count
    if pg < 4:
        d.close()
        return False, f"stub ({pg}p)", {"pg": pg}
    tot = 0
    for i in range(pg):
        tot += len(d.load_page(i).get_text())
    cpp = tot // max(pg, 1)
    if cpp < 200:
        d.close()
        return False, f"scanned ({cpp}c/p)", {"pg": pg, "cpp": cpp}
    first = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
    d.close()
    tl = first.lower()[:8000]
    title = first[:160].replace("\n", " ").strip()
    if kw.lower() not in tl:
        return False, f"kw-miss '{kw}'", {"pg": pg, "cpp": cpp, "title": title[:80]}
    for b in BAD:
        if b in tl:
            return False, f"bad-marker '{b}'", {"pg": pg, "cpp": cpp}
    return True, "ok", {"pg": pg, "cpp": cpp, "size_kb": len(content)//1024, "title": title[:100]}


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36",
            accept_downloads=True,
        )
        page = await ctx.new_page()

        # Warm-up: visit homepage to run JS challenge & set cookies
        print("=== Warming cookies via homepage ===")
        try:
            await page.goto("https://www.indiacode.nic.in/", wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(3000)
            cookies = await ctx.cookies("https://www.indiacode.nic.in/")
            print(f"  got {len(cookies)} cookies: {[c['name'] for c in cookies]}")
        except Exception as e:
            print(f"  homepage load err: {e}")

        saved = 0
        failed = []

        for label, search_name, kw, rel, urls in JOBS:
            got = False
            for u in urls:
                print(f"\n  [{label}] -> {u[:90]}")
                try:
                    # Use APIRequestContext from the authenticated browser context
                    resp = await ctx.request.get(u, timeout=45000, max_redirects=10)
                except Exception as e:
                    print(f"    EXC: {type(e).__name__}: {e}")
                    continue
                if resp.status != 200:
                    print(f"    HTTP {resp.status}")
                    continue
                try:
                    content = await resp.body()
                except Exception as e:
                    print(f"    body-err: {e}")
                    continue
                ok, reason, meta = qc_pdf_bytes(content, kw)
                if not ok:
                    print(f"    QC-REJECT: {reason} {meta}")
                    continue
                dest = os.path.join(STAGE, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(content)
                print(f"    SAVED {meta['size_kb']}KB {meta['pg']}p c/p={meta['cpp']}")
                print(f"    title: {meta['title']}")
                got = True
                saved += 1
                break
            if not got:
                failed.append(label)

        await browser.close()

        print("\n" + "="*70)
        print(f"PLAYWRIGHT HARVEST: {saved}/{len(JOBS)} saved")
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
