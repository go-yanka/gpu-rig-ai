#!/usr/bin/env python3
"""Debug: what does indiacode search actually render?"""
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36",
        )
        page = await ctx.new_page()

        # Try 1: homepage
        print("=== 1. Homepage ===")
        await page.goto("https://www.indiacode.nic.in/", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(4000)
        print(f"  URL after goto: {page.url}")
        print(f"  Title: {await page.title()}")
        html = await page.content()
        print(f"  HTML len: {len(html)}")
        print(f"  first 500: {html[:500]}")

        # Try 2: search URL patterns
        for url in [
            "https://www.indiacode.nic.in/simple-search?query=general+clauses+act",
            "https://www.indiacode.nic.in/handle/123456789/15479",  # known good IBC
            "https://www.indiacode.nic.in/handle/123456789/2309",   # guessed general clauses handle
        ]:
            print(f"\n=== {url} ===")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(4000)
                print(f"  final URL: {page.url}")
                print(f"  title: {await page.title()}")
                h = await page.content()
                # look for handles and bitstreams
                import re
                handles = re.findall(r'/handle/123456789/\d+', h)
                bitstreams = re.findall(r'/bitstream/123456789/[^\s"\'<>]+\.pdf', h)
                print(f"  handle refs: {len(set(handles))} uniq, sample: {list(set(handles))[:5]}")
                print(f"  bitstream refs: {len(set(bitstreams))} uniq, sample: {list(set(bitstreams))[:3]}")
                # page text containing the act name
                visible = await page.evaluate("() => document.body.innerText")
                print(f"  body text len: {len(visible)}, first 300: {visible[:300]}")
            except Exception as e:
                print(f"  ERR: {type(e).__name__}: {e}")

        await browser.close()

asyncio.run(main())
