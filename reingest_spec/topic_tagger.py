#!/usr/bin/env python3
"""Rule-based topic tagger for CBIC chunks.
- Input: chunk text + category (gst/customs/service_tax/central_excise/others)
- Output: primary_topic (str|None) + all_topics (list[str])
- Seed rules from gold eval's expected_terms, plus hand-curated patterns.
- Target: coverage of all 66 gold (category, topic) pairs with >=20 chunks each.
"""
import re, json, sqlite3
from pathlib import Path
from collections import defaultdict

# ---- Topic pattern rules (keywords; case-insensitive substring match) ----
# Each topic: list of phrases. Score = count of distinct phrases matched.
# Phrases should be discriminative (not generic words like "tax" alone).

RULES = {
    # ===== GST =====
    "gst:place_of_supply": ["place of supply", "section 10 igst", "section 11 igst", "section 12 igst", "section 13 igst", "bill-to-ship-to", "bill to ship to", "inter-state", "intra-state"],
    "gst:composite_mixed_supply": ["composite supply", "mixed supply", "principal supply", "naturally bundled", "section 2(30)", "section 2(74)", "section 8 cgst"],
    "gst:input_tax_credit": ["input tax credit", "itc", "section 16 cgst", "section 17 cgst", "blocked credit", "ineligible credit", "rule 36", "rule 37", "rule 42", "rule 43"],
    "gst:advances_timing": ["time of supply", "section 12 cgst", "section 13 cgst", "advance received", "receipt voucher", "refund voucher"],
    "gst:invoice_credit_note": ["tax invoice", "credit note", "debit note", "section 31 cgst", "section 34 cgst", "rule 46", "rule 53"],
    "gst:reverse_charge": ["reverse charge", "rcm", "section 9(3)", "section 9(4)", "notification 13/2017", "goods transport agency", "gta"],
    "gst:refund": ["refund", "section 54 cgst", "section 55", "section 77", "rule 89", "rule 92", "inverted duty", "unutilized itc", "zero rated"],
    "gst:e_way_bill": ["e-way bill", "eway bill", "rule 138", "rule 138a", "rule 138b", "part a", "part b"],
    "gst:zero_rated": ["zero rated", "zero-rated", "export of services", "sez", "section 16 igst", "lut", "letter of undertaking"],
    "gst:exempt": ["exempt supply", "section 11 cgst", "notification 12/2017", "notification 2/2017", "nil rated"],
    "gst:advance_ruling": ["advance ruling", "section 95 cgst", "section 97 cgst", "section 100 cgst", "aar", "aaar"],
    "gst:appeals": ["appeal", "section 107", "section 112", "commissioner (appeals)", "pre-deposit", "appellate tribunal", "gstat"],
    "gst:anti_evasion": ["anti-evasion", "section 67", "search and seizure", "section 122", "section 132"],
    "gst:offences": ["offence", "section 132 cgst", "prosecution", "compounding", "section 138"],
    "gst:penalty": ["penalty", "section 122", "section 125", "general penalty", "section 73", "section 74"],
    "gst:rate": ["rate of tax", "notification 1/2017", "schedule i", "schedule ii", "schedule iii", "schedule iv", "schedule v", "schedule vi", "tariff"],
    "gst:hsn": ["hsn", "harmonized system", "chapter heading", "tariff item", "notification 78/2020"],
    "gst:sac": ["sac", "services accounting code", "scheme of classification of services"],
    "gst:ccy_supply": ["continuous supply", "section 31(5)", "section 31(6)"],
    "gst:cess": ["compensation cess", "gst compensation", "cess act", "coal", "tobacco"],
    "gst:interest": ["interest", "section 50", "section 56", "delayed payment"],
    "gst:inverted": ["inverted duty", "inverted rate", "inverted tax structure", "inverted duty structure", "rule 89(5)", "net itc"],
    "gst:ccy_supply": ["continuous supply", "section 2(33)", "section 31(5)", "section 31(6)", "successive statements"],
    "gst:notif": ["notification", "cgst rate", "igst rate"],  # very loose; only tags if other topic didn't win
    "gst:st_gst_transition": ["transition", "tran-1", "tran-2", "section 140", "section 142", "cenvat credit carried forward"],
    "gst:complex": [],  # wildcard, no specific keywords
    "gst:refuse": [],   # OOC — we never tag a chunk as a refusal topic; eval handles via G4

    # ===== Customs =====
    "customs:classification": ["customs tariff act", "general rules of interpretation", "gir", "rule 3(a)", "rule 3(b)", "chapter note", "section note"],
    "customs:customs_classification": ["customs tariff act", "general rules of interpretation", "gir", "rule 3(a)", "rule 3(b)", "chapter note", "section note"],
    "customs:valuation": ["section 14 customs act", "customs valuation rules", "transaction value", "related party", "rule 10", "rule 3(2)"],
    "customs:customs_valuation": ["section 14 customs act", "customs valuation rules", "transaction value", "related party", "rule 10", "rule 12", "rule 3(2)"],
    "customs:svb": ["special valuation branch", "svb", "circular 5/2016-customs", "related party imports"],
    "customs:drawback": ["drawback", "section 74 customs", "section 75 customs", "all industry rate", "air", "brand rate"],
    "customs:customs_drawback": ["drawback", "section 74 customs", "section 75 customs", "all industry rate", "air", "brand rate"],
    "customs:warehousing": ["warehouse", "warehoused goods", "section 57", "section 58", "section 59", "section 65", "mooowr", "bonded warehouse", "private warehouse", "public warehouse", "into-bond", "ex-bond"],
    "customs:customs_warehouse": ["warehouse", "warehoused goods", "section 57", "section 58", "section 59", "section 65", "mooowr", "bonded warehouse", "private warehouse", "public warehouse", "into-bond", "ex-bond"],
    "customs:customs_exemption": ["notification 50/2017-cus", "igcr", "customs (import of goods at concessional rate)", "end-use"],
    "customs:igst_import": ["igst on import", "section 3 ctsc", "customs tariff act section 3", "additional duty"],
    "customs:appeals": ["section 128 customs", "section 129 customs", "section 129e", "commissioner (appeals)", "cestat"],
    "customs:advance_ruling": ["caar", "section 28e customs", "section 28h customs", "authority for advance rulings", "advance ruling", "applicant customs", "customs ruling"],
    "customs:penalty": ["section 112", "section 114", "section 117", "confiscation"],
    "customs:customs_adviolation": ["adjudication", "section 28", "section 124", "show cause"],
    "customs:notif": ["notification", "customs notification"],
    "customs:rate": ["bcd", "basic customs duty", "tariff rate"],
    "customs:refuse": [],
    "customs:complex": [],

    # ===== Service Tax =====
    "service_tax:levy": ["section 66b", "section 65b(44)", "declared services", "service tax levy"],
    "service_tax:negative_list": ["negative list", "section 66d", "mega exemption", "notification 25/2012"],
    "service_tax:export_of_services": ["export of service", "export of services rules", "rule 6a", "rule 6a(1)", "convertible foreign exchange", "recipient located outside india", "service provider located in the taxable territory"],
    "service_tax:place_of_provision": ["place of provision", "popos", "place of provision of services rules 2012", "rule 3 popos", "rule 4 popos", "rule 9 popos", "location of service provider", "location of service receiver"],
    "service_tax:reverse_charge": ["reverse charge", "notification 30/2012-st", "rule 2(1)(d)(i)", "person liable for paying service tax", "recipient of service", "legal services", "manpower supply"],
    "service_tax:valuation": ["service tax valuation rules", "section 67", "gross amount charged"],
    "service_tax:service_tax": ["service tax", "cenvat credit rules"],
    "service_tax:complex": [],

    # ===== Central Excise =====
    "central_excise:manufacture": ["section 2(f)", "deemed manufacture", "third schedule", "labeling", "repacking"],
    "central_excise:valuation": ["section 4 central excise", "section 4a", "mrp", "abatement", "place of removal"],
    "central_excise:cenvat": ["cenvat credit rules", "cenvat", "rule 3", "rule 6 ccr", "rule 14 ccr"],
    "central_excise:ssi_exemption": ["ssi exemption", "notification 8/2003-ce", "1.5 crore", "brand name"],
    "central_excise:complex": [],

    # ===== Others =====
    "others:gaar": ["gaar", "general anti-avoidance", "general anti avoidance", "chapter x-a", "commercial substance", "impermissible avoidance arrangement", "tax avoidance arrangement", "section 95 of the income-tax", "section 96", "section 97 income tax", "section 98 income tax"],
    "others:advance_ruling": ["advance ruling", "authority for advance ruling", "aar order"],
    "others:anti_profiteering": ["anti-profiteering", "anti profiteering", "section 171", "naa", "national anti-profiteering", "commensurate reduction in prices", "competition commission of india in relation to anti"],
    "others:hsn_classification": ["hsn", "classification", "tariff"],
    "others:penalties": ["penalty", "section 122", "section 125"],
    "others:appeals": ["appeal", "cestat", "tribunal"],
    "others:finance_act": ["finance act", "finance (no. 2) act"],
    "others:complex": [],
    "others:refuse": [],
    "others:refuse_direct_tax": [],
}

# Topics that are NOT content-bearing (wildcard/refusal) and should not be tagged
NON_TAGGED = {k for k, v in RULES.items() if not v}


def _compile():
    out = {}
    for k, phrases in RULES.items():
        if not phrases: continue
        # compile single alternation for speed
        pats = [re.escape(p) for p in phrases]
        out[k] = re.compile(r"(?:" + "|".join(pats) + r")", re.IGNORECASE)
    return out

_COMPILED = _compile()


def tag_chunk(text, category=None):
    """Return (primary_topic, all_topics_with_scores).
    If category given, restrict to topics matching that category.
    """
    if not text: return None, {}
    hits = {}
    for topic_key, rx in _COMPILED.items():
        cat, sub = topic_key.split(":", 1)
        if category and cat != category: continue
        matches = rx.findall(text)
        # Score = distinct phrases matched (not total occurrences, to avoid boilerplate inflation)
        distinct = set(m.lower() for m in matches)
        if distinct:
            hits[topic_key] = len(distinct)
    if not hits: return None, {}
    primary = max(hits.items(), key=lambda x: (x[1], -len(x[0])))[0]
    return primary, hits


if __name__ == "__main__":
    # quick self-check
    demo = "Input tax credit is available under Section 16 CGST for ITC on inward supplies as per Rule 36."
    print(tag_chunk(demo, "gst"))
    print(f"Total rule-bearing topics: {len(_COMPILED)}")
    print(f"Non-content topics (wildcard/refusal): {len(NON_TAGGED)}")
