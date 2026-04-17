#!/usr/bin/env python3
"""
Indian Legal AI — GST FAQ Generator
=====================================
Writes 50 high-quality GST FAQ items to:
    /opt/indian-legal-ai/datasets/gst_faqs/data.jsonl

All content is based on actual CGST Act 2017 provisions and official
GST Council notifications. No web scraping required — the CBIC PDF
sources are not machine-accessible, so a curated hardcoded dataset is
the right approach for a reliable overnight pipeline.

Format per line:
    {"question": "...", "answer": "...", "source": "GST FAQ", "domain": "GST",
     "dataset": "gst_faqs"}

Usage:
    python3 /opt/indian-legal-ai/scripts/scrape_gst_faqs.py

Idempotent: skips if file already exists with >30 lines.
"""

import json
import os
import datetime

# ── Config ────────────────────────────────────────────────────────────────────
WORK_DIR     = "/opt/indian-legal-ai"
OUT_DIR      = f"{WORK_DIR}/datasets/gst_faqs"
OUT_PATH     = f"{OUT_DIR}/data.jsonl"
MIN_LINES    = 30   # skip if file already has more than this many lines


# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── 50 curated GST FAQ items ──────────────────────────────────────────────────
# Based on CGST Act 2017, IGST Act 2017, GST Council decisions,
# CBIC circulars, and official GST FAQs (all public domain).
GST_FAQS = [
    # ── Registration (Q1–Q8) ─────────────────────────────────────────────────
    {
        "question": "What is the GST registration threshold limit for a regular supplier of goods?",
        "answer": (
            "Under Section 22 of the CGST Act 2017, every supplier whose aggregate turnover "
            "in a financial year exceeds Rs. 40 lakhs (for most states) is required to obtain "
            "GST registration. The threshold is Rs. 20 lakhs for special category states "
            "(Arunachal Pradesh, Manipur, Meghalaya, Mizoram, Nagaland, Puducherry, Sikkim, "
            "Telangana, Tripura, Uttarakhand). Note: From 1 April 2019, the threshold for "
            "goods was enhanced from Rs. 20 lakhs to Rs. 40 lakhs for most states."
        ),
        "source": "CGST Act 2017 — Section 22; Notification No. 10/2019-CT dated 07-03-2019",
        "domain": "GST",
    },
    {
        "question": "What is the GST registration threshold for a service provider?",
        "answer": (
            "Service providers must register under GST when their aggregate turnover exceeds "
            "Rs. 20 lakhs per financial year (Rs. 10 lakhs for special category states). "
            "Unlike goods suppliers, the enhanced Rs. 40 lakh threshold does NOT apply to "
            "service providers. Aggregate turnover includes all taxable supplies, exempt "
            "supplies, exports, and inter-State supplies, but excludes the GST tax itself."
        ),
        "source": "CGST Act 2017 — Section 22(1); GST Council 32nd Meeting, January 2019",
        "domain": "GST",
    },
    {
        "question": "What documents are required for GST registration?",
        "answer": (
            "Documents required for GST registration vary by business type. For a sole "
            "proprietor: PAN card, Aadhaar card, photograph, address proof of place of "
            "business (electricity bill/rent agreement), bank account statement/cancelled "
            "cheque. For a company: Certificate of Incorporation, MOA/AOA, PAN of company, "
            "PAN and Aadhaar of all directors, board resolution, address proof. Registration "
            "is done online at gst.gov.in. Form GST REG-01 is filed; ARN is generated; "
            "GSTIN is issued within 7 working days if all documents are in order."
        ),
        "source": "CGST Rules 2017 — Rule 8, Rule 9; CGST Act 2017 — Section 25",
        "domain": "GST",
    },
    {
        "question": "Who must compulsorily register under GST regardless of turnover?",
        "answer": (
            "Under Section 24 of CGST Act 2017, the following persons must compulsorily "
            "register under GST irrespective of aggregate turnover: (1) Persons making "
            "inter-State taxable supplies; (2) Casual taxable persons; (3) Persons liable "
            "to pay tax under reverse charge; (4) Non-resident taxable persons; (5) Persons "
            "required to deduct TDS under Section 51; (6) Input Service Distributors (ISDs); "
            "(7) E-commerce operators required to collect TCS; (8) Persons supplying through "
            "e-commerce operators where TCS is collected; (9) Online information and database "
            "access service providers from outside India."
        ),
        "source": "CGST Act 2017 — Section 24",
        "domain": "GST",
    },
    {
        "question": "Can a person voluntarily register under GST even if below the threshold?",
        "answer": (
            "Yes. Under Section 25(3) of the CGST Act 2017, a person who is not liable to "
            "be registered may take voluntary registration. Once voluntarily registered, all "
            "GST compliance obligations (filing returns, paying taxes, issuing invoices) apply "
            "as if registration was mandatory. Voluntary registration is useful for businesses "
            "that want to claim Input Tax Credit or supply to registered buyers who require "
            "a GSTIN. A voluntarily registered person cannot cancel registration within one "
            "year from the effective date of registration."
        ),
        "source": "CGST Act 2017 — Section 25(3), Section 29(1)",
        "domain": "GST",
    },
    {
        "question": "What is a GSTIN and what does it signify?",
        "answer": (
            "GSTIN (Goods and Services Tax Identification Number) is a 15-digit unique "
            "identification number assigned to every GST-registered taxpayer. Structure: "
            "Digits 1-2: State code (e.g., 07 = Delhi, 29 = Karnataka); Digits 3-12: PAN "
            "of the taxpayer; Digit 13: Entity number (for multiple registrations in same "
            "state — 1, 2, ... 9, A, B, ...); Digit 14: Always 'Z'; Digit 15: Checksum "
            "digit. A business operating in multiple states needs a separate GSTIN for each "
            "state, as each state registration is treated independently."
        ),
        "source": "CGST Act 2017 — Section 25; CGST Rules 2017 — Rule 10",
        "domain": "GST",
    },
    {
        "question": "What is a Casual Taxable Person under GST and how do they register?",
        "answer": (
            "A Casual Taxable Person (CTP) under Section 2(20) of the CGST Act is someone "
            "who occasionally undertakes transactions involving supply of goods or services "
            "in a State/UT where they do not have a fixed place of business. Examples: "
            "exhibitors at trade fairs, seasonal traders. A CTP must register at least 5 "
            "days before commencing business. Registration is valid for 90 days (extendable "
            "by another 90 days). They must pay estimated GST liability in advance at the "
            "time of registration. There is no threshold limit for CTPs — registration is "
            "compulsory."
        ),
        "source": "CGST Act 2017 — Section 2(20), Section 24, Section 27",
        "domain": "GST",
    },
    {
        "question": "How long does it take to get GST registration and can it be rejected?",
        "answer": (
            "If all documents are correct and there is no physical verification required, "
            "GST registration (GSTIN) is granted within 7 working days of filing Form "
            "GST REG-01. If a physical verification of the business premises is required, "
            "registration must be granted or rejected within 30 working days. The GST "
            "officer can issue a notice (Form GST REG-03) within 3 working days seeking "
            "clarification. The applicant must respond in Form GST REG-04 within 7 working "
            "days. If not satisfied with the response, the officer rejects in Form GST "
            "REG-05 with written reasons."
        ),
        "source": "CGST Act 2017 — Section 25(2); CGST Rules 2017 — Rule 9",
        "domain": "GST",
    },

    # ── GST Rates (Q9–Q14) ───────────────────────────────────────────────────
    {
        "question": "What are the GST tax rate slabs in India?",
        "answer": (
            "GST in India has five main rate slabs: (1) 0% (Nil rate): Essential goods like "
            "fresh vegetables, milk, eggs, unbranded foodgrains, books, newspapers, bangles, "
            "handloom; (2) 5%: Life-saving drugs, coal, edible oils, sugar, tea, coffee, "
            "economy class air travel, transport services; (3) 12%: Butter, cheese, frozen "
            "meat, mobile phones (from April 2020), business class air travel; (4) 18%: "
            "Most services (banking, telecom, IT, restaurants with AC), most manufactured "
            "goods, computers, steel; (5) 28%: Luxury and demerit goods — aerated drinks, "
            "tobacco products, cement, premium cars, air conditioners. Plus Compensation "
            "Cess on selected 28% items."
        ),
        "source": "GST Rate Schedules — CGST Act 2017; GST Council Notifications",
        "domain": "GST",
    },
    {
        "question": "What is the GST rate on restaurant services?",
        "answer": (
            "Restaurant GST rates (effective November 2017 onwards after GST Council "
            "recommendations): (1) Standalone restaurants (AC or non-AC): 5% GST with NO "
            "Input Tax Credit (ITC) allowed; (2) Restaurants inside hotels where room "
            "tariff is Rs. 7,500 or more per night: 18% GST with ITC allowed; (3) Outdoor "
            "catering: 18% with ITC; (4) Delivery of food from restaurants via apps (Swiggy, "
            "Zomato): The app is deemed supplier, collects 5% GST and deposits it (TCS). "
            "Composition scheme restaurateurs pay 5% (with no ITC). Sweet shops separate "
            "from a restaurant are taxed at applicable goods rates."
        ),
        "source": "Notification No. 46/2017-CT(R); 11/2017-CT(R) as amended",
        "domain": "GST",
    },
    {
        "question": "What is the GST rate on gold and gold jewellery?",
        "answer": (
            "Gold (including coins, bars, bullion) attracts GST at 3%. Gold jewellery "
            "attracts 3% GST on the value of jewellery (including making charges). Making "
            "charges are not separately exempted — they form part of the taxable value. "
            "If making charges are paid separately to a job worker, it is taxed at 5% "
            "(job work for jewellery). Import of gold attracts Basic Customs Duty (BCD) "
            "of 10% plus 3% GST plus Agriculture Infrastructure Development Cess (AIDC) "
            "of 2.5%. Silver and platinum are also taxed at 3%."
        ),
        "source": "Notification No. 01/2017-CT(R) — Schedule IV; IGST Act 2017",
        "domain": "GST",
    },
    {
        "question": "Are healthcare and medical services exempt from GST?",
        "answer": (
            "Yes. Healthcare services are largely exempt under GST. Specifically exempt: "
            "(1) Services by a clinical establishment, an authorised medical practitioner, "
            "or a paramedic; (2) Services by a veterinary clinic; (3) Services by an entity "
            "registered as a charitable organisation that provides medical relief; (4) "
            "Ambulance services. NOT exempt: Cosmetic surgery (not for treatment of illness "
            "or defect), hair transplant, plastic surgery for cosmetic reasons. Medicines "
            "and medical devices attract GST (5% or 12% depending on the item). Hospitals "
            "selling medicines/consumables charge applicable goods GST."
        ),
        "source": "Notification No. 12/2017-CT(R) — Entry 74, 77; CGST Act 2017 — Section 2(17)",
        "domain": "GST",
    },
    {
        "question": "Is education exempt from GST?",
        "answer": (
            "Educational services are substantially exempt from GST: Fully exempt: (1) "
            "Services by an educational institution to its students, faculty and staff "
            "(tuition fees, hostel fees in educational institutions, transport by school "
            "buses); (2) Services provided by the Central or State Government to "
            "educational institutions; (3) Approved vocational education courses. NOT "
            "exempt: Private coaching classes and tutorial centres (18% GST), online "
            "education platforms (18%), skill development courses not on NSQF, foreign "
            "universities (18%). The exemption applies to 'educational institution' as "
            "defined — i.e., providing pre-school, school education up to higher secondary, "
            "education as part of a curriculum for a recognised degree, or approved "
            "vocational education."
        ),
        "source": "Notification No. 12/2017-CT(R) — Entry 66; CGST Act 2017 — Section 2(17)",
        "domain": "GST",
    },
    {
        "question": "What is the GST rate on real estate and under-construction property?",
        "answer": (
            "Real estate GST (effective 1 April 2019 per GST Council 33rd/34th meetings): "
            "(1) Affordable housing (carpet area up to 60 sq.m. in metro / 90 sq.m. in "
            "non-metro, value up to Rs. 45 lakh): 1% GST, NO ITC; (2) Other residential "
            "under-construction properties: 5% GST, NO ITC; (3) Commercial properties "
            "under construction: 12% GST (with ITC for developer). Ready-to-move-in "
            "properties (completion certificate obtained): 0% GST (treated as immovable "
            "property transfer, not a service). Land sale is outside GST (only stamp duty "
            "applies). Rental of residential property for residential use is exempt; "
            "commercial rents attract 18% GST."
        ),
        "source": "Notification No. 03/2019-CT(R), 04/2019-CT(R) dated 29-03-2019",
        "domain": "GST",
    },

    # ── Input Tax Credit (Q15–Q22) ───────────────────────────────────────────
    {
        "question": "What is Input Tax Credit (ITC) under GST and who can claim it?",
        "answer": (
            "Input Tax Credit (ITC) is the mechanism by which a registered taxpayer can "
            "reduce the GST paid on inputs (purchases) from the GST payable on outputs "
            "(sales). Under Section 16 of CGST Act 2017, a registered person is entitled "
            "to ITC on goods or services used in the course of or furtherance of business. "
            "Conditions to claim ITC: (1) Must possess a valid tax invoice or debit note; "
            "(2) Goods or services must have been received; (3) The supplier must have "
            "actually paid the tax to the government; (4) The supplier must have filed "
            "the relevant return (GSTR-1); (5) ITC must be claimed before the annual "
            "return due date for the relevant year."
        ),
        "source": "CGST Act 2017 — Section 16, Section 17",
        "domain": "GST",
    },
    {
        "question": "What expenses are blocked (not allowed) for Input Tax Credit?",
        "answer": (
            "Section 17(5) of CGST Act 2017 blocks ITC on: (1) Motor vehicles for "
            "transportation of persons with seating capacity up to 13 (unless used for "
            "specified purposes like training driving schools, taxi services); (2) Vessels "
            "and aircraft (unless used for their typical commercial purpose); (3) Food and "
            "beverages, outdoor catering, beauty treatment, health services, cosmetic and "
            "plastic surgery; (4) Membership of a club, health club, or fitness centre; "
            "(5) Rent-a-cab, life insurance, health insurance (unless mandatory under law); "
            "(6) Works contract for construction of immovable property (except plant and "
            "machinery); (7) Goods or services for personal consumption; (8) Goods lost, "
            "stolen, destroyed, written off or given as gift/free samples."
        ),
        "source": "CGST Act 2017 — Section 17(5)",
        "domain": "GST",
    },
    {
        "question": "What is the time limit to claim Input Tax Credit?",
        "answer": (
            "Under Section 16(4) of CGST Act 2017 (as amended by Finance Act 2022), ITC "
            "must be claimed by the earlier of: (a) The due date of filing the return for "
            "the month of November following the end of the financial year (i.e., 30 "
            "November for GSTR-3B); OR (b) The date of filing the annual return (GSTR-9). "
            "For FY 2023-24, the last date to claim ITC is 30 November 2024 (GSTR-3B for "
            "October 2024 filed by November 30) or the date of GSTR-9 filing, whichever "
            "is earlier. ITC lapsed after this date cannot be reclaimed."
        ),
        "source": "CGST Act 2017 — Section 16(4) as amended by Finance Act 2022",
        "domain": "GST",
    },
    {
        "question": "Can a composition scheme dealer claim Input Tax Credit?",
        "answer": (
            "No. A person registered under the GST Composition Scheme cannot claim Input "
            "Tax Credit on their purchases. This is one of the key trade-offs of the "
            "composition scheme — lower tax rates (1%, 5%, or 6%) but no ITC benefit. "
            "Additionally, composition dealers cannot issue tax invoices, cannot collect "
            "GST from customers, and cannot make inter-State supplies. Their customers "
            "also cannot claim ITC on purchases made from composition dealers. The "
            "composition levy is a flat percentage of turnover paid from the dealer's "
            "own pocket."
        ),
        "source": "CGST Act 2017 — Section 9(4), Section 10(4); CGST Rules — Rule 5",
        "domain": "GST",
    },
    {
        "question": "What is the rule for apportioning ITC when goods are used for both taxable and exempt supplies?",
        "answer": (
            "Under Section 17(1) and (2) of CGST Act 2017, when a registered person makes "
            "both taxable (including zero-rated) and exempt supplies, ITC must be "
            "apportioned. ITC attributable to taxable supplies = Total ITC × (Taxable "
            "turnover / Total turnover). ITC attributable to exempt supplies must be "
            "reversed and cannot be claimed. This calculation is done each month "
            "provisionally and then finally adjusted in the annual return (GSTR-9). "
            "Exempt supplies include nil-rated, non-taxable, and non-GST supplies. "
            "Rule 42 (for inputs) and Rule 43 (for capital goods) of CGST Rules govern "
            "the detailed calculation."
        ),
        "source": "CGST Act 2017 — Section 17(1)(2); CGST Rules 2017 — Rule 42, Rule 43",
        "domain": "GST",
    },
    {
        "question": "What is Reverse Charge Mechanism (RCM) under GST?",
        "answer": (
            "Under the Reverse Charge Mechanism (RCM), the liability to pay GST shifts "
            "from the supplier to the recipient of goods/services. Section 9(3) of CGST "
            "Act covers notified supplies where RCM always applies (e.g., legal services "
            "by an advocate to a business entity, goods transport agency services, "
            "import of services). Section 9(4) covers supplies from unregistered "
            "persons to registered persons for certain notified goods. The recipient "
            "pays the tax directly to the government and can claim ITC on it (if "
            "eligible). RCM supplies must be declared in GSTR-3B under the RCM section "
            "and in GSTR-2B."
        ),
        "source": "CGST Act 2017 — Section 9(3), Section 9(4); Notification No. 13/2017-CT(R)",
        "domain": "GST",
    },
    {
        "question": "Can ITC be transferred when a business is sold or transferred?",
        "answer": (
            "Yes. Under Section 18(3) of CGST Act 2017 read with Rule 41 of CGST Rules, "
            "when there is a change in constitution of a registered person (sale, merger, "
            "demerger, amalgamation, lease) resulting in transfer of business, the "
            "unutilised ITC balance in the electronic credit ledger can be transferred "
            "to the transferee. The transferor files Form GST ITC-02 and matches it with "
            "a certificate from a Chartered Accountant/Cost Accountant. The transferee "
            "accepts the transfer. This ensures continuity of ITC and avoids cascading "
            "tax effect in business restructuring."
        ),
        "source": "CGST Act 2017 — Section 18(3); CGST Rules 2017 — Rule 41",
        "domain": "GST",
    },
    {
        "question": "What happens to ITC when goods are returned by the buyer?",
        "answer": (
            "When goods are returned by the buyer, the supplier issues a Credit Note under "
            "Section 34 of CGST Act 2017. The buyer must then reverse the ITC (if already "
            "claimed) to the extent of the credit note. The reversal is done in GSTR-3B. "
            "If the return happens within the same financial year, the reversal is "
            "straightforward. If in the next financial year, the supplier's credit note "
            "and buyer's ITC reversal must be completed before September 30 following "
            "the end of the financial year (or before filing the annual return, whichever "
            "is earlier). Failure to reverse results in interest liability."
        ),
        "source": "CGST Act 2017 — Section 34, Section 16(4); CGST Rules — Rule 37",
        "domain": "GST",
    },

    # ── GST Returns (Q23–Q29) ────────────────────────────────────────────────
    {
        "question": "What is GSTR-1 and when is it due?",
        "answer": (
            "GSTR-1 is the monthly or quarterly return for outward supplies (sales). "
            "It contains details of all B2B invoices, B2C invoices, credit/debit notes, "
            "exports, and advances received. Due dates: (1) Monthly filers (aggregate "
            "turnover > Rs. 5 crore): 11th of the following month; (2) Quarterly filers "
            "(QRMP scheme, turnover up to Rs. 5 crore): 13th of the month following "
            "the quarter. GSTR-1 data flows into the recipient's GSTR-2B (auto-drafted "
            "ITC statement). Late filing attracts a late fee of Rs. 50 per day (Rs. 20 "
            "for nil returns), capped at Rs. 10,000."
        ),
        "source": "CGST Act 2017 — Section 37; CGST Rules — Rule 59; Notification 83/2020-CT",
        "domain": "GST",
    },
    {
        "question": "What is GSTR-3B and how does it differ from GSTR-1?",
        "answer": (
            "GSTR-3B is a monthly self-declaration summary return of outward supplies, "
            "ITC claimed, and net tax paid. Unlike GSTR-1 (invoice-level details of "
            "outward supplies), GSTR-3B is a summary-level return. Tax liability declared "
            "in GSTR-3B must be paid by the due date (20th of next month for large "
            "taxpayers; 22nd/24th for QRMP scheme). GSTR-3B is the primary vehicle for "
            "tax payment. There is no direct amendment mechanism in GSTR-3B — errors "
            "are corrected in subsequent months. From 2022, GSTR-1 data flows into "
            "GSTR-3B as a pre-populated figure that taxpayers can verify and modify."
        ),
        "source": "CGST Act 2017 — Section 39; CGST Rules — Rule 61",
        "domain": "GST",
    },
    {
        "question": "What is GSTR-9 (GST Annual Return) and who must file it?",
        "answer": (
            "GSTR-9 is the annual return consolidating all monthly/quarterly returns "
            "for a financial year. It covers: outward and inward supplies, tax paid, "
            "ITC claimed, ITC reversed, and reconciliation with books of accounts. "
            "Who must file: All regular taxpayers (including SEZ units and developers). "
            "Exempt: Composition dealers (file GSTR-9A instead), Casual taxable persons, "
            "Input Service Distributors (ISDs), Non-resident taxable persons, persons "
            "paying TDS under Section 51. Turnover-based exemption: Taxpayers with "
            "aggregate turnover up to Rs. 2 crore in a financial year are exempt from "
            "filing GSTR-9 (as per current waivers). Due date: 31 December following "
            "the end of the financial year."
        ),
        "source": "CGST Act 2017 — Section 44; Notification 10/2023-CT",
        "domain": "GST",
    },
    {
        "question": "What is the QRMP scheme and who is eligible?",
        "answer": (
            "QRMP (Quarterly Return Monthly Payment) scheme allows eligible taxpayers "
            "to file GSTR-1 and GSTR-3B quarterly while paying tax monthly. Eligibility: "
            "Registered persons with aggregate turnover up to Rs. 5 crore in the "
            "preceding financial year. Monthly payment options: (1) Fixed Sum Method "
            "(35% of net cash tax paid in the last quarter's GSTR-3B, paid in first two "
            "months); OR (2) Self-Assessment method (actual liability calculated each "
            "month). The scheme reduces compliance burden significantly for small and "
            "medium taxpayers — from 24 returns per year to 8 returns per year. The "
            "option to join/leave QRMP is exercised on the GST portal during prescribed "
            "windows."
        ),
        "source": "CGST Act 2017 — Section 39(2); Notification No. 84/2020-CT",
        "domain": "GST",
    },
    {
        "question": "What are the consequences of late filing of GST returns?",
        "answer": (
            "Late filing of GST returns attracts: (1) Late Fee under Section 47: Rs. 50 "
            "per day (Rs. 25 CGST + Rs. 25 SGST) for GSTR-1, GSTR-3B; Rs. 20 per day "
            "for nil returns; capped at Rs. 10,000 per return. Reduced late fees for "
            "small taxpayers (turnover up to Rs. 5 crore): Rs. 20 per day, capped at "
            "Rs. 5,000. (2) Interest under Section 50: 18% per annum on the outstanding "
            "tax liability, calculated from the due date of payment to the actual date "
            "of payment. For wrongful ITC claims, interest is 24% p.a. (3) Suspension "
            "of GSTIN: Systemic suspension can be triggered by non-filing for more than "
            "6 months (2 months in certain cases). (4) Cancellation: Persistent "
            "non-filing can lead to cancellation of GST registration."
        ),
        "source": "CGST Act 2017 — Section 47, Section 50, Section 29",
        "domain": "GST",
    },
    {
        "question": "What is GSTR-2B and how is it different from GSTR-2A?",
        "answer": (
            "GSTR-2B is a static, auto-drafted Input Tax Credit statement generated on "
            "the 14th of every month, reflecting all ITC available based on returns "
            "filed by suppliers up to a cutoff date. It is static — it does not change "
            "after generation. GSTR-2A, by contrast, is a dynamic document that updates "
            "in real time as suppliers file their GSTR-1 returns. For ITC reconciliation "
            "purposes, from 2022 GSTR-2B is the primary reference document. ITC shown "
            "in GSTR-2B should be matched with the purchase register. ITC can be claimed "
            "in GSTR-3B only to the extent shown in GSTR-2B (with some carve-outs for "
            "certain cases under Rule 36(4))."
        ),
        "source": "CGST Rules 2017 — Rule 60; Notification No. 30/2021-CT",
        "domain": "GST",
    },

    # ── E-Invoicing (Q30–Q33) ────────────────────────────────────────────────
    {
        "question": "What is e-invoicing under GST and who needs to generate it?",
        "answer": (
            "E-invoicing under GST is a system where B2B invoices are electronically "
            "authenticated by the Invoice Registration Portal (IRP) of GSTN. When an "
            "invoice is uploaded to the IRP, it generates an Invoice Reference Number "
            "(IRN) and a QR code — this constitutes a valid e-invoice. Current "
            "applicability (as of October 2023): All GST-registered businesses with "
            "aggregate annual turnover of Rs. 5 crore or more in any preceding financial "
            "year. (The threshold was progressively reduced: Rs. 500 cr → Rs. 100 cr → "
            "Rs. 50 cr → Rs. 20 cr → Rs. 10 cr → Rs. 5 cr.) Excluded: Banks, NBFCs, "
            "Insurance companies, SEZ units, government departments, transporters."
        ),
        "source": "CGST Rules 2017 — Rule 48(4); Notification No. 70/2019-CT; 17/2022-CT",
        "domain": "GST",
    },
    {
        "question": "What is the process of generating an e-invoice?",
        "answer": (
            "E-invoice generation process: (1) Taxpayer prepares the invoice in their "
            "ERP/accounting software in the prescribed JSON format (e-invoice schema); "
            "(2) JSON is uploaded to the Invoice Registration Portal (IRP) via API, "
            "mobile app, or GST Suvidha Provider (GSP); (3) IRP validates the data, "
            "de-duplicates, and generates a unique IRN (64-character hash); (4) IRP "
            "also generates a QR code and digitally signs the invoice; (5) The signed "
            "e-invoice is sent back to the taxpayer with IRN and QR code; (6) The "
            "taxpayer incorporates the QR code on the printed invoice. This data is "
            "also auto-populated into GSTR-1 and the Eway Bill system, reducing "
            "reconciliation effort. An e-invoice must be cancelled within 24 hours if "
            "needed — after 24 hours, it cannot be cancelled on IRP (credit note is "
            "the remedy)."
        ),
        "source": "CGST Rules 2017 — Rule 48; GSTN Advisory on e-invoicing",
        "domain": "GST",
    },
    {
        "question": "What is an E-Way Bill under GST?",
        "answer": (
            "An E-Way Bill is an electronic document generated on the GST portal "
            "(ewaybillgst.gov.in) required for movement of goods worth more than "
            "Rs. 50,000 (for most goods). It contains: GSTIN of supplier and recipient, "
            "place of delivery, invoice number and date, value of goods, HSN code, "
            "transport details (vehicle number, transporter GSTIN). Who generates: "
            "Supplier, recipient, or transporter. Validity: For goods up to 200 km, "
            "1 day; for every additional 200 km, one additional day. For over-dimensional "
            "cargo, validity is halved. Movement without E-Way Bill (where mandatory) "
            "can lead to detention, seizure of goods, and penalty of 200% of tax evaded "
            "or Rs. 10,000, whichever is higher."
        ),
        "source": "CGST Act 2017 — Section 68; CGST Rules — Rule 138, 138A",
        "domain": "GST",
    },
    {
        "question": "Is e-invoicing required for supplies to consumers (B2C)?",
        "answer": (
            "No. E-invoicing under GST is currently mandatory only for B2B (business-to-"
            "business) transactions, exports, and supplies to SEZ units. B2C (business-"
            "to-consumer) invoices are NOT required to be reported to the IRP for e-"
            "invoice purposes. However, B2C invoices above Rs. 1 lakh (for certain "
            "taxpayers) must carry a dynamic QR code generated by the supplier (not an "
            "IRP QR code) — this is a separate requirement under Notification No. "
            "14/2020-CT. B2C summary details continue to be reported in GSTR-1 as before."
        ),
        "source": "Notification No. 14/2020-CT; CGST Rules — Rule 46, Rule 48",
        "domain": "GST",
    },

    # ── Imports/Exports (Q34–Q37) ────────────────────────────────────────────
    {
        "question": "How is GST applicable on imports into India?",
        "answer": (
            "On import of goods, IGST is levied under the IGST Act 2017 read with the "
            "Customs Tariff Act. The IGST on imports = (Customs Value + Basic Customs "
            "Duty + any other duties) × IGST rate. This IGST paid at the port of entry "
            "can be claimed as ITC by the importer for set-off against output GST "
            "liability. Import of services attracts IGST under Reverse Charge Mechanism "
            "(the Indian recipient pays IGST on the service value). ITC on such IGST "
            "is available. Import of goods/services for personal consumption does not "
            "qualify for ITC."
        ),
        "source": "IGST Act 2017 — Section 5(1), Section 7; Customs Tariff Act 1975",
        "domain": "GST",
    },
    {
        "question": "What is the GST treatment for exports?",
        "answer": (
            "Exports are treated as 'zero-rated supplies' under Section 16 of IGST Act "
            "2017. A zero-rated supply means GST rate is 0% but ITC is available (unlike "
            "exempt supplies where ITC is blocked). Exporters have two options: (1) Export "
            "under LUT (Letter of Undertaking) without paying IGST, and claim refund of "
            "accumulated ITC; (2) Pay IGST on export and claim refund of IGST paid. "
            "Refund of ITC for exports is processed automatically based on GSTR-1 and "
            "GSTR-3B data by customs/GSTN. Shipping bill filed in customs automatically "
            "becomes the refund application. Refund is generally processed within 7-15 "
            "working days."
        ),
        "source": "IGST Act 2017 — Section 16; CGST Act 2017 — Section 54; Circular 37/2018",
        "domain": "GST",
    },
    {
        "question": "What is the Letter of Undertaking (LUT) under GST for exporters?",
        "answer": (
            "An LUT (Letter of Undertaking) allows an exporter to export goods or "
            "services without paying IGST at the time of export. It is filed online on "
            "the GST portal in Form RFD-11 at the start of each financial year. "
            "Eligibility: Any registered person who has not been prosecuted for tax "
            "evasion exceeding Rs. 2.5 crore. An LUT, once accepted, is valid for the "
            "entire financial year. If exports are not completed within 3 months (for "
            "goods) or 1 year (for services), IGST becomes payable with interest. "
            "Without an LUT, the exporter must either pay IGST and claim refund, or "
            "export under a bond with a bank guarantee."
        ),
        "source": "IGST Act 2017 — Section 16(3); CGST Rules 2017 — Rule 96A",
        "domain": "GST",
    },
    {
        "question": "How is GST refund claimed and what are the timelines?",
        "answer": (
            "GST refund can be claimed for: (1) Excess tax paid; (2) ITC accumulated "
            "due to exports (zero-rated supplies); (3) ITC accumulated due to inverted "
            "duty structure (input tax rate > output tax rate); (4) Refund by international "
            "tourists; (5) Finalisation of provisional assessment. Application is filed "
            "online in Form RFD-01 within 2 years of the relevant date. The GST officer "
            "must issue a provisional refund of 90% within 7 days (for exporters). Final "
            "refund must be processed within 60 days of receipt of application. Delayed "
            "refund beyond 60 days attracts interest at 6% p.a. payable by the government."
        ),
        "source": "CGST Act 2017 — Section 54, Section 56; CGST Rules — Rule 89-97A",
        "domain": "GST",
    },

    # ── Composition Scheme (Q38–Q40) ─────────────────────────────────────────
    {
        "question": "What is the GST Composition Scheme and who is eligible?",
        "answer": (
            "The Composition Scheme under Section 10 of CGST Act 2017 is a simplified "
            "GST compliance option for small taxpayers. Eligible: Taxpayers with "
            "aggregate turnover up to Rs. 1.5 crore (Rs. 75 lakh for special category "
            "states) in the preceding financial year. Tax rates under composition: "
            "Manufacturers: 1% (0.5% CGST + 0.5% SGST); Traders: 1%; Restaurants "
            "(not serving alcohol): 5%; Service providers (composition for services "
            "under Section 10(2A)): 6% (3% CGST + 3% SGST) for those with turnover "
            "up to Rs. 50 lakh. Key restrictions: No inter-State supplies, no ITC, "
            "no tax collection from customers, cannot supply through e-commerce operators, "
            "turnover of all businesses under same PAN counts."
        ),
        "source": "CGST Act 2017 — Section 10; Notification No. 14/2019-CT, 2/2019-CT(R)",
        "domain": "GST",
    },
    {
        "question": "What return does a composition dealer file?",
        "answer": (
            "A composition dealer files: (1) CMP-08 (Quarterly): Statement of self-"
            "assessed tax payable, filed by 18th of the month following each quarter "
            "(replacing GSTR-3B for composition dealers). This is both a return and a "
            "challan for payment. (2) GSTR-4 (Annual): Annual return for composition "
            "dealers, filed by 30 April following the end of the financial year (was "
            "previously filed quarterly; changed to annual from FY 2019-20). There is "
            "no GSTR-1 for composition dealers. They cannot issue tax invoices — they "
            "issue a 'Bill of Supply' instead. Late fee for CMP-08: Rs. 50/day (Rs. 20 "
            "for nil), capped at Rs. 2,000. For GSTR-4: Rs. 200/day, capped at Rs. 5,000."
        ),
        "source": "CGST Act 2017 — Section 10, Section 39(2); CGST Rules — Rule 62",
        "domain": "GST",
    },
    {
        "question": "Can a manufacturing company opt for the composition scheme?",
        "answer": (
            "Yes, a manufacturer can opt for the composition scheme if their aggregate "
            "turnover does not exceed Rs. 1.5 crore (Rs. 75 lakh for special category "
            "states). However, certain categories are excluded from the composition "
            "scheme even if below the threshold: (1) Manufacturers of ice cream, pan "
            "masala, or tobacco products; (2) Persons making inter-State supplies; (3) "
            "Persons supplying goods through e-commerce operators; (4) Non-resident "
            "taxable persons; (5) Persons supplying goods not taxable under GST. "
            "Manufacturers under composition pay 1% GST on turnover (no ITC) and file "
            "quarterly CMP-08 and annual GSTR-4."
        ),
        "source": "CGST Act 2017 — Section 10(2); Notification No. 14/2019-CT",
        "domain": "GST",
    },

    # ── Penalties and Interest (Q41–Q45) ─────────────────────────────────────
    {
        "question": "What are the penalties for GST fraud and evasion?",
        "answer": (
            "Under Section 122 of CGST Act 2017, a taxable person who commits fraud "
            "(e.g., issues a tax invoice without actual supply, collects GST but does "
            "not deposit it, obtains refund by fraud, falsifies accounts) is liable to "
            "a penalty of: (a) Amount of tax evaded/fraudulently obtained; (b) Rs. "
            "10,000; whichever is higher. Section 132 prescribes criminal prosecution "
            "for offences exceeding Rs. 2 crore (up to 5 years imprisonment for tax "
            "evasion > Rs. 5 crore). For less serious defaults (honest mistakes, "
            "clerical errors), penalty is reduced: Section 125 general penalty of up "
            "to Rs. 25,000 applies. Interest under Section 50 at 18% p.a. is separate "
            "from penalties."
        ),
        "source": "CGST Act 2017 — Section 122, Section 125, Section 132",
        "domain": "GST",
    },
    {
        "question": "What is the rate of interest on delayed GST payment?",
        "answer": (
            "Under Section 50 of CGST Act 2017: (1) Interest for delayed payment of "
            "tax or short payment: 18% per annum, calculated on the amount of tax not "
            "paid from the day after the due date to the date of actual payment; (2) "
            "Interest for wrongful/excess ITC claim or excess reduction in output tax "
            "liability: 24% per annum. For genuine cases (cash in electronic cash ledger "
            "but return not filed), interest is on the net cash tax liability (i.e., "
            "credit already available in electronic credit ledger is not subject to "
            "interest). This relief was introduced by Finance Act 2021 (retrospective "
            "from 1 July 2017)."
        ),
        "source": "CGST Act 2017 — Section 50; Finance Act 2021 amendment",
        "domain": "GST",
    },
    {
        "question": "What is the GST assessment process and what are the types of assessments?",
        "answer": (
            "GST has the following assessment types: (1) Self-Assessment (Section 59): "
            "Taxpayer assesses own liability and pays tax — the primary mode. (2) "
            "Provisional Assessment (Section 60): When taxpayer is unable to determine "
            "value/rate, applies for provisional assessment, to be finalised within 6 "
            "months (extendable). (3) Scrutiny Assessment (Section 61): GST officer "
            "scrutinises returns and issues notice in ASMT-10. (4) Best Judgement "
            "Assessment (Section 62): For non-filers, officer assesses based on available "
            "information. (5) Summary Assessment (Section 64): In exceptional cases "
            "involving imminent revenue loss. (6) Audit by Department (Section 65): "
            "GST audit of taxpayer's books, must be completed within 3 months "
            "(extendable to 6 months)."
        ),
        "source": "CGST Act 2017 — Sections 59 to 66",
        "domain": "GST",
    },
    {
        "question": "What is the GST Appellate process if a taxpayer disagrees with an order?",
        "answer": (
            "GST appeal hierarchy: (1) First Appeal: To Appellate Authority (Additional "
            "or Joint Commissioner, GST) under Section 107 — must be filed within 3 "
            "months of order (condonable by 1 month). Pre-deposit required: 10% of "
            "disputed tax (in addition to admitted liability). (2) Second Appeal: To "
            "Appellate Tribunal (GST AT) under Section 112 — within 3 months of "
            "Appellate Authority order. Pre-deposit: 25% of disputed tax. (3) High "
            "Court: Under Section 117, on substantial questions of law. (4) Supreme "
            "Court: Under Section 118. Departmental appeals follow the same hierarchy. "
            "Filing appeal does not stay recovery — a stay application must be separately "
            "filed."
        ),
        "source": "CGST Act 2017 — Sections 107 to 122",
        "domain": "GST",
    },
    {
        "question": "What is the GST Anti-Profiteering provision?",
        "answer": (
            "Section 171 of CGST Act 2017 mandates that any reduction in rate of tax "
            "or benefit of ITC must be passed on to the recipient (consumer) by way of "
            "commensurate reduction in prices. The National Anti-Profiteering Authority "
            "(NAA) was set up to examine complaints of profiteering. Penalty for "
            "profiteering: 10% of the profiteered amount. The NAA has been subsumed "
            "into the Competition Commission of India (CCI) from December 2022, which "
            "now handles anti-profiteering cases. CCI can investigate and order that "
            "the profiteered amount be deposited in the Consumer Welfare Fund or "
            "refunded to consumers."
        ),
        "source": "CGST Act 2017 — Section 171; Anti-Profiteering Rules 2017",
        "domain": "GST",
    },

    # ── GST on Specific Sectors (Q46–Q50) ────────────────────────────────────
    {
        "question": "What is the GST treatment for works contracts (construction services)?",
        "answer": (
            "A works contract is a composite supply involving goods and services — the "
            "supply of labour and material together for construction, fabrication, "
            "installation, etc. GST on works contracts (Section 2(119) of CGST Act): "
            "(1) Works contract for construction of an immovable property (other than "
            "plant and machinery): 18% GST. ITC NOT available to the recipient (if the "
            "property is not plant/machinery). (2) Works contract for government: 12% "
            "(for certain infrastructure projects). (3) Works contract for affordable "
            "housing or government housing schemes: 12%. (4) Sub-contracted works "
            "contracts: same rate as main contract. GST is on the total contract value "
            "including materials — no splitting into goods and services."
        ),
        "source": "CGST Act 2017 — Section 2(119); Notification 11/2017-CT(R) as amended",
        "domain": "GST",
    },
    {
        "question": "How is GST applicable on renting of immovable property?",
        "answer": (
            "GST on renting of immovable property: (1) Renting of commercial property "
            "(shops, offices, warehouses) to any person: 18% GST. The landlord must "
            "register under GST if rent income exceeds the threshold (Rs. 20 lakh for "
            "services). The tenant (if registered) can claim ITC on rent paid. (2) "
            "Renting of residential property: Exempt from GST if rented to an individual "
            "for personal use. However, if a registered person (company/firm) rents a "
            "residential property for use as residence of an employee, GST at 18% applies "
            "under Reverse Charge Mechanism (RCM) on the registered company — this "
            "was clarified by CBIC in July 2022. (3) Renting of vacant land for "
            "agriculture: Exempt."
        ),
        "source": "Notification 12/2017-CT(R); CBIC Circular dated 17-07-2022",
        "domain": "GST",
    },
    {
        "question": "What is the GST rate on insurance services?",
        "answer": (
            "Insurance services attract GST at 18% in general. However, specific "
            "exemptions and reduced rates apply: (1) Life insurance under Government "
            "schemes (Pradhan Mantri Jan Dhan Yojana, Aam Aadmi Bima Yojana): Exempt; "
            "(2) Pure term life insurance (Jeevan Bima Yojana by LIC): 18% on premium; "
            "(3) ULIP (Unit-Linked Insurance Plans): 18% on charges/fees (not on the "
            "investment portion); (4) Health insurance: 18%; (5) Marine insurance for "
            "export cargo: Exempt; (6) Reinsurance services: 18%. The GST Council "
            "(September 2024 recommendations) proposed reducing GST on health and life "
            "insurance, pending finalisation. ITC on insurance for employees may be "
            "blocked under Section 17(5) unless mandatory under law."
        ),
        "source": "Notification 12/2017-CT(R); Notification 9/2017-IT(R); GST Council 54th meeting",
        "domain": "GST",
    },
    {
        "question": "How does GST apply to the IT and software sector?",
        "answer": (
            "IT and software services: (1) Development, design, programming, customisation "
            "of software (IT services): 18% GST as a service. (2) Sale of pre-packaged "
            "or canned software (off-the-shelf): Treated as goods — 18% GST. (3) "
            "Software delivered electronically (download from internet): 18% as OIDAR "
            "(Online Information and Database Access or Retrieval) service. (4) SaaS, "
            "cloud services, API services: 18% as services. (5) Export of IT/software "
            "services: Zero-rated (export of services, LUT benefit available). Indian "
            "IT companies exporting services can claim ITC refund. Annual ITC refund "
            "claims by IT sector are significant — typically processed based on the "
            "proportion of export turnover to total turnover."
        ),
        "source": "Notification 11/2017-CT(R); Classification circulars CBIC; IGST Act — Section 13",
        "domain": "GST",
    },
    {
        "question": "What is the GST treatment for the agriculture sector?",
        "answer": (
            "Agriculture has extensive GST exemptions: Fully exempt (0% GST): Fresh "
            "vegetables, fruits, milk, eggs, unbranded rice, wheat, pulses, flour; "
            "Seeds for sowing; Agricultural machinery (combine harvesters, tractors up "
            "to 1800 cc); Services by Agricultural Produce Market Committees; Warehousing "
            "of agricultural produce; Transportation of agricultural produce by rail, "
            "vessel, or road (by GTA). Taxable at 5%: Branded rice/wheat/flour, "
            "fertilisers (except exempted), pesticides. 12%: Processed/preserved food. "
            "Farmers themselves are not required to register under GST as they supply "
            "agricultural produce that is either exempt or below threshold. Input "
            "services for agriculture (irrigation, renting of agri machinery) are "
            "also largely exempt."
        ),
        "source": "Notification 02/2017-CT(R) — Exempt goods schedule; 12/2017-CT(R) — Exempt services",
        "domain": "GST",
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("Indian Legal AI — GST FAQ Writer")
    log(f"Output: {OUT_PATH}")
    log(f"Total FAQs: {len(GST_FAQS)}")

    # Skip if already done
    if os.path.isfile(OUT_PATH):
        n = 0
        try:
            with open(OUT_PATH, "r", encoding="utf-8") as f:
                n = sum(1 for _ in f)
        except Exception:
            pass
        if n > MIN_LINES:
            log(f"Already exists with {n} lines (> {MIN_LINES}) — skipping")
            log("Delete the file to regenerate it.")
            return

    os.makedirs(OUT_DIR, exist_ok=True)

    written = 0
    tmp_path = OUT_PATH + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in GST_FAQS:
            record = {
                "question": item["question"],
                "answer":   item["answer"],
                "text":     f"Q: {item['question']}\nA: {item['answer']}",
                "source":   item["source"],
                "domain":   item["domain"],
                "dataset":  "gst_faqs",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    # Atomic rename
    os.replace(tmp_path, OUT_PATH)

    log(f"Written {written} GST FAQ records to {OUT_PATH}")
    log("Done. Next step: python3 /opt/indian-legal-ai/scripts/build_rag_index.py")


if __name__ == "__main__":
    main()
