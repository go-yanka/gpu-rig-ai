#!/usr/bin/env python3
"""Hand-authored synthetic G1 queries for the 10 smoke docs.

Each query targets content actually present in the doc's canonical chunks.
Grounded in the chunk previews extracted from ingest_manifest_v2.sqlite.
Authored 2026-04-24 after the filter_training_pairs heuristic gold yielded
recall@10=0.69 (noisy because the 5-gram overlap "best-doc" attribution
was unreliable — CGST/Customs boilerplate bled across docs).
"""
import json, random, sys

SYN = {
    "cbic-act-msts:1000001": [  # Customs Act, 1962
        "What does Section 1 of the Customs Act 1962 say about its short title, extent and commencement?",
        "Under the Customs Act 1962, who are the classes of officers of customs and who appoints them?",
        "What is Section 28F of the Customs Act regarding the Authority for Advance Rulings?",
        "What does Section 28H of the Customs Act provide about application for advance ruling?",
        "When does the Customs Act make an advance ruling void in certain circumstances (Section 28K)?",
        "What is Section 70 of the Customs Act about allowance in case of volatile goods in warehouses?",
        "What does Section 72 of the Customs Act say about goods improperly removed from warehouse?",
        "What does Section 114A of the Customs Act provide about penalty for short-levy or non-levy of duty?",
        "What is Section 114AB of the Customs Act regarding penalty for obtaining an instrument by fraud?",
        "What does Section 115 of the Customs Act say about confiscation of conveyances?",
        "What is Section 133 of the Customs Act about obstruction of an officer of customs?",
        "What does Section 135 of the Customs Act provide about evasion of duty or prohibitions?",
    ],
    "cbic-act-msts:1000006": [  # CGST Act, 2017
        "What does Section 68 of the CGST Act 2017 say about inspection of goods in movement?",
        "What is Section 69 of the CGST Act about the power to arrest?",
        "What does Section 70 of the CGST Act 2017 provide for summoning persons to give evidence and produce documents?",
        "What is Section 71 of the CGST Act about access to business premises?",
        "Under the CGST Act, what is Section 73 about determination of tax not paid or short paid?",
        "What does Section 125 of the CGST Act provide as general penalty?",
        "What is Section 129 of the CGST Act about detention, seizure and release of goods and conveyances in transit?",
        "What does Section 130 of the CGST Act 2017 say about confiscation of goods or conveyances?",
        "How does the CGST Act 2017 define 'capital goods' (Section 2(19))?",
        "What is the definition of 'deemed exports' under Section 2(39) of the CGST Act 2017?",
        "What is the definition of 'document' under Section 2(41) of the CGST Act referring to the IT Act 2000?",
        "How has the CGST (Extension to Jammu and Kashmir) Ordinance 2017 been repealed?",
    ],
    "cbic-act-msts:1000007": [  # Central Excise Act, 1944
        "What does Section 1 of the Central Excise Act 1944 say about its short title, extent and commencement?",
        "Under Central Excise Act Section 2(a), how is 'Adjudicating authority' defined?",
        "What is Section 23A of the Central Excise Act about advance ruling definitions?",
        "What does Section 23C of the Central Excise Act provide about application for advance ruling?",
        "What is the procedure on receipt of an advance ruling application under Section 23D of the Central Excise Act?",
        "Under the Central Excise Act, when is an advance ruling void in certain circumstances (Section 23F)?",
        "What does Section 37D of the Central Excise Act 1944 say about rounding off of duty?",
        "What is Section 37E of the Central Excise Act about publication of information respecting persons in certain cases?",
        "What does Section 38 of the Central Excise Act say about publication of rules and notifications and laying of rules before Parliament?",
        "What is Section 38A of the Central Excise Act about effect of amendments of rules, notifications or orders?",
        "What does Section 3 of the Central Excise Act provide about duties specified in the First and Second Schedule of the Central Excise Tariff Act 1985?",
        "Under the Central Excise Act, what does Section 3A say about levy of duty on basis of capacity of production?",
    ],
    "cbic-act-msts:1000009": [  # Customs Tariff Act, 1975
        "What does Section 1 of the Customs Tariff Act 1975 say about short title, extent and commencement?",
        "Under the Customs Tariff Act Section 2, what are duties specified in the Schedules to be levied?",
        "What is Section 3 of the Customs Tariff Act 1975 about levy of additional duty equal to excise duty, sales tax and local taxes?",
        "Under the Customs Tariff Act, how is additional duty computed when multiple retail sale prices are declared on an imported article?",
        "What does the Customs Tariff Act say about additional duty leviable where the Central Government has fixed a tariff value for a like article?",
        "How does the Customs Tariff Act Section 3 treat alcoholic liquor imported — what excise duty reference is used?",
        "Under Section 3 of the Customs Tariff Act, what is meant by 'sales tax, value added tax, local tax or other charges' for the time being leviable on a like article?",
        "What does the Customs Tariff Act say about safeguard duty under Section 8B and 8C being excluded from the additional duty?",
        "Under the Customs Tariff Act, are countervailing duty (Section 9) and anti-dumping duty (Section 9A) included in the additional duty under sub-sections (5), (7) and (9)?",
        "What is Section 4 of the Customs Tariff Act about emergency power to increase import duties?",
        "What does the Customs Tariff Act 1975 say about its enactment date of 18 August 1975?",
        "Under the Customs Tariff Act, what is the scope of 'any article imported into India' for levy of additional duty?",
    ],
    "cbic-allied-act-dtls:1000221": [  # Medical Devices Rules, 2017
        "What is the definition of 'medical device' under the Medical Devices Rules 2017 published by authority in January 2017?",
        "How do the Medical Devices Rules 2017 define 'clinical investigation' for medical devices?",
        "What do the Medical Devices Rules 2017 say about 'clinical research organisation' as a sponsor of clinical investigation?",
        "Under the Medical Devices Rules 2017, what does 'long-term use' mean in the context of a medical device?",
        "How is 'manufacturing site' defined under the Medical Devices Rules 2017 when used by another licensee?",
        "What do the Medical Devices Rules say about performance evaluation study on collected samples?",
        "What is 'post-market surveillance' under the Medical Devices Rules 2017 for a device brought into the market?",
        "Under the Medical Devices Rules 2017, how are undefined words interpreted by reference to the Drugs and Cosmetics Act?",
        "What does the Medical Devices Rules 2017 notification say about being published in Gazette Part II Section 3(i) on 31 January 2017?",
        "Under the Medical Devices Rules, what does a 'medical device utensil' operated by electrical or human/animal body energy refer to?",
        "What is meant by the D.L. 33004/99 gazette registration number on the Medical Devices Rules 2017 notification?",
        "What do the Medical Devices Rules 2017 provide about determination and analysis of data to verify device performance?",
    ],
    "cbic-circular-msts:1001000": [  # Circular 7/2004, disposal of unclaimed cargo
        "What does Circular 7/2004 dated 28 January 2004 say about the procedure for disposal of unclaimed or uncleared cargo under Section 48 of the Customs Act 1962?",
        "Under CBEC Circular 7/2004, what role do Customs Appraisers and expert panels play in valuation of unclaimed cargo?",
        "What does CBEC Circular 7/2004 say about custodians requesting Customs Appraiser services on the valuation panel?",
        "Under Circular 7/2004, how shall sale proceeds of disposed unclaimed cargo be shared per Section 150 of the Customs Act 1962?",
        "What does CBEC Circular 7/2004 describe as a 'one-time interim administrative arrangement' for disposal of long-pending unclaimed cargo?",
        "What does Circular 7/2004 F.No. 450/97/2003-Cus.IV say about inconsistency with prior instructions?",
        "Which CBEC circular of January 2004 addresses disposal of unclaimed or uncleared cargo lying with custodians?",
        "Under Circular 7/2004, what is the referenced regulation for the Customs House Agents Licensing Regulations 1972 context?",
        "What does the 2004 Customs circular say about panel of three valuers in cases of doubt?",
        "Under Section 48 of the Customs Act 1962, what procedure does Circular 7/2004 prescribe for custodians?",
        "What department issued Circular 7/2004 dated 28 January 2004 on disposal of unclaimed cargo?",
        "What does Circular 7/2004 say about disposal being done to ensure unclaimed cargo pending for long is cleared?",
    ],
    "cbic-notification-msts:1001146": [  # Notification 2/2018 IGST rate, 25 Jan 2018
        "What does Notification No. 2/2018 Integrated Tax (Rate) dated 25 January 2018 amend?",
        "Under IGST Rate Notification 2/2018, what new serial numbers 20A and 20B under Heading 9965 were inserted for transportation of goods?",
        "What does Notification 2/2018 IGST say about substituting 'one year' with 'three years' in an existing entry?",
        "Under Integrated Tax Rate Notification 2/2018, how is a 'person in IFSC' defined — including recognition by Government or Regulator of IFSC?",
        "What does Notification 2/2018 Integrated Tax Rate say about services by way of fumigation in a warehouse of agricultural produce?",
        "Under IGST Rate Notification 2/2018, what is the exemption for admission to a planetarium where consideration is not more than Rs 500 per person?",
        "What does Notification 2/2018 IGST say about IFSC being regulated under Foreign Exchange Management (International Financial Services Centre) Regulations 2015?",
        "Which officer signed Notification No. 2/2018 Integrated Tax (Rate) and under which file number?",
        "What power was invoked under sub-section (1) of Section 6 of the Integrated Goods and Services Tax Act to issue Notification 2/2018?",
        "Under Notification 2/2018 IGST, what Heading 9985 amendment inserts item (h) for fumigation in agricultural produce warehouse?",
        "What does Notification 2/2018 IGST say about the principal notification that was published in the Gazette of India?",
        "Under IGST Rate Notification 2/2018 dated 25 January 2018, what is F.No. 354/13/2018-TRU?",
    ],
    "cbic-notification-msts:1008308": [  # Re-export Drawback Rules 1995
        "What does Notification No. 36/95-Cus(NT) dated 26 May 1995 prescribe regarding Re-export of imported goods (Drawback of Customs duties) Rules 1995?",
        "Which amending notifications modified the Re-export Drawback Rules 1995 (29/1999, 05/2003 etc.)?",
        "Under Rule 4 of the Re-export Drawback Rules 1995, what statements and declarations must an exporter make at time of export other than by post?",
        "What does Rule 7 of the Re-export Drawback Rules 1995 say about repayment of erroneous or excess drawback and interest?",
        "Under Rule 8 of the Re-export Drawback Rules 1995, what savings apply to claims for drawback on goods exported before commencement of these rules?",
        "What are the particulars an exporter must state on the shipping bill or bill of export under Re-export Drawback Rules 1995?",
        "Under the Re-export Drawback Rules 1995, on demand by an officer of customs, what is the claimant's obligation when drawback paid erroneously?",
        "What does Notification 63/1995-Cus(NT) dated 20 October 1995 do to the Re-export Drawback Rules?",
        "Under the Re-export Drawback Rules 1995, how are claims made before commencement disposed of under the savings provision?",
        "What is the purpose of the Re-export of imported goods (Drawback of Customs duties) Rules 1995 notified under M.F. (D.R.)?",
        "Under Rule 4 of the Re-export Drawback Rules, how does the exporter indicate goods are entitled to drawback?",
        "What does Notification 5/2003-Cus(NT) dated 21 January 2003 amend in the Re-export Drawback Rules 1995?",
    ],
    "cbic-others-document-msts:1000041": [  # Finance Act 1997 (gazette 14 May 1997)
        "What does the Finance Act 1997 Chapter II say about rates of income-tax for assessment year commencing 1 April 1997?",
        "Under the Finance Act 1997 published on 14 May 1997, how is surcharge computed per Paragraph E of Part I of the First Schedule?",
        "What does the Finance Act 1997 Chapter III provide under Direct Taxes regarding Income-tax?",
        "Under the Finance Act 1997, what new clause (6BB) was inserted after Section 10(6B) of the Income-tax Act from 1 April 1998?",
        "How does the Finance Act 1997 compute deduction where salary is due from more than one employer?",
        "Under the Finance Act 1997 amendment to Section 35, what Explanation was added for removal of doubts about salary?",
        "What does the Finance Act 1997 say about computing tax with reference to the rates imposed by sub-section or those specified in that Chapter/section?",
        "Under the Finance Act 1997 Paragraph A of Part III of the First Schedule, what provisions apply to the assessee's previous year?",
        "What does the Finance Act 1997 provide for a scheme of amalgamation regarding unallowed expenditure and capital sums?",
        "Under the Finance Act 1997, for transfer of a licence, how is the expenditure remaining unallowed divided by relevant previous years?",
        "What does the Finance Act 1997 gazette notification DL-33004/97 No. 40 dated 14 May 1997 / Vaisakha 1919 contain?",
        "Under the Finance Act 1997, what does Section 10(6BB) say about Government of foreign State or foreign enterprise deriving income from Indian company in aircraft operation?",
    ],
    "circular-cgst-128-Formats": [  # CGST form templates
        "What is the template for a summons issued under Section 70 of the Central Goods and Services Tax Act 2017?",
        "Under CGST form templates, what does the arrest memo require about informing someone of the arrest?",
        "What is the authorisation format under Section 174(2) read with Section 12F of the Central Excise Act 1944 for search of secreted goods?",
        "What does the CGST arrest memo form state about grounds of arrest being explained to the arrestee?",
        "What does the CGST bond template require about payment of duty fine or penalty adjudged by the Adjudicating Authority?",
        "Under CGST form templates, what must a Bank Guarantee clause bind the issuing bank to?",
        "What does the CGST summons format say about inquiry being deemed a judicial proceeding under Sections 193 and 228 IPC?",
        "In the CGST arrest memo, what is the role of the witness (name, designation, father/mother, address) on signing?",
        "Under the CGST search authorisation format, what powers are invoked under Section 12F of Central Excise Act 1944 with Section 174(2) CGST?",
        "What does the CGST summons format require about attendance for giving evidence and/or producing documents under Section 70?",
        "Under CGST templates, which circular prescribes the standard SUMMONS format under Section 70?",
        "What does the CGST form template say about the signature of the arrestee and receipt of copy of arrest memo?",
    ],
}

def build(out_path):
    queries = []
    for doc_id, qs in SYN.items():
        for i, q in enumerate(qs):
            queries.append({
                "id": f"synth_{doc_id.replace(':','_')}_{i:02d}",
                "query": q,
                "expected_doc_ids": [doc_id],
            })
    random.seed(42)
    random.shuffle(queries)
    payload = {"meta": {"kind":"hand_authored_synthetic","docs":len(SYN),"n":len(queries)},
               "queries": queries}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(queries)} synthetic queries across {len(SYN)} docs → {out_path}")

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv)>1 else "/tmp/g1_gold_synth.json")
