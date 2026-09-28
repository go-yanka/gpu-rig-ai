"""Tests for amendment_graph, using sentences taken verbatim from CBIC chunks."""
import json

import amendment_graph as ag


def keys(edges):
    return {(e.src, e.rel, e.dst) for e in edges}


def test_series_normalisation():
    assert ag.parse_key('35/2020-Central Tax') == '35/2020-CT'
    assert ag.parse_key('1/2017-Central Tax (Rate)') == '1/2017-CT-RATE'
    assert ag.parse_key('31/86-Customs') == '31/1986-CUS'
    assert ag.parse_key('20/2004-Cus (N.T.)') == '20/2004-CUS-NT'
    assert ag.parse_key('4/2006- Central Excise') == '4/2006-CE'
    assert ag.parse_key('25/2005-CE(NT)') == '25/2005-CE-NT'
    assert ag.parse_key('56/2016-Customs (ADD)') == '56/2016-CUS-ADD'
    assert ag.parse_key('9/2017-Integrated Tax (Rate)') == '9/2017-IT-RATE'
    assert ag.parse_key('65/2020–Centra') is None  # truncated: ambiguous


def test_forward_amendment_with_date():
    text = ('hereby makes the following further amendments in the notification of the government of '
            'India in the ministry of Finance (Department of Revenue) No. 31/86-Customs, dated the '
            '5th February, 1986, namely')
    (e,) = ag.extract_edges(text, '20/2004-CUS-NT')
    assert (e.src, e.rel, e.dst, e.dst_date) == ('20/2004-CUS-NT', 'amends', '31/1986-CUS', '1986-02-05')


def test_principal_last_amended_by_links_third_parties():
    text = ('Note: The principal notification No. 35/2020-Central Tax, dated the 3rd April, 2020 was '
            'published in the Gazette of India, Extraordinary, Part II, Section 3, Sub-section (i) vide '
            'number G.S.R. 235(E), dated the 3rd April, 2020 and was last amended by notification No. '
            '55/2020 – Central Tax, dated the 27th June, 2020')
    assert ('55/2020-CT', 'amends', '35/2020-CT') in keys(ag.extract_edges(text, '65/2020-CT'))


def test_title_rescind_and_supersede_are_separated():
    title = ('Seeks to rescind notification No. 08/2013-Customs (ADD) dated 18.04.2013, in supersession '
             'of notification No. 56/2016-Customs (ADD) dated 21.12.2016.')
    assert keys(ag.extract_edges('', '51/2017-CUS', title)) == {
        ('51/2017-CUS', 'rescinds', '8/2013-CUS-ADD'),
        ('51/2017-CUS', 'supersedes', '56/2016-CUS-ADD'),
    }


def test_passive_rescinded_vide_points_at_this_document():
    text = 'Rescinded vide Notification No. 7/2001-CE, dated 1-3-2001.'
    assert keys(ag.extract_edges(text, '19/2000-CE')) == {('7/2001-CE', 'rescinds', '19/2000-CE')}


def test_list_of_amended_notifications():
    title = ('Amends notifications No. 55/2001-Cus, dated the 16th May, 2001, No. 41/1999-Customs, '
             'dated the 28th April, 1999, No. 52/2003-Customs, dated the 31st March, 2003')
    got = keys(ag.extract_edges('', '76/2007-CUS', title))
    assert got == {('76/2007-CUS', 'amends', d) for d in ('55/2001-CUS', '41/1999-CUS', '52/2003-CUS')}


def test_unless_revoked_boilerplate_is_not_an_edge():
    text = ('The provisional anti-dumping duty imposed under this notification shall be effective for a '
            'period of six months (unless revoked, amended or superseded earlier) from the date of '
            'publication of this notification in the Official Gazette')
    assert ag.extract_edges(text, '2/2025-CUS') == []


def test_self_key_from_truncated_doc_number_uses_title_family():
    assert ag.self_key('65/2020–Centra',
                       'Seeks to amend notification no. 35/2020-Central Tax dt. 03.04.2020',
                       'Note: The principal notification No. 35/2020-Central Tax ...') == '65/2020-CT'


def test_build_and_chain(tmp_path):
    chunks = [
        {'doc_id': 'd65', 'doc_number': '65/2020–Centra',
         'title': 'Seeks to amend notification no. 35/2020-Central Tax dt. 03.04.2020',
         'text': 'Note: The principal notification No. 35/2020-Central Tax, dated the 3rd April, 2020 was '
                 'published vide number G.S.R. 235(E) and was last amended by notification No. 55/2020 – '
                 'Central Tax, dated the 27th June, 2020'},
        {'doc_id': 'd35', 'doc_number': '35/2020-Central Tax', 'title': 'Due date extension',
         'text': 'Notification No. 35/2020-Central Tax New Delhi, the 3rd April, 2020 ...'},
    ]
    src = tmp_path / 'c.jsonl'
    src.write_text('\n'.join(json.dumps(c) for c in chunks))
    db = str(tmp_path / 'g.sqlite')
    ag.build(ag.iter_jsonl(str(src)), db)
    ch = ag.Graph(db).chain('35/2020-CT')
    assert ch['doc_ids'] == ['d35']
    assert ch['date'] == '2020-04-03'
    assert [a['key'] for a in ch['amended_by']] == ['55/2020-CT', '65/2020-CT']
    assert ch['status'].startswith('In force as amended')
    assert ag.Graph(db).key_for_doc('d65') == '65/2020-CT'


def test_date_formats():
    for text, iso in [('No. 42/2021-Customs, dated the 10th day of September, 2021', '2021-09-10'),
                      ('No. 4/2006- Central Excise, dated the 1 st March, 2006', '2006-03-01'),
                      ('No. 35/2020-Central Tax dt. 03.04.2020', '2020-04-03'),
                      ('No. 7/2001-CE, dated 1-3-2001', '2001-03-01')]:
        (r,) = ag.find_refs(text)
        assert r.date == iso, text
