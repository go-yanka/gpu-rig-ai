-- tariff.db schema
-- Source: D:\_gpu_rig_ai\consults\p2_1_tariff_db_schema.md
-- SQLite 3

PRAGMA foreign_keys = ON;

-- 1. Master code list (HSN goods + SAC services, all granularities)
CREATE TABLE IF NOT EXISTS codes (
  code           TEXT PRIMARY KEY,
  code_type      TEXT NOT NULL,             -- 'HSN' | 'SAC'
  level          INTEGER NOT NULL,          -- 2/4/6/8 for HSN, 4/6 for SAC
  parent_code    TEXT REFERENCES codes(code),
  description    TEXT NOT NULL,
  chapter        INTEGER                    -- 1..99 HSN chapter
);

-- 2. Notifications (provenance spine)
CREATE TABLE IF NOT EXISTS notifications (
  notif_id       TEXT PRIMARY KEY,          -- '01/2017-CT(R)', '50/2017-Cus'
  series         TEXT NOT NULL,             -- 'CT(R)','IT(R)','Cus','Cus(ADD)','Comp-Cess'
  number         INTEGER NOT NULL,
  year           INTEGER NOT NULL,
  issued_on      DATE NOT NULL,
  effective_from DATE NOT NULL,
  superseded_by  TEXT REFERENCES notifications(notif_id),
  title          TEXT,
  source_doc_id  TEXT
);

-- 3. Rate history (bi-temporal: one row per (code, levy, period))
CREATE TABLE IF NOT EXISTS rates (
  rate_id        INTEGER PRIMARY KEY,
  code           TEXT NOT NULL REFERENCES codes(code),
  levy_type      TEXT NOT NULL,             -- 'CGST','SGST','IGST','BCD','SWS','COMP_CESS','AIDC'
  rate_pct       REAL,
  rate_specific  TEXT,
  condition_no   TEXT,
  schedule       TEXT,                      -- Sch I/II/III/IV/V/VI for 01/2017-CT(R)
  sno            INTEGER,
  effective_from DATE NOT NULL,
  effective_to   DATE,
  notif_id       TEXT NOT NULL REFERENCES notifications(notif_id),
  amended_by     TEXT REFERENCES notifications(notif_id)
);

-- 4. List memberships (RCM, inverted-duty, exempt, nil-rated)
CREATE TABLE IF NOT EXISTS list_membership (
  id             INTEGER PRIMARY KEY,
  code           TEXT REFERENCES codes(code),
  list_type      TEXT NOT NULL,             -- 'RCM','INVERTED_DUTY','EXEMPT','NIL','COMP_CESS_LEVIABLE'
  sno            INTEGER,
  description    TEXT,
  notif_id       TEXT NOT NULL REFERENCES notifications(notif_id),
  effective_from DATE NOT NULL,
  effective_to   DATE
);

-- 5. Exemption entries (50/2017-Cus style S.No. table)
CREATE TABLE IF NOT EXISTS exemptions (
  id             INTEGER PRIMARY KEY,
  notif_id       TEXT NOT NULL REFERENCES notifications(notif_id),
  sno            INTEGER NOT NULL,
  code           TEXT REFERENCES codes(code),
  description    TEXT NOT NULL,
  std_rate       TEXT,
  igst_rate      TEXT,
  condition_no   TEXT,
  effective_from DATE NOT NULL,
  effective_to   DATE,
  UNIQUE(notif_id, sno, code)
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_rates_code_date ON rates(code, effective_from, effective_to);
CREATE INDEX IF NOT EXISTS idx_rates_notif ON rates(notif_id);
CREATE INDEX IF NOT EXISTS idx_list_code_type ON list_membership(list_type, code);
CREATE INDEX IF NOT EXISTS idx_exempt_notif_sno ON exemptions(notif_id, sno);
CREATE INDEX IF NOT EXISTS idx_codes_chapter ON codes(chapter, level);
