-- tariff_schema.sql — CBIC tariff rate DB (A4 patch, sentinel tariff_v1)
-- Target file on rig: /opt/indian-legal-ai/tariff.db
-- Apply with: sqlite3 /opt/indian-legal-ai/tariff.db < tariff_schema.sql

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tariff_rate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hsn TEXT,                        -- HSN code (goods)
    sac TEXT,                        -- SAC code (services)
    description TEXT NOT NULL,
    rate_igst REAL,                  -- percent
    rate_cgst REAL,
    rate_sgst REAL,
    rate_cess TEXT,                  -- can be complex (e.g., '12% + Rs 4170/1000'), store as text
    effective_from TEXT NOT NULL,    -- ISO date
    effective_to TEXT,               -- NULL = currently in force
    notification_id TEXT,            -- e.g., '1/2017-CT(Rate)'
    notification_date TEXT,
    doc_page INTEGER,
    pdf_path TEXT,
    schedule TEXT,                   -- e.g., 'Schedule I'
    chapter TEXT,                    -- HS chapter
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tariff_hsn ON tariff_rate(hsn);
CREATE INDEX IF NOT EXISTS idx_tariff_sac ON tariff_rate(sac);
CREATE INDEX IF NOT EXISTS idx_tariff_eff_from ON tariff_rate(effective_from);
CREATE INDEX IF NOT EXISTS idx_tariff_eff_to ON tariff_rate(effective_to);
CREATE INDEX IF NOT EXISTS idx_tariff_notif ON tariff_rate(notification_id);

-- FTS5 virtual table for description / code fuzzy lookup
CREATE VIRTUAL TABLE IF NOT EXISTS tariff_fts USING fts5(
    description,
    hsn,
    sac,
    content='tariff_rate',
    content_rowid='id'
);

-- Keep FTS in sync with tariff_rate
CREATE TRIGGER IF NOT EXISTS tariff_ai AFTER INSERT ON tariff_rate BEGIN
    INSERT INTO tariff_fts(rowid, description, hsn, sac)
    VALUES (new.id, new.description, new.hsn, new.sac);
END;

CREATE TRIGGER IF NOT EXISTS tariff_ad AFTER DELETE ON tariff_rate BEGIN
    INSERT INTO tariff_fts(tariff_fts, rowid, description, hsn, sac)
    VALUES('delete', old.id, old.description, old.hsn, old.sac);
END;

CREATE TRIGGER IF NOT EXISTS tariff_au AFTER UPDATE ON tariff_rate BEGIN
    INSERT INTO tariff_fts(tariff_fts, rowid, description, hsn, sac)
    VALUES('delete', old.id, old.description, old.hsn, old.sac);
    INSERT INTO tariff_fts(rowid, description, hsn, sac)
    VALUES (new.id, new.description, new.hsn, new.sac);
END;

-- Sentinel row for deploy verification
CREATE TABLE IF NOT EXISTS tariff_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
INSERT OR REPLACE INTO tariff_meta(key, value) VALUES ('sentinel', 'tariff_v1');
INSERT OR REPLACE INTO tariff_meta(key, value) VALUES ('schema_version', '1');
