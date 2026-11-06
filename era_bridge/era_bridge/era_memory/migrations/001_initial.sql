-- ERA persistent memory: initial schema (version 1)
CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    type              TEXT NOT NULL,
    category          TEXT NOT NULL DEFAULT 'fact',
    content           TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    slot              TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    source_kind       TEXT NOT NULL DEFAULT 'chat',
    conversation_id   TEXT,
    source_ref        TEXT,
    importance        REAL NOT NULL DEFAULT 0.5,
    confidence        REAL NOT NULL DEFAULT 0.9,
    status            TEXT NOT NULL DEFAULT 'active',
    superseded_by     TEXT,
    conflict_with     TEXT,
    sensitivity       TEXT NOT NULL DEFAULT 'normal',
    share_with_model  INTEGER NOT NULL DEFAULT 1,
    project_id        TEXT,
    times_seen        INTEGER NOT NULL DEFAULT 1,
    embedding         BLOB,
    embedding_model   TEXT,
    metadata          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_mem_status_type ON memories(status, type);
CREATE INDEX IF NOT EXISTS idx_mem_project     ON memories(project_id, status);
CREATE INDEX IF NOT EXISTS idx_mem_slot        ON memories(slot, status);
CREATE INDEX IF NOT EXISTS idx_mem_hash        ON memories(content_hash, status);
CREATE INDEX IF NOT EXISTS idx_mem_superseded  ON memories(superseded_by);

CREATE TABLE IF NOT EXISTS conversations (
    id                 TEXT PRIMARY KEY,
    title              TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    project_id         TEXT,
    active_task        TEXT,
    summary            TEXT,
    summary_updated_at TEXT,
    turn_count         INTEGER NOT NULL DEFAULT 0,
    closed_at          TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    aliases            TEXT NOT NULL DEFAULT '[]',
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    summary            TEXT,
    summary_updated_at TEXT,
    status             TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conversation_id, id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
