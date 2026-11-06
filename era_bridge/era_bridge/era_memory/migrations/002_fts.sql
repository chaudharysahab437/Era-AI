-- optional: keyword index. If this SQLite build has no FTS5, the engine
-- falls back to LIKE-based candidate search and everything still works.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize = 'unicode61'
);
