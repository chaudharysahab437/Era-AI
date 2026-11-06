-- Record which provider / model / dimension produced each embedding, so vectors
-- from different configurations are never compared with each other.
ALTER TABLE memories ADD COLUMN embedding_provider TEXT;
ALTER TABLE memories ADD COLUMN embedding_dim INTEGER;
UPDATE memories SET embedding_provider = embedding_model,
                    embedding_dim = length(embedding) / 4
 WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mem_embedkey ON memories(embedding_provider, embedding_model, embedding_dim, status);
