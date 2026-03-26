CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_textsearch CASCADE;

CREATE TABLE IF NOT EXISTS articles (
    pmid      INTEGER PRIMARY KEY,
    title     TEXT    NOT NULL,
    abstract  TEXT,
    full_text TEXT    GENERATED ALWAYS AS (
        title || ' ' || COALESCE(abstract, '')
    ) STORED,
    embedding vector(1024)
);
