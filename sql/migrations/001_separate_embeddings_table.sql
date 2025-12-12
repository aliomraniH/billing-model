-- Migration: Separate embeddings into dedicated table
-- This improves architecture by:
-- - Allowing multiple embedding models per note
-- - Better indexing and query performance
-- - Cleaner separation of concerns

-- Step 1: Create the new embeddings table
CREATE TABLE IF NOT EXISTS clinical_note_embeddings (
    embedding_id BIGSERIAL PRIMARY KEY,
    note_id BIGINT NOT NULL REFERENCES clinical_notes(note_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,  -- e.g., 'sentence-transformers/all-MiniLM-L6-v2'
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(note_id, model_name)  -- One embedding per note per model
);

COMMENT ON TABLE clinical_note_embeddings IS 'Vector embeddings for clinical notes, supports multiple models';
COMMENT ON COLUMN clinical_note_embeddings.model_name IS 'HuggingFace model identifier used to generate embedding';

-- Create index for similarity search (using cosine distance operator)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
ON clinical_note_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Step 2: Add unique constraint to clinical_notes for ON CONFLICT support
-- This allows upsert operations on (claim_id, note_type)
ALTER TABLE clinical_notes
ADD CONSTRAINT unique_claim_note_type UNIQUE (claim_id, note_type);

-- Step 3: Migrate existing embeddings (if any) to new table
-- This assumes the model was 'sentence-transformers/all-MiniLM-L6-v2'
INSERT INTO clinical_note_embeddings (note_id, model_name, embedding)
SELECT
    note_id,
    'sentence-transformers/all-MiniLM-L6-v2' as model_name,
    embedding
FROM clinical_notes
WHERE embedding IS NOT NULL
ON CONFLICT (note_id, model_name) DO UPDATE
SET embedding = EXCLUDED.embedding,
    updated_at = NOW();

-- Step 4: (Optional) Drop the embedding column from clinical_notes
-- Uncomment if you want to fully migrate to the new schema
-- ALTER TABLE clinical_notes DROP COLUMN IF EXISTS embedding;

COMMENT ON CONSTRAINT unique_claim_note_type ON clinical_notes
IS 'Ensures one note per claim per note type, enables UPSERT operations';
