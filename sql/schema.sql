-- Medical Billing ML Database Schema
-- Designed for Vercel Postgres (Neon) with pgvector extension
-- Run this schema after enabling pgvector: CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Claims: Main billing claims table
CREATE TABLE IF NOT EXISTS claims (
    claim_id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    provider_id BIGINT,
    service_date DATE NOT NULL,
    claim_type VARCHAR(20),  -- 'inpatient', 'outpatient', 'professional'
    total_charge NUMERIC(12,2),
    total_paid NUMERIC(12,2),
    is_outlier BOOLEAN DEFAULT FALSE,
    outlier_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE claims IS 'Main billing claims with outlier detection results';
COMMENT ON COLUMN claims.claim_type IS 'Type of claim: inpatient, outpatient, or professional';
COMMENT ON COLUMN claims.is_outlier IS 'Binary flag set by outlier detection model';
COMMENT ON COLUMN claims.outlier_score IS 'Anomaly score from Isolation Forest (-1 to 1, higher = more anomalous)';

-- Diagnoses: ICD-10 diagnosis codes linked to claims
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT REFERENCES claims(claim_id) ON DELETE CASCADE,
    icd_code VARCHAR(10) NOT NULL,
    icd_version INTEGER DEFAULT 10,
    is_primary BOOLEAN DEFAULT FALSE,
    sequence_num INTEGER
);

COMMENT ON TABLE diagnoses IS 'ICD diagnosis codes associated with claims';
COMMENT ON COLUMN diagnoses.is_primary IS 'TRUE if this is the principal diagnosis';
COMMENT ON COLUMN diagnoses.sequence_num IS 'Order of diagnosis (1 = primary)';

-- Procedures: CPT and HCPCS procedure codes
CREATE TABLE IF NOT EXISTS procedures (
    procedure_id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT REFERENCES claims(claim_id) ON DELETE CASCADE,
    procedure_code VARCHAR(10) NOT NULL,
    code_type VARCHAR(10),  -- 'CPT', 'HCPCS'
    units INTEGER DEFAULT 1,
    charge_amount NUMERIC(12,2)
);

COMMENT ON TABLE procedures IS 'CPT/HCPCS procedure codes billed on claims';
COMMENT ON COLUMN procedures.code_type IS 'CPT (physician services) or HCPCS (supplies/equipment)';
COMMENT ON COLUMN procedures.units IS 'Number of times procedure was performed';

-- Clinical Notes: Text documentation with embeddings
CREATE TABLE IF NOT EXISTS clinical_notes (
    note_id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT REFERENCES claims(claim_id) ON DELETE CASCADE,
    note_type VARCHAR(50),  -- 'discharge_summary', 'progress_note', 'radiology', etc.
    note_text TEXT,
    embedding VECTOR(384),  -- all-MiniLM-L6-v2 embeddings (384 dimensions)
    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE clinical_notes IS 'Clinical documentation with semantic embeddings for similarity search';
COMMENT ON COLUMN clinical_notes.embedding IS 'Vector embedding from all-MiniLM-L6-v2 (384-dim)';

-- Predictions: ML model outputs
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT REFERENCES claims(claim_id) ON DELETE CASCADE,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    prediction_type VARCHAR(50),  -- 'outlier', 'icd_suggestion', 'validity_check'
    prediction_value JSONB,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE predictions IS 'Model predictions stored for audit and comparison';
COMMENT ON COLUMN predictions.prediction_value IS 'JSON structure containing model output (flexible schema)';

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Claims indexes
CREATE INDEX IF NOT EXISTS idx_claims_patient ON claims(patient_id);
CREATE INDEX IF NOT EXISTS idx_claims_date ON claims(service_date);
CREATE INDEX IF NOT EXISTS idx_claims_outlier ON claims(is_outlier) WHERE is_outlier = TRUE;
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);

-- Diagnoses indexes
CREATE INDEX IF NOT EXISTS idx_diagnoses_claim ON diagnoses(claim_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_code ON diagnoses(icd_code);
CREATE INDEX IF NOT EXISTS idx_diagnoses_primary ON diagnoses(claim_id, is_primary) WHERE is_primary = TRUE;

-- Procedures indexes
CREATE INDEX IF NOT EXISTS idx_procedures_claim ON procedures(claim_id);
CREATE INDEX IF NOT EXISTS idx_procedures_code ON procedures(procedure_code);

-- Clinical notes indexes
CREATE INDEX IF NOT EXISTS idx_notes_claim ON clinical_notes(claim_id);
CREATE INDEX IF NOT EXISTS idx_notes_type ON clinical_notes(note_type);

-- pgvector cosine similarity index (HNSW for fast nearest neighbor)
-- Only create if you have > 1000 notes, otherwise sequential scan is faster
-- CREATE INDEX IF NOT EXISTS idx_notes_embedding ON clinical_notes USING hnsw (embedding vector_cosine_ops);

-- Predictions indexes
CREATE INDEX IF NOT EXISTS idx_predictions_claim ON predictions(claim_id);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name, model_version);

-- ============================================================================
-- USEFUL QUERIES
-- ============================================================================

-- Example: Find top 10 most expensive outliers
-- SELECT claim_id, total_charge, outlier_score
-- FROM claims
-- WHERE is_outlier = TRUE
-- ORDER BY total_charge DESC
-- LIMIT 10;

-- Example: Get all diagnoses for a specific claim
-- SELECT d.icd_code, d.is_primary, d.sequence_num
-- FROM diagnoses d
-- WHERE d.claim_id = 12345
-- ORDER BY d.sequence_num;

-- Example: Semantic search for similar clinical notes
-- WITH query_embedding AS (
--     SELECT embedding FROM clinical_notes WHERE note_id = 1
-- )
-- SELECT note_id, note_text, 1 - (embedding <=> query_embedding.embedding) AS similarity
-- FROM clinical_notes, query_embedding
-- WHERE embedding IS NOT NULL
-- ORDER BY embedding <=> query_embedding.embedding
-- LIMIT 10;

-- Example: Claims with high charges but low payment ratios
-- SELECT claim_id, total_charge, total_paid,
--        (total_paid / NULLIF(total_charge, 0)) AS payment_ratio
-- FROM claims
-- WHERE total_charge > 10000 AND (total_paid / NULLIF(total_charge, 0)) < 0.5
-- ORDER BY total_charge DESC;
