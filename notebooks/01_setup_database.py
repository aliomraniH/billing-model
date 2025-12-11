# Medical Billing ML - Notebook 1: Setup & Connect to Database
# Copy-paste this entire script into Deepnote and run cells sequentially

#@title 1️⃣ Install Dependencies (Run Once)
!pip install -q psycopg2-binary sqlalchemy pandas numpy scikit-learn sentence-transformers datasets huggingface_hub xgboost

print("✅ Dependencies installed!")

#@title 2️⃣ Connect to Vercel Postgres
import os
from sqlalchemy import create_engine, text
import pandas as pd

DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
if not DATABASE_URL:
    raise ValueError("VERCEL_POSTGRES_URL not found! Add it to Project Settings → Environment Variables")
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    result = conn.execute(text("SELECT version();"))
    print(f"✅ Connected to: {result.fetchone()[0][:50]}...")

#@title 3️⃣ Enable pgvector & Create Schema
schema_sql = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS claims (
    claim_id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    provider_id BIGINT,
    service_date DATE NOT NULL,
    claim_type VARCHAR(20),
    total_charge NUMERIC(12,2),
    total_paid NUMERIC(12,2),
    is_outlier BOOLEAN DEFAULT FALSE,
    outlier_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT REFERENCES claims(claim_id) ON DELETE CASCADE,
    icd_code VARCHAR(10) NOT NULL,
    icd_version INTEGER DEFAULT 10,
    is_primary BOOLEAN DEFAULT FALSE,
    sequence_num INTEGER
);

CREATE TABLE IF NOT EXISTS procedures (
    procedure_id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT REFERENCES claims(claim_id) ON DELETE CASCADE,
    procedure_code VARCHAR(10) NOT NULL,
    code_type VARCHAR(10),
    units INTEGER DEFAULT 1,
    charge_amount NUMERIC(12,2)
);

CREATE TABLE IF NOT EXISTS clinical_notes (
    note_id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT REFERENCES claims(claim_id) ON DELETE CASCADE,
    note_type VARCHAR(50),
    note_text TEXT,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT REFERENCES claims(claim_id) ON DELETE CASCADE,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    prediction_type VARCHAR(50),
    prediction_value JSONB,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_claims_patient ON claims(patient_id);
CREATE INDEX IF NOT EXISTS idx_claims_date ON claims(service_date);
CREATE INDEX IF NOT EXISTS idx_diagnoses_claim ON diagnoses(claim_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_code ON diagnoses(icd_code);
CREATE INDEX IF NOT EXISTS idx_procedures_claim ON procedures(claim_id);
CREATE INDEX IF NOT EXISTS idx_notes_claim ON clinical_notes(claim_id);
"""

with engine.connect() as conn:
    for statement in schema_sql.split(';'):
        if statement.strip():
            conn.execute(text(statement))
    conn.commit()

print("✅ Schema created successfully!")

# Verify tables
tables_df = pd.read_sql("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public' ORDER BY table_name
""", engine)
print("📋 Tables:", ', '.join(tables_df['table_name'].tolist()))
