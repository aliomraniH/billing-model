# Medical Billing ML - Notebook 1: Setup & Connect to Database
# Run this in Deepnote to set up your database schema

# Install Dependencies
!pip install -q psycopg2-binary sqlalchemy pandas numpy scikit-learn sentence-transformers datasets huggingface_hub xgboost

print("✅ Dependencies installed!")

# Connect to Vercel Postgres
import os
from sqlalchemy import create_engine, text
import pandas as pd

# Get database URL from environment variables (set in Project Settings)
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')

if not DATABASE_URL:
    raise ValueError("""
    ❌ VERCEL_POSTGRES_URL not found!

    📝 To fix:
    1. Click ⚙️ (Settings) at top right
    2. Go to 'Environment variables'
    3. Click '+ Add variable'
    4. Name: VERCEL_POSTGRES_URL
    5. Value: postgresql://user:pass@host:5432/db?sslmode=require
       ⚠️  Make sure it's DIRECT connection (no -pooler in URL)
    6. Click 'Create'
    """)

# Verify it's a direct connection (not pooled)
if '-pooler' in DATABASE_URL:
    raise ValueError("""
    ⚠️  POOLED connection detected! This won't work with pgvector.

    📝 To fix:
    1. Go to Vercel Dashboard → Your Database → Settings
    2. Find 'POSTGRES_URL_NON_POOLING' (not POSTGRES_URL)
    3. Copy that URL (should have :5432, no -pooler)
    4. Update VERCEL_POSTGRES_URL in Deepnote environment variables
    """)

# Create engine
engine = create_engine(DATABASE_URL)

# Test connection
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        print(f"✅ Connection successful!")
        print(f"📊 Database: {version[:60]}...")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    raise

# Enable pgvector & Create Schema
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
    icd_code VARCHAR(50) NOT NULL,
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

# Execute schema creation
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

print("📋 Tables created:", ', '.join(tables_df['table_name'].tolist()))
print(f"📊 Total tables: {len(tables_df)}")
