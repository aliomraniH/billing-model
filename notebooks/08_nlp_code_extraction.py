"""
Notebook 08: NLP Code Extraction
=================================
Batch extract ICD-10 and HCPCS codes from clinical notes.
Creates ClinicalCodeObject for each claim.

Prerequisites:
- Run 07_nlp_knowledge_base_setup.py first
- Claims data loaded in database (from notebooks 01-02)

Time: ~2-5 minutes per 1000 claims
"""

print("🔬 Medical Billing NLP - Code Extraction")
print("=" * 60)

# %% [markdown]
# ## Step 1: Setup

# %%
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.getcwd(), '..'))

from config.settings import db_config, processing_config
from sqlalchemy import create_engine, text

engine = create_engine(db_config.url)

# %% [markdown]
# ## Step 2: Load Claims Data

# %%
print("📥 Loading claims with clinical notes...")

with engine.connect() as conn:
    # Load claims that have visit notes
    query = text("""
        SELECT claim_id, condition, visit_notes, total_charge
        FROM claims
        WHERE visit_notes IS NOT NULL AND visit_notes != ''
        LIMIT :limit
    """)

    df = pd.read_sql(query, conn, params={'limit': processing_config.max_claims_per_run})

print(f"✅ Loaded {len(df)} claims with clinical notes")
print(df.head())

# %% [markdown]
# ## Step 3: Initialize Extractor

# %%
from src.features.nlp_extractor import ClinicalCodeExtractor

extractor = ClinicalCodeExtractor(db_connection=engine)

# %% [markdown]
# ## Step 4: Extract Codes

# %%
print("\n🔄 Extracting codes from clinical notes...")

clinical_objects = extractor.batch_extract(
    df,
    text_column='visit_notes',
    claim_id_column='claim_id',
    condition_column='condition'
)

print(f"\n✅ Extracted codes for {len(clinical_objects)} claims")

# %% [markdown]
# ## Step 5: Preview Results

# %%
print("\n📋 Sample Extraction Results:")
print("-" * 60)

for obj in clinical_objects[:5]:
    print(f"\nClaim: {obj.claim_id}")
    print(f"  Diagnoses: {len(obj.extracted_diagnoses)}")
    for diag in obj.billable_diagnoses[:3]:
        print(f"    - {diag.code}: {diag.term} (conf: {diag.confidence:.2f})")
    print(f"  Procedures: {len(obj.extracted_procedures)}")
    for proc in obj.billable_procedures[:3]:
        print(f"    - {proc.code}: {proc.term} (conf: {proc.confidence:.2f})")

# %% [markdown]
# ## Step 6: Store Results in Database

# %%
print("\n💾 Storing extraction results...")

with engine.connect() as conn:
    # Create nlp_extractions table if not exists
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS nlp_extractions (
            claim_id VARCHAR(50) PRIMARY KEY,
            extraction_timestamp TIMESTAMP DEFAULT NOW(),
            extraction_json JSONB,
            diagnosis_count INT,
            procedure_count INT
        )
    """))
    conn.commit()

    # Insert results
    for obj in clinical_objects:
        conn.execute(text("""
            INSERT INTO nlp_extractions (claim_id, extraction_json, diagnosis_count, procedure_count)
            VALUES (:claim_id, :json, :diag_count, :proc_count)
            ON CONFLICT (claim_id) DO UPDATE SET
                extraction_json = EXCLUDED.extraction_json,
                extraction_timestamp = NOW(),
                diagnosis_count = EXCLUDED.diagnosis_count,
                procedure_count = EXCLUDED.procedure_count
        """), {
            'claim_id': obj.claim_id,
            'json': obj.to_json(),
            'diag_count': len(obj.billable_diagnoses),
            'proc_count': len(obj.billable_procedures)
        })

    conn.commit()

print(f"✅ Stored {len(clinical_objects)} extraction results")

# %% [markdown]
# ## Step 7: Summary Statistics

# %%
total_diagnoses = sum(len(obj.billable_diagnoses) for obj in clinical_objects)
total_procedures = sum(len(obj.billable_procedures) for obj in clinical_objects)

print("\n" + "=" * 60)
print("📊 EXTRACTION SUMMARY")
print("=" * 60)
print(f"  Claims processed: {len(clinical_objects)}")
print(f"  Total diagnoses extracted: {total_diagnoses}")
print(f"  Total procedures extracted: {total_procedures}")
print(f"  Avg diagnoses per claim: {total_diagnoses/len(clinical_objects):.1f}")
print(f"  Avg procedures per claim: {total_procedures/len(clinical_objects):.1f}")
print("\n  Next: Run 09_gap_analysis_reporting.py for validation")
