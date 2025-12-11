# Medical Billing ML - Notebook 2: Load Synthea Data
# Prerequisites: Run notebook 01 first to create database schema

# Download Synthea Sample Data
!mkdir -p /work/synthea_data
!wget -q -O /work/synthea_data/synthea_sample.zip \
    "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip"
!unzip -q -o /work/synthea_data/synthea_sample.zip -d /work/synthea_data/

print("✅ Synthea data downloaded to /work/synthea_data/")
!ls /work/synthea_data/csv/

# Connect to Database
import os
from sqlalchemy import create_engine, text
import pandas as pd

DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
if not DATABASE_URL:
    raise ValueError("VERCEL_POSTGRES_URL not found! Add it to Project Settings → Environment Variables")

engine = create_engine(DATABASE_URL)

# Test connection
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM claims"))
        existing_claims = result.fetchone()[0]
        print(f"✅ Connected! Current claims in database: {existing_claims}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    raise

# Load Claims Data
print("\n📥 Loading claims data...")
encounters_df = pd.read_csv('/work/synthea_data/csv/encounters.csv')
print(f"   Found {len(encounters_df):,} encounters in Synthea dataset")

claims_data = pd.DataFrame({
    'patient_id': encounters_df['PATIENT'].apply(lambda x: hash(x) % 10**9),
    'provider_id': encounters_df['PROVIDER'].apply(lambda x: hash(x) % 10**6 if pd.notna(x) else None),
    'service_date': pd.to_datetime(encounters_df['START']).dt.date,
    'claim_type': encounters_df['ENCOUNTERCLASS'].map({
        'inpatient': 'inpatient',
        'outpatient': 'outpatient',
        'ambulatory': 'outpatient',
        'emergency': 'inpatient',
        'urgentcare': 'outpatient',
        'wellness': 'outpatient'
    }).fillna('outpatient'),
    'total_charge': encounters_df['TOTAL_CLAIM_COST'].fillna(0),
    'total_paid': encounters_df['PAYER_COVERAGE'].fillna(0)
})

# Load first 5000 claims
claims_to_load = claims_data.head(5000)
claims_to_load.to_sql('claims', engine, if_exists='append', index=False, method='multi', chunksize=500)
print(f"✅ Loaded {len(claims_to_load):,} claims into database")

# Load Diagnoses (Conditions)
print("\n📥 Loading diagnoses data...")
conditions_df = pd.read_csv('/work/synthea_data/csv/conditions.csv')
print(f"   Found {len(conditions_df):,} conditions in Synthea dataset")

# Get claim IDs for matching
with engine.connect() as conn:
    result = conn.execute(text("SELECT claim_id, patient_id, service_date FROM claims"))
    claims_lookup = pd.DataFrame(result.fetchall(), columns=['claim_id', 'patient_id', 'service_date'])

# Prepare diagnoses data
conditions_df['patient_id'] = conditions_df['PATIENT'].apply(lambda x: hash(x) % 10**9)
conditions_df['start_date'] = pd.to_datetime(conditions_df['START']).dt.date

# Merge with claims
merged = conditions_df.merge(
    claims_lookup,
    left_on=['patient_id', 'start_date'],
    right_on=['patient_id', 'service_date'],
    how='inner'
)

diagnoses_data = pd.DataFrame({
    'claim_id': merged['claim_id'],
    'icd_code': merged['CODE'].astype(str),
    'icd_version': 10,
    'is_primary': True,
    'sequence_num': 1
}).drop_duplicates(subset=['claim_id', 'icd_code'])

# Load first 10000 diagnoses
diagnoses_to_load = diagnoses_data.head(10000)
diagnoses_to_load.to_sql('diagnoses', engine, if_exists='append', index=False, method='multi', chunksize=500)
print(f"✅ Loaded {len(diagnoses_to_load):,} diagnoses into database")

# Verify Data Load
summary = pd.read_sql("""
    SELECT
        (SELECT COUNT(*) FROM claims) as total_claims,
        (SELECT COUNT(*) FROM diagnoses) as total_diagnoses,
        (SELECT AVG(total_charge) FROM claims) as avg_charge,
        (SELECT MAX(total_charge) FROM claims) as max_charge
""", engine)

print("\n" + "="*50)
print("📊 DATABASE SUMMARY")
print("="*50)
print(f"Total Claims:    {summary['total_claims'].iloc[0]:,}")
print(f"Total Diagnoses: {summary['total_diagnoses'].iloc[0]:,}")
print(f"Avg Charge:      ${summary['avg_charge'].iloc[0]:,.2f}")
print(f"Max Charge:      ${summary['max_charge'].iloc[0]:,.2f}")
print("="*50)
