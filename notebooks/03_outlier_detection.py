# Medical Billing ML - Notebook 3: Outlier Detection Model
# Prerequisites: Run notebooks 01 and 02 first

# Connect to Database
import os
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
if not DATABASE_URL:
    raise ValueError("VERCEL_POSTGRES_URL not found! Add it to Project Settings → Environment Variables")

engine = create_engine(DATABASE_URL)

# Test connection
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM claims"))
        total_claims = result.fetchone()[0]
        print(f"✅ Connected! Found {total_claims:,} claims in database")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    raise

# Prepare Features
print("\n📊 Extracting features from claims...")
feature_query = """
WITH claim_features AS (
    SELECT
        c.claim_id, c.total_charge, c.total_paid, c.claim_type,
        CASE WHEN c.total_charge > 0 THEN c.total_paid / c.total_charge ELSE 0 END as payment_ratio,
        COUNT(DISTINCT d.diagnosis_id) as num_diagnoses,
        COUNT(DISTINCT p.procedure_id) as num_procedures
    FROM claims c
    LEFT JOIN diagnoses d ON c.claim_id = d.claim_id
    LEFT JOIN procedures p ON c.claim_id = p.claim_id
    GROUP BY c.claim_id, c.total_charge, c.total_paid, c.claim_type
)
SELECT * FROM claim_features WHERE total_charge > 0
"""

df = pd.read_sql(feature_query, engine)
print(f"✅ Loaded {len(df):,} claims with features")

# Feature Engineering & Training
print("\n🔧 Engineering features...")
df['charge_per_diagnosis'] = df['total_charge'] / (df['num_diagnoses'] + 1)
df['charge_per_procedure'] = df['total_charge'] / (df['num_procedures'] + 1)

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['claim_type'], prefix='type')
feature_cols = [c for c in df_encoded.columns if c not in ['claim_id']]
print(f"   Total features: {len(feature_cols)}")

# Prepare data for model
X = df_encoded[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
print("\n🤖 Training Isolation Forest model...")
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expect 5% outliers
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled)

# Generate predictions
df['is_outlier'] = model.predict(X_scaled) == -1
df['outlier_score'] = -model.decision_function(X_scaled)

print("\n" + "="*50)
print("🔍 OUTLIER DETECTION RESULTS")
print("="*50)
print(f"Total Claims:     {len(df):,}")
print(f"Outliers Found:   {df['is_outlier'].sum():,} ({df['is_outlier'].mean()*100:.1f}%)")
print(f"Normal Claims:    {(~df['is_outlier']).sum():,}")
print("="*50)

# Write Predictions to Database
print("\n💾 Writing predictions to database...")
with engine.begin() as conn:  # Changed from engine.connect() to engine.begin()
    for _, row in df.iterrows():
        conn.execute(text("""
            UPDATE claims
            SET is_outlier = :is_outlier, outlier_score = :outlier_score
            WHERE claim_id = :claim_id
        """), {
            'claim_id': int(row['claim_id']),
            'is_outlier': bool(row['is_outlier']),
            'outlier_score': float(row['outlier_score'])
        })
    # Auto-commits on context exit

print(f"✅ Updated {len(df):,} claims with outlier predictions")

# Analyze Top Outliers
print("\n" + "="*50)
print("🚨 TOP 10 MOST ANOMALOUS CLAIMS")
print("="*50)
top_outliers = df[df['is_outlier']].nlargest(10, 'outlier_score')[
    ['claim_id', 'total_charge', 'num_diagnoses', 'num_procedures', 'outlier_score']
]
print(top_outliers.to_string(index=False))
print("="*50)

# Save Model
import pickle

model_artifacts = {
    'model': model,
    'scaler': scaler,
    'feature_cols': feature_cols
}

model_path = '/work/outlier_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_artifacts, f)

print(f"\n✅ Model saved to {model_path}")
print(f"📦 Model size: {os.path.getsize(model_path) / 1024:.1f} KB")
