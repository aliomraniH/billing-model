# Medical Billing ML - Notebook 3: Outlier Detection Model
# Prerequisites: Run notebooks 01 and 02 first

#@title 1️⃣ Connect to Database
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

#@title 2️⃣ Prepare Features
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

#@title 3️⃣ Feature Engineering & Training
df['charge_per_diagnosis'] = df['total_charge'] / (df['num_diagnoses'] + 1)
df['charge_per_procedure'] = df['total_charge'] / (df['num_procedures'] + 1)

df_encoded = pd.get_dummies(df, columns=['claim_type'], prefix='type')
feature_cols = [c for c in df_encoded.columns if c not in ['claim_id']]

X = df_encoded[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
model.fit(X_scaled)

df['is_outlier'] = model.predict(X_scaled) == -1
df['outlier_score'] = -model.decision_function(X_scaled)

print(f"\n🔍 OUTLIER DETECTION RESULTS")
print(f"Outliers Found: {df['is_outlier'].sum():,} ({df['is_outlier'].mean()*100:.1f}%)")

#@title 4️⃣ Write Predictions to Database
with engine.connect() as conn:
    for _, row in df.iterrows():
        conn.execute(text("""
            UPDATE claims SET is_outlier = :is_outlier, outlier_score = :outlier_score
            WHERE claim_id = :claim_id
        """), {'claim_id': int(row['claim_id']), 'is_outlier': bool(row['is_outlier']),
               'outlier_score': float(row['outlier_score'])})
    conn.commit()

print(f"✅ Updated {len(df):,} claims with outlier predictions!")

#@title 5️⃣ Analyze Top Outliers
print("\n🚨 TOP 10 OUTLIERS")
top_outliers = df[df['is_outlier']].nlargest(10, 'outlier_score')[
    ['claim_id', 'total_charge', 'num_diagnoses', 'num_procedures', 'outlier_score']]
print(top_outliers.to_string(index=False))

#@title 6️⃣ Save Model
import pickle
model_artifacts = {'model': model, 'scaler': scaler, 'feature_cols': feature_cols}
with open('/work/outlier_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)
print("\n✅ Model saved to /work/outlier_model.pkl")
