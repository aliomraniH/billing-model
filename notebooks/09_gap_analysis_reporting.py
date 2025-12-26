"""
Notebook 09: Gap Analysis & Reporting
======================================
Compare NLP-extracted codes against billed codes.
Identify revenue leakage and compliance risks.

Prerequisites:
- Run 08_nlp_code_extraction.py first
- Billed codes available in claims table

Time: ~1-2 minutes
"""

print("📊 Medical Billing NLP - Gap Analysis")
print("=" * 60)

# %% [markdown]
# ## Step 1: Setup

# %%
import os
import sys
import pandas as pd
import json

sys.path.insert(0, os.path.join(os.getcwd(), '..'))

from config.settings import db_config
from sqlalchemy import create_engine, text

engine = create_engine(db_config.url)

# %% [markdown]
# ## Step 2: Load Extraction Results

# %%
print("📥 Loading NLP extraction results...")

with engine.connect() as conn:
    extractions_df = pd.read_sql(text("""
        SELECT claim_id, extraction_json
        FROM nlp_extractions
    """), conn)

print(f"✅ Loaded {len(extractions_df)} extraction results")

# Reconstruct ClinicalCodeObjects
from src.features.clinical_code_object import ClinicalCodeObject

clinical_objects = []
for _, row in extractions_df.iterrows():
    obj = ClinicalCodeObject.from_dict(json.loads(row['extraction_json']))
    clinical_objects.append(obj)

# %% [markdown]
# ## Step 3: Load Billed Codes

# %%
print("📥 Loading billed codes...")

with engine.connect() as conn:
    # Get billed diagnosis codes
    billed_df = pd.read_sql(text("""
        SELECT c.claim_id,
               COALESCE(STRING_AGG(d.icd_code, ','), '') as codes
        FROM claims c
        LEFT JOIN diagnoses d ON c.claim_id = d.claim_id
        GROUP BY c.claim_id
    """), conn)

print(f"✅ Loaded billed codes for {len(billed_df)} claims")

# %% [markdown]
# ## Step 4: Run Gap Analysis

# %%
from src.features.gap_analyzer import GapAnalyzer

print("\n🔍 Running gap analysis...")

analyzer = GapAnalyzer(db_connection=engine)

gap_results = analyzer.batch_analyze(
    clinical_objects=clinical_objects,
    billed_df=billed_df,
    claim_id_column='claim_id',
    codes_column='codes'
)

print(f"✅ Analyzed {len(gap_results)} claims")

# %% [markdown]
# ## Step 5: Summary Report

# %%
print("\n" + "=" * 60)
print("📊 GAP ANALYSIS SUMMARY")
print("=" * 60)

total_claims = len(gap_results)
claims_with_issues = gap_results[gap_results['flag_count'] > 0]

print(f"\n📋 Overall Statistics:")
print(f"  Total claims analyzed: {total_claims}")
print(f"  Claims with issues: {len(claims_with_issues)} ({len(claims_with_issues)/total_claims*100:.1f}%)")

print(f"\n💰 Revenue Impact:")
total_leakage = gap_results['revenue_leakage_flags'].sum()
total_compliance = gap_results['compliance_risk_flags'].sum()
est_revenue = gap_results['revenue_impact'].sum()

print(f"  Revenue leakage flags: {total_leakage}")
print(f"  Compliance risk flags: {total_compliance}")
print(f"  Estimated revenue impact: ${est_revenue:,.0f}")

print(f"\n⚠️  Risk Distribution:")
high_risk = gap_results[gap_results['risk_score'] >= 0.5]
medium_risk = gap_results[(gap_results['risk_score'] >= 0.25) & (gap_results['risk_score'] < 0.5)]
low_risk = gap_results[gap_results['risk_score'] < 0.25]

print(f"  High risk (>=0.5): {len(high_risk)} claims")
print(f"  Medium risk (0.25-0.5): {len(medium_risk)} claims")
print(f"  Low risk (<0.25): {len(low_risk)} claims")

# %% [markdown]
# ## Step 6: Top Issues

# %%
print("\n📋 TOP 10 CLAIMS BY RISK SCORE:")
print("-" * 60)

top_issues = gap_results.nlargest(10, 'risk_score')
for _, row in top_issues.iterrows():
    print(f"\nClaim: {row['claim_id']}")
    print(f"  Risk Score: {row['risk_score']:.2f}")
    print(f"  Revenue Leakage: {row['revenue_leakage_flags']} flags")
    print(f"  Compliance Risk: {row['compliance_risk_flags']} flags")
    print(f"  Est. Impact: ${row['revenue_impact']:,.0f}")

# %% [markdown]
# ## Step 7: Store Results

# %%
print("\n💾 Storing gap analysis results...")

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS gap_analysis_results (
            claim_id VARCHAR(50) PRIMARY KEY,
            analysis_timestamp TIMESTAMP DEFAULT NOW(),
            flag_count INT,
            revenue_leakage_flags INT,
            compliance_risk_flags INT,
            revenue_impact FLOAT,
            risk_score FLOAT,
            flags_json JSONB
        )
    """))
    conn.commit()

    for _, row in gap_results.iterrows():
        conn.execute(text("""
            INSERT INTO gap_analysis_results
            (claim_id, flag_count, revenue_leakage_flags, compliance_risk_flags, revenue_impact, risk_score, flags_json)
            VALUES (:claim_id, :flag_count, :rev_flags, :comp_flags, :revenue, :risk, :flags)
            ON CONFLICT (claim_id) DO UPDATE SET
                analysis_timestamp = NOW(),
                flag_count = EXCLUDED.flag_count,
                revenue_leakage_flags = EXCLUDED.revenue_leakage_flags,
                compliance_risk_flags = EXCLUDED.compliance_risk_flags,
                revenue_impact = EXCLUDED.revenue_impact,
                risk_score = EXCLUDED.risk_score,
                flags_json = EXCLUDED.flags_json
        """), {
            'claim_id': row['claim_id'],
            'flag_count': row['flag_count'],
            'rev_flags': row['revenue_leakage_flags'],
            'comp_flags': row['compliance_risk_flags'],
            'revenue': row['revenue_impact'],
            'risk': row['risk_score'],
            'flags': json.dumps(row['flags_json'])
        })

    conn.commit()

print(f"✅ Stored {len(gap_results)} gap analysis results")

# %% [markdown]
# ## Step 8: Export Report

# %%
# Export to CSV for review
report_filename = 'gap_analysis_report.csv'
gap_results.to_csv(report_filename, index=False)
print(f"\n📄 Report exported to: {report_filename}")

print("\n" + "=" * 60)
print("✅ GAP ANALYSIS COMPLETE")
print("=" * 60)
print("""
Summary:
- Revenue leakage identified (documented but not billed)
- Compliance risks flagged (billed but not documented)
- Results stored in gap_analysis_results table
- CSV report exported for review

Next steps:
1. Review high-risk claims with billing team
2. Implement cluster consensus for code suggestions
3. Build automated validation pipeline
""")
