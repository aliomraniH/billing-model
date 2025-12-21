"""
Quick script to check clinical_notes table schema and data state
"""
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')

if not DATABASE_URL:
    print("❌ Error: VERCEL_POSTGRES_URL not set")
    exit(1)

engine = create_engine(DATABASE_URL)

print("=" * 70)
print("📋 CLINICAL_NOTES TABLE INSPECTION")
print("=" * 70)

with engine.connect() as conn:
    # Check columns
    print("\n1. Table Schema:")
    result = conn.execute(text("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'clinical_notes'
        ORDER BY ordinal_position
    """))

    for row in result:
        print(f"   {row[0]:25} {row[1]:20} {'NULL' if row[2] == 'YES' else 'NOT NULL'}")

    # Check data statistics
    print("\n2. Data Statistics:")
    result = conn.execute(text("""
        SELECT
            COUNT(*) as total_notes,
            COUNT(last_embedded_at) as with_timestamp,
            COUNT(embedding_model) as with_model,
            COUNT(CASE WHEN last_embedded_at > NOW() - INTERVAL '12 hours' THEN 1 END) as fresh_embeddings,
            COUNT(CASE WHEN last_embedded_at <= NOW() - INTERVAL '12 hours' THEN 1 END) as stale_embeddings,
            COUNT(CASE WHEN last_embedded_at IS NULL THEN 1 END) as never_embedded
        FROM clinical_notes
    """))

    row = result.fetchone()
    print(f"   Total notes: {row[0]:,}")
    print(f"   With timestamp: {row[1]:,}")
    print(f"   With embedding_model: {row[2]:,}")
    print(f"   Fresh embeddings (<12h old): {row[3]:,}")
    print(f"   Stale embeddings (>12h old): {row[4]:,}")
    print(f"   Never embedded: {row[5]:,}")

    # Sample some records
    print("\n3. Sample Records:")
    result = conn.execute(text("""
        SELECT
            claim_id,
            note_type,
            last_embedded_at,
            embedding_model,
            EXTRACT(EPOCH FROM (NOW() - last_embedded_at))/3600 as hours_old
        FROM clinical_notes
        LIMIT 5
    """))

    for row in result:
        hours_old = f"{row[4]:.1f}h" if row[4] else "Never"
        print(f"   Claim {row[0]}: {row[1]:10} | {row[2]} | {row[3]} | Age: {hours_old}")

print("\n" + "=" * 70)
