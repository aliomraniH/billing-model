"""
Build and populate the code_embeddings table in Vercel Postgres.
Downloads ICD-10 and HCPCS codes from CMS and generates embeddings.
"""
import os
import requests
import zipfile
import io
import pandas as pd
from typing import Optional
from sqlalchemy import create_engine, text


def download_icd10_codes(year: int = 2025) -> pd.DataFrame:
    """
    Download ICD-10-CM codes from CMS.

    Args:
        year: Code year (default: 2025)

    Returns:
        DataFrame with code_id, short_description, long_description
    """
    # CMS publishes codes at predictable URLs
    # For 2025: https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip

    print(f"📥 Downloading ICD-10-CM {year} codes from CMS...")

    # Try multiple possible URLs
    urls = [
        f"https://www.cms.gov/files/zip/{year}-code-descriptions-tabular-order.zip",
        f"https://www.cms.gov/Medicare/Coding/ICD10/Downloads/{year}-ICD-10-CM-Code-Descriptions.zip"
    ]

    for url in urls:
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                print(f"✅ Downloaded from {url}")
                break
        except:
            continue
    else:
        print("⚠️  Could not download ICD-10 codes. Using sample data.")
        return _get_sample_icd10_codes()

    # Extract and parse
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Find the descriptions file
        for filename in z.namelist():
            if 'order' in filename.lower() and filename.endswith('.txt'):
                with z.open(filename) as f:
                    # ICD-10 order file format: code (5-7 chars), is_header, short_desc, long_desc
                    lines = f.read().decode('utf-8', errors='ignore').splitlines()

                    codes = []
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 4:
                            code = parts[0]
                            is_header = parts[1] == '0'
                            short_desc = ' '.join(parts[2:])[:60]
                            long_desc = ' '.join(parts[2:])

                            if not is_header:  # Only billable codes
                                codes.append({
                                    'code_id': code,
                                    'short_description': short_desc,
                                    'long_description': long_desc,
                                    'code_type': 'ICD-10',
                                    'is_billable': True
                                })

                    return pd.DataFrame(codes)

    return _get_sample_icd10_codes()


def download_hcpcs_codes(year: int = 2025) -> pd.DataFrame:
    """
    Download HCPCS Level II codes from CMS (public domain, no licensing required).

    Args:
        year: Code year

    Returns:
        DataFrame with HCPCS codes
    """
    print(f"📥 Downloading HCPCS Level II {year} codes from CMS...")

    # CMS quarterly updates
    url = f"https://www.cms.gov/files/zip/{year}-alpha-numeric-hcpcs-file.zip"

    try:
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            raise Exception("Download failed")

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for filename in z.namelist():
                if filename.endswith('.csv') or filename.endswith('.txt'):
                    with z.open(filename) as f:
                        # Parse HCPCS format
                        df = pd.read_csv(f, sep='\t' if filename.endswith('.txt') else ',',
                                        encoding='utf-8', on_bad_lines='skip')

                        # Standardize column names
                        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

                        # Build output DataFrame
                        codes = []
                        for _, row in df.iterrows():
                            code = row.get('hcpc', row.get('hcpcs_code', row.get('code', '')))
                            desc = row.get('long_description', row.get('description', ''))
                            short = row.get('short_description', desc[:60] if desc else '')

                            if code and desc:
                                codes.append({
                                    'code_id': str(code),
                                    'short_description': short,
                                    'long_description': desc,
                                    'code_type': 'HCPCS',
                                    'is_billable': True
                                })

                        return pd.DataFrame(codes)

    except Exception as e:
        print(f"⚠️  HCPCS download failed: {e}. Using sample data.")
        return _get_sample_hcpcs_codes()


def _get_sample_icd10_codes() -> pd.DataFrame:
    """Get sample ICD-10 codes for testing."""
    return pd.DataFrame([
        {'code_id': 'E11.9', 'short_description': 'Type 2 diabetes mellitus', 'long_description': 'Type 2 diabetes mellitus without complications', 'code_type': 'ICD-10', 'is_billable': True},
        {'code_id': 'I10', 'short_description': 'Essential hypertension', 'long_description': 'Essential (primary) hypertension', 'code_type': 'ICD-10', 'is_billable': True},
        {'code_id': 'J06.9', 'short_description': 'Acute URI', 'long_description': 'Acute upper respiratory infection, unspecified', 'code_type': 'ICD-10', 'is_billable': True},
        {'code_id': 'R07.9', 'short_description': 'Chest pain', 'long_description': 'Chest pain, unspecified', 'code_type': 'ICD-10', 'is_billable': True},
        {'code_id': 'I48.91', 'short_description': 'Atrial fibrillation', 'long_description': 'Unspecified atrial fibrillation', 'code_type': 'ICD-10', 'is_billable': True},
    ])


def _get_sample_hcpcs_codes() -> pd.DataFrame:
    """Get sample HCPCS codes for testing."""
    return pd.DataFrame([
        {'code_id': 'G0008', 'short_description': 'Admin influenza virus vaccine', 'long_description': 'Administration of influenza virus vaccine', 'code_type': 'HCPCS', 'is_billable': True},
        {'code_id': 'J0129', 'short_description': 'Abatacept injection', 'long_description': 'Injection, abatacept, 10 mg', 'code_type': 'HCPCS', 'is_billable': True},
        {'code_id': 'A4253', 'short_description': 'Blood glucose strips', 'long_description': 'Blood glucose test or reagent strips for home blood glucose monitor', 'code_type': 'HCPCS', 'is_billable': True},
    ])


def build_knowledge_base(database_url: str = None, batch_size: int = 100):
    """
    Build the complete code embeddings knowledge base.

    Args:
        database_url: Vercel Postgres URL (reads from env if not provided)
        batch_size: Embedding batch size
    """
    from config.settings import db_config, embedding_config
    from src.data.hf_embeddings import get_embeddings_model

    url = database_url or db_config.url
    engine = create_engine(url)

    print("🔧 Building Medical Billing Knowledge Base")
    print("=" * 60)

    # 1. Create table if not exists
    print("\n📋 Step 1: Creating code_embeddings table...")
    with engine.begin() as conn:  # Use begin() for auto-commit
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS code_embeddings (
                code_id VARCHAR(20) PRIMARY KEY,
                code_type VARCHAR(10) NOT NULL,
                short_description TEXT,
                long_description TEXT,
                category VARCHAR(50),
                is_billable BOOLEAN DEFAULT TRUE,
                embedding vector(768)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_code_embeddings_hnsw
            ON code_embeddings USING hnsw (embedding vector_cosine_ops)
        """))
    print("✅ Table created")

    # 2. Download codes
    print("\n📋 Step 2: Downloading code sets...")
    icd10_df = download_icd10_codes()
    hcpcs_df = download_hcpcs_codes()

    all_codes = pd.concat([icd10_df, hcpcs_df], ignore_index=True)
    print(f"✅ Downloaded {len(icd10_df)} ICD-10 + {len(hcpcs_df)} HCPCS = {len(all_codes)} total codes")

    # 3. Generate embeddings via HuggingFace Inference API
    print(f"\n📋 Step 3: Generating embeddings with {embedding_config.model_name}...")
    print("  Using HuggingFace Inference API (no local download needed)")
    model = get_embeddings_model(embedding_config.model_name)

    # Embed descriptions in batches
    descriptions = all_codes['long_description'].tolist()
    embeddings = []

    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i+batch_size, len(descriptions))}/{len(descriptions)} codes")

    all_codes['embedding'] = embeddings
    print(f"✅ Generated {len(embeddings)} embeddings via API")

    # 4. Insert into database
    print("\n📋 Step 4: Inserting into Vercel Postgres...")
    with engine.begin() as conn:  # Use begin() for auto-commit
        # Clear existing
        conn.execute(text("DELETE FROM code_embeddings"))

        # Insert in batches
        for i in range(0, len(all_codes), batch_size):
            batch = all_codes.iloc[i:i+batch_size]

            for _, row in batch.iterrows():
                conn.execute(text("""
                    INSERT INTO code_embeddings (code_id, code_type, short_description, long_description, is_billable, embedding)
                    VALUES (:code_id, :code_type, :short_desc, :long_desc, :is_billable, :embedding)
                    ON CONFLICT (code_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        long_description = EXCLUDED.long_description
                """), {
                    'code_id': row['code_id'],
                    'code_type': row['code_type'],
                    'short_desc': row['short_description'],
                    'long_desc': row['long_description'],
                    'is_billable': row['is_billable'],
                    'embedding': row['embedding'].tolist()
                })

            print(f"  Inserted {min(i+batch_size, len(all_codes))}/{len(all_codes)} codes")

    print("\n" + "=" * 60)
    print("✅ Knowledge base build complete!")
    print(f"   Total codes: {len(all_codes)}")
    print(f"   ICD-10: {len(icd10_df)}")
    print(f"   HCPCS: {len(hcpcs_df)}")


if __name__ == "__main__":
    build_knowledge_base()
