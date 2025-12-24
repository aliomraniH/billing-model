"""
Build vector embeddings for medical billing codes and populate pgvector database.

This module creates the knowledge base by:
1. Initializing pgvector database schema
2. Loading BioClinical ModernBERT model
3. Generating embeddings for all code descriptions
4. Inserting embeddings into database with HNSW index
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    RAW_DATA_DIR,
    BATCH_SIZE
)


class EmbeddingBuilder:
    """Build and manage medical code embeddings in pgvector."""

    def __init__(
        self,
        database_url: str = DATABASE_URL,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = BATCH_SIZE
    ):
        """
        Initialize the embedding builder.

        Args:
            database_url: PostgreSQL connection string
            model_name: HuggingFace model name
            batch_size: Batch size for embedding generation
        """
        self.database_url = database_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.conn = None

    def load_model(self) -> None:
        """Load the sentence-transformers model."""
        print(f"Loading model: {self.model_name}")
        print("This may take a few minutes on first run...")

        self.model = SentenceTransformer(self.model_name)

        print(f"Model loaded successfully")
        print(f"  Max sequence length: {self.model.max_seq_length}")
        print(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def connect_db(self) -> None:
        """Connect to PostgreSQL database."""
        print(f"Connecting to database...")
        self.conn = psycopg2.connect(self.database_url)
        print("Connected successfully")

    def init_schema(self) -> None:
        """Initialize database schema with pgvector extension and tables."""
        print("\n" + "=" * 60)
        print("Initializing Database Schema")
        print("=" * 60)

        with self.conn.cursor() as cur:
            # Enable pgvector extension
            print("Enabling pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Drop existing table if needed (for clean rebuild)
            print("Creating code_embeddings table...")
            cur.execute("DROP TABLE IF EXISTS code_embeddings;")

            # Create main table
            cur.execute(f"""
                CREATE TABLE code_embeddings (
                    code_id VARCHAR(20) PRIMARY KEY,
                    code_type VARCHAR(10) NOT NULL,
                    short_description TEXT,
                    long_description TEXT NOT NULL,
                    category VARCHAR(50),
                    is_billable BOOLEAN DEFAULT TRUE,
                    embedding vector({EMBEDDING_DIM})
                );
            """)

            self.conn.commit()
            print("Table created successfully")

    def create_index(self) -> None:
        """Create HNSW index for fast similarity search."""
        print("\n" + "=" * 60)
        print("Creating HNSW Index")
        print("=" * 60)
        print("This may take several minutes for large datasets...")

        with self.conn.cursor() as cur:
            # Drop existing index if present
            cur.execute("DROP INDEX IF EXISTS code_embeddings_hnsw_idx;")

            # Create HNSW index
            # m=16: number of connections per layer
            # ef_construction=64: size of dynamic candidate list during construction
            cur.execute("""
                CREATE INDEX code_embeddings_hnsw_idx
                ON code_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)

            self.conn.commit()
            print("HNSW index created successfully")

    def embed_codes(
        self,
        df: pd.DataFrame,
        code_type: str,
        description_column: str = 'long_description'
    ) -> List[Tuple]:
        """
        Generate embeddings for code descriptions.

        Args:
            df: DataFrame with code data
            code_type: 'ICD-10' or 'HCPCS'
            description_column: Column to use for embedding

        Returns:
            List of tuples ready for database insertion
        """
        print(f"\n" + "=" * 60)
        print(f"Embedding {code_type} Codes")
        print("=" * 60)
        print(f"Total codes: {len(df)}")

        # Prepare descriptions
        descriptions = df[description_column].fillna('').tolist()
        codes = df['code'].tolist()

        # Generate embeddings in batches
        all_embeddings = []

        for i in tqdm(range(0, len(descriptions), self.batch_size),
                     desc=f"Embedding {code_type}"):
            batch_desc = descriptions[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_desc,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.extend(batch_embeddings)

        # Prepare data for insertion
        records = []
        for idx, (code, embedding) in enumerate(zip(codes, all_embeddings)):
            row = df.iloc[idx]

            record = (
                code,                                          # code_id
                code_type,                                     # code_type
                row.get('short_description', ''),             # short_description
                row.get('long_description', ''),              # long_description
                row.get('category', ''),                      # category
                row.get('is_billable', True),                 # is_billable
                embedding.tolist()                            # embedding
            )
            records.append(record)

        print(f"Generated {len(records)} embeddings")
        return records

    def insert_embeddings(self, records: List[Tuple]) -> None:
        """
        Insert embedding records into database.

        Args:
            records: List of tuples with code data and embeddings
        """
        print(f"Inserting {len(records)} records into database...")

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO code_embeddings
                (code_id, code_type, short_description, long_description,
                 category, is_billable, embedding)
                VALUES %s
                ON CONFLICT (code_id) DO UPDATE SET
                    code_type = EXCLUDED.code_type,
                    short_description = EXCLUDED.short_description,
                    long_description = EXCLUDED.long_description,
                    category = EXCLUDED.category,
                    is_billable = EXCLUDED.is_billable,
                    embedding = EXCLUDED.embedding;
                """,
                records,
                page_size=1000
            )

        self.conn.commit()
        print(f"Inserted successfully")

    def verify_embeddings(self) -> None:
        """Verify embeddings were inserted correctly."""
        print("\n" + "=" * 60)
        print("Verifying Embeddings")
        print("=" * 60)

        with self.conn.cursor() as cur:
            # Count total records
            cur.execute("SELECT COUNT(*) FROM code_embeddings;")
            total_count = cur.fetchone()[0]
            print(f"Total records: {total_count}")

            # Count by type
            cur.execute("""
                SELECT code_type, COUNT(*) as count
                FROM code_embeddings
                GROUP BY code_type;
            """)
            for code_type, count in cur.fetchall():
                print(f"  {code_type}: {count}")

            # Sample records
            cur.execute("""
                SELECT code_id, code_type, short_description
                FROM code_embeddings
                LIMIT 5;
            """)
            print("\nSample records:")
            for code_id, code_type, desc in cur.fetchall():
                print(f"  {code_id} ({code_type}): {desc}")

    def test_search(self, query: str = "diabetes type 2", top_k: int = 5) -> None:
        """
        Test semantic search functionality.

        Args:
            query: Search query
            top_k: Number of results to return
        """
        print("\n" + "=" * 60)
        print(f"Testing Search: '{query}'")
        print("=" * 60)

        # Embed query
        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()

        with self.conn.cursor() as cur:
            # Cosine similarity search
            cur.execute("""
                SELECT
                    code_id,
                    code_type,
                    long_description,
                    1 - (embedding <=> %s::vector) as similarity
                FROM code_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, query_embedding, top_k))

            print(f"Top {top_k} results:")
            for code_id, code_type, desc, similarity in cur.fetchall():
                print(f"  {similarity:.3f} | {code_id} ({code_type}): {desc}")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("\nDatabase connection closed")


def build_knowledge_base(
    icd10_path: Optional[Path] = None,
    hcpcs_path: Optional[Path] = None,
    rebuild: bool = False
) -> None:
    """
    Main function to build the complete knowledge base.

    Args:
        icd10_path: Path to ICD-10 CSV file
        hcpcs_path: Path to HCPCS CSV file
        rebuild: If True, rebuild from scratch
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "MEDICAL CODE EMBEDDING BUILDER")
    print("=" * 70 + "\n")

    # Use default paths if not provided
    if icd10_path is None:
        icd10_path = RAW_DATA_DIR / "icd10cm_codes_2025.csv"
    if hcpcs_path is None:
        hcpcs_path = RAW_DATA_DIR / "hcpcs_codes_2025.csv"

    # Check if files exist
    if not icd10_path.exists():
        print(f"ERROR: ICD-10 file not found: {icd10_path}")
        print("Please run src/data/download_codes.py first")
        sys.exit(1)

    if not hcpcs_path.exists():
        print(f"ERROR: HCPCS file not found: {hcpcs_path}")
        print("Please run src/data/download_codes.py first")
        sys.exit(1)

    # Initialize builder
    builder = EmbeddingBuilder()

    try:
        # Load model
        builder.load_model()

        # Connect to database
        builder.connect_db()

        # Initialize schema
        if rebuild:
            builder.init_schema()

        # Load code data
        print("\nLoading code data...")
        icd10_df = pd.read_csv(icd10_path)
        hcpcs_df = pd.read_csv(hcpcs_path)
        print(f"  ICD-10: {len(icd10_df)} codes")
        print(f"  HCPCS: {len(hcpcs_df)} codes")

        # Embed and insert ICD-10 codes
        icd10_records = builder.embed_codes(icd10_df, 'ICD-10')
        builder.insert_embeddings(icd10_records)

        # Embed and insert HCPCS codes
        hcpcs_records = builder.embed_codes(hcpcs_df, 'HCPCS')
        builder.insert_embeddings(hcpcs_records)

        # Create index
        builder.create_index()

        # Verify
        builder.verify_embeddings()

        # Test search
        builder.test_search("diabetes type 2")
        builder.test_search("chest pain")

        print("\n" + "=" * 70)
        print(" " * 25 + "BUILD COMPLETE!")
        print("=" * 70)
        print("\nThe knowledge base is ready for use.")
        print("You can now run the NLP extraction pipeline.")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise

    finally:
        builder.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build medical code embeddings knowledge base"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild database from scratch"
    )
    parser.add_argument(
        "--icd10",
        type=Path,
        help="Path to ICD-10 CSV file"
    )
    parser.add_argument(
        "--hcpcs",
        type=Path,
        help="Path to HCPCS CSV file"
    )

    args = parser.parse_args()

    build_knowledge_base(
        icd10_path=args.icd10,
        hcpcs_path=args.hcpcs,
        rebuild=args.rebuild
    )
