"""
NLP-based clinical entity extraction and code mapping pipeline.

This module uses medspaCy for clinical entity extraction with context detection
(negation, historical, hypothetical) and BioClinical ModernBERT embeddings
for semantic code matching via pgvector.
"""

from typing import List, Dict, Optional, Tuple
import warnings

import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import medspacy
    from medspacy.ner import TargetRule
except ImportError:
    warnings.warn("medspacy not installed. Install with: pip install medspacy")
    medspacy = None

from config.settings import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    CONFIDENCE_THRESHOLD,
    BATCH_SIZE
)
from src.features.clinical_code_object import (
    ClinicalCodeObject,
    ExtractedEntity,
    CodeSuggestion
)


class ClinicalCodeExtractor:
    """Extract clinical entities and map to billing codes using NLP."""

    def __init__(
        self,
        database_url: str = DATABASE_URL,
        model_name: str = EMBEDDING_MODEL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        """
        Initialize the extractor.

        Args:
            database_url: PostgreSQL connection string
            model_name: HuggingFace model name for embeddings
            confidence_threshold: Minimum confidence for code matching
        """
        self.database_url = database_url
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nlp = None
        self.model = None
        self.conn = None

        # Initialize pipelines
        self._load_nlp_pipeline()
        self._load_embedding_model()
        self._connect_db()

    def _load_nlp_pipeline(self) -> None:
        """Load and configure medspaCy pipeline."""
        if medspacy is None:
            raise ImportError("medspacy is required. Install with: pip install medspacy")

        print("Loading medspaCy pipeline...")

        # Load base pipeline with sentencizer
        self.nlp = medspacy.load(enable=["sentencizer"])

        # Add ConText component for negation/historical/hypothetical detection
        self.nlp.add_pipe("medspacy_context")

        # Add target matcher
        self.nlp.add_pipe("medspacy_target_matcher")

        # Add custom target rules for common clinical patterns
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")

        # Diagnosis patterns (PROBLEM entities)
        diagnosis_rules = [
            # Diabetes
            TargetRule(
                "diabetes",
                "PROBLEM",
                pattern=[{"LOWER": {"IN": ["diabetes", "diabetic", "dm", "dm2", "t2dm", "niddm"]}}]
            ),
            TargetRule(
                "hyperglycemia",
                "PROBLEM"
            ),

            # Cardiovascular
            TargetRule(
                "hypertension",
                "PROBLEM",
                pattern=[{"LOWER": {"IN": ["hypertension", "htn", "high blood pressure"]}}]
            ),
            TargetRule(
                "heart failure",
                "PROBLEM",
                pattern=[{"LOWER": "heart"}, {"LOWER": {"IN": ["failure", "fail"]}}]
            ),
            TargetRule(
                "atrial fibrillation",
                "PROBLEM",
                pattern=[{"LOWER": {"IN": ["afib", "a-fib", "af", "atrial fibrillation"]}}]
            ),
            TargetRule(
                "coronary artery disease",
                "PROBLEM",
                pattern=[{"LOWER": {"IN": ["cad", "coronary artery disease"]}}]
            ),
            TargetRule(
                "angina",
                "PROBLEM"
            ),
            TargetRule(
                "myocardial infarction",
                "PROBLEM",
                pattern=[{"LOWER": {"IN": ["mi", "myocardial infarction", "heart attack"]}}]
            ),

            # Respiratory
            TargetRule(
                "COPD",
                "PROBLEM",
                pattern=[{"LOWER": {"IN": ["copd", "chronic obstructive pulmonary disease"]}}]
            ),
            TargetRule(
                "pneumonia",
                "PROBLEM"
            ),
            TargetRule(
                "asthma",
                "PROBLEM"
            ),
            TargetRule(
                "chest pain",
                "PROBLEM",
                pattern=[{"LOWER": "chest"}, {"LOWER": "pain"}]
            ),
            TargetRule(
                "shortness of breath",
                "PROBLEM",
                pattern=[{"LOWER": {"IN": ["sob", "dyspnea", "shortness of breath"]}}]
            ),

            # Renal
            TargetRule(
                "chronic kidney disease",
                "PROBLEM",
                pattern=[{"LOWER": {"IN": ["ckd", "chronic kidney disease"]}}]
            ),
            TargetRule(
                "renal failure",
                "PROBLEM"
            ),

            # Other common diagnoses
            TargetRule(
                "anemia",
                "PROBLEM"
            ),
            TargetRule(
                "hyperlipidemia",
                "PROBLEM"
            ),
            TargetRule(
                "obesity",
                "PROBLEM"
            ),
        ]

        # Procedure patterns
        procedure_rules = [
            TargetRule(
                "colonoscopy",
                "PROCEDURE"
            ),
            TargetRule(
                "echocardiogram",
                "PROCEDURE",
                pattern=[{"LOWER": {"IN": ["echo", "echocardiogram", "tte", "echocardiography"]}}]
            ),
            TargetRule(
                "CT scan",
                "PROCEDURE",
                pattern=[{"LOWER": "ct"}, {"LOWER": {"IN": ["scan", "imaging"]}}]
            ),
            TargetRule(
                "MRI",
                "PROCEDURE",
                pattern=[{"LOWER": {"IN": ["mri", "magnetic resonance imaging"]}}]
            ),
            TargetRule(
                "x-ray",
                "PROCEDURE",
                pattern=[{"LOWER": {"IN": ["xray", "x-ray", "radiograph"]}}]
            ),
            TargetRule(
                "EKG",
                "PROCEDURE",
                pattern=[{"LOWER": {"IN": ["ekg", "ecg", "electrocardiogram"]}}]
            ),
            TargetRule(
                "ultrasound",
                "PROCEDURE"
            ),
            TargetRule(
                "blood test",
                "PROCEDURE",
                pattern=[{"LOWER": "blood"}, {"LOWER": {"IN": ["test", "work", "draw"]}}]
            ),
        ]

        # Add all rules
        target_matcher.add(diagnosis_rules + procedure_rules)

        print(f"Loaded {len(diagnosis_rules)} diagnosis rules")
        print(f"Loaded {len(procedure_rules)} procedure rules")

    def _load_embedding_model(self) -> None:
        """Load sentence-transformers model."""
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Embedding model loaded")

    def _connect_db(self) -> None:
        """Connect to pgvector database."""
        print("Connecting to database...")
        self.conn = psycopg2.connect(self.database_url)
        print("Connected to database")

    def _search_codes(
        self,
        query: str,
        code_type: str,
        top_k: int = 3
    ) -> List[Dict[str, any]]:
        """
        Semantic search for billing codes using pgvector.

        Args:
            query: Search text (e.g., "diabetes type 2")
            code_type: 'ICD-10' or 'HCPCS'
            top_k: Number of results to return

        Returns:
            List of dicts with 'code', 'description', 'similarity' keys
        """
        # Embed query
        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    code_id,
                    long_description,
                    short_description,
                    1 - (embedding <=> %s::vector) as similarity
                FROM code_embeddings
                WHERE code_type = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, code_type, query_embedding, top_k))

            results = []
            for code_id, long_desc, short_desc, similarity in cur.fetchall():
                results.append({
                    'code': code_id,
                    'description': long_desc,
                    'short_description': short_desc,
                    'similarity': float(similarity)
                })

            return results

    def extract(
        self,
        claim_id: str,
        visit_notes: str,
        condition_text: str = ""
    ) -> ClinicalCodeObject:
        """
        Extract clinical entities and map to billing codes.

        Args:
            claim_id: Unique claim identifier
            visit_notes: Clinical documentation text
            condition_text: Structured condition field (optional)

        Returns:
            ClinicalCodeObject with extracted entities and mapped codes
        """
        # Create clinical object
        clinical_obj = ClinicalCodeObject(
            claim_id=claim_id,
            visit_notes=visit_notes,
            condition_text=condition_text
        )

        # Combine all text for processing
        full_text = f"{visit_notes}\n{condition_text}"

        # Process with medspaCy
        doc = self.nlp(full_text)

        # Extract entities
        for ent in doc.ents:
            # Get context attributes
            is_negated = ent._.is_negated if hasattr(ent._, 'is_negated') else False
            is_historical = ent._.is_historical if hasattr(ent._, 'is_historical') else False
            is_hypothetical = ent._.is_hypothetical if hasattr(ent._, 'is_hypothetical') else False
            is_family = ent._.is_family if hasattr(ent._, 'is_family') else False

            # Get surrounding sentence for context
            source_span = ent.sent.text if ent.sent else ent.text

            # Determine code type based on entity label
            if ent.label_ == "PROBLEM":
                code_type = "ICD-10"
            elif ent.label_ == "PROCEDURE":
                code_type = "HCPCS"
            else:
                continue  # Skip other entity types

            # Search for matching codes
            search_results = self._search_codes(
                query=ent.text,
                code_type=code_type,
                top_k=1
            )

            if search_results:
                top_match = search_results[0]

                # Only use if confidence meets threshold
                if top_match['similarity'] >= self.confidence_threshold:
                    entity = ExtractedEntity(
                        term=ent.text,
                        code=top_match['code'],
                        code_type=code_type,
                        confidence=top_match['similarity'],
                        source_span=source_span,
                        is_negated=is_negated,
                        is_historical=is_historical,
                        is_hypothetical=is_hypothetical,
                        is_family_history=is_family
                    )

                    # Add to appropriate list
                    if code_type == "ICD-10":
                        clinical_obj.add_diagnosis(entity)
                    else:
                        clinical_obj.add_procedure(entity)

                    # Add as suggestion if billable
                    if entity.is_billable():
                        suggestion = CodeSuggestion(
                            code=entity.code,
                            code_type=code_type,
                            description=top_match['description'],
                            confidence=entity.confidence,
                            source='nlp',
                            rationale=f"Extracted from: '{entity.source_span}'"
                        )
                        clinical_obj.add_suggestion(suggestion)

        return clinical_obj

    def batch_extract(
        self,
        claims_df: pd.DataFrame,
        claim_id_col: str = 'claim_id',
        notes_col: str = 'visit_notes',
        condition_col: str = 'condition'
    ) -> List[ClinicalCodeObject]:
        """
        Extract entities from multiple claims.

        Args:
            claims_df: DataFrame with claims data
            claim_id_col: Name of claim ID column
            notes_col: Name of visit notes column
            condition_col: Name of condition column

        Returns:
            List of ClinicalCodeObject instances
        """
        print(f"\nExtracting entities from {len(claims_df)} claims...")

        clinical_objects = []

        for _, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Processing claims"):
            claim_id = str(row[claim_id_col])
            visit_notes = str(row.get(notes_col, ''))
            condition_text = str(row.get(condition_col, ''))

            clinical_obj = self.extract(claim_id, visit_notes, condition_text)
            clinical_objects.append(clinical_obj)

        # Summary statistics
        total_diagnoses = sum(obj.diagnosis_count for obj in clinical_objects)
        total_procedures = sum(obj.procedure_count for obj in clinical_objects)

        print(f"\nExtraction complete:")
        print(f"  Total diagnoses: {total_diagnoses}")
        print(f"  Total procedures: {total_procedures}")
        print(f"  Avg diagnoses per claim: {total_diagnoses / len(claims_df):.1f}")
        print(f"  Avg procedures per claim: {total_procedures / len(claims_df):.1f}")

        return clinical_objects

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Demo extraction on sample data."""
    print("\n" + "=" * 70)
    print(" " * 20 + "CLINICAL CODE EXTRACTOR DEMO")
    print("=" * 70 + "\n")

    # Sample clinical notes
    test_claims = [
        {
            "claim_id": "TEST001",
            "visit_notes": "Patient presents with chest pain. History of hypertension. "
                          "Denies shortness of breath. Assessment: Unstable angina. "
                          "Plan: ECG, troponins, cardiology consult.",
            "condition": "Chest pain; Hypertension"
        },
        {
            "claim_id": "TEST002",
            "visit_notes": "Follow-up for type 2 diabetes. A1c improved to 7.2%. "
                          "No hypoglycemic episodes. Continue metformin.",
            "condition": "Type 2 diabetes mellitus"
        },
        {
            "claim_id": "TEST003",
            "visit_notes": "Patient with COPD exacerbation. Increased dyspnea and productive cough. "
                          "CXR shows hyperinflation. Started on prednisone burst and albuterol nebs.",
            "condition": "COPD exacerbation"
        }
    ]

    df = pd.DataFrame(test_claims)

    # Extract
    extractor = ClinicalCodeExtractor()

    try:
        clinical_objects = extractor.batch_extract(df)

        # Display results
        print("\n" + "=" * 70)
        print("EXTRACTION RESULTS")
        print("=" * 70)

        for obj in clinical_objects:
            print(f"\nClaim: {obj.claim_id}")
            print(f"  Diagnoses ({obj.diagnosis_count}):")
            for diag in obj.billable_diagnoses:
                print(f"    - {diag.term} → {diag.code} (confidence: {diag.confidence:.2f})")

            print(f"  Procedures ({obj.procedure_count}):")
            for proc in obj.billable_procedures:
                print(f"    - {proc.term} → {proc.code} (confidence: {proc.confidence:.2f})")

    finally:
        extractor.close()


if __name__ == "__main__":
    main()
