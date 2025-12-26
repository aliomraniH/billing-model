"""
Clinical NLP Extraction Engine using medspaCy + BioClinical ModernBERT.
Extracts ICD-10 and HCPCS codes from clinical text with negation detection.
"""
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

# NLP imports (lazy loaded for Deepnote compatibility)
_nlp = None
_embedding_model = None

def _get_nlp():
    """Lazy load medspaCy pipeline."""
    global _nlp
    if _nlp is None:
        import medspacy
        _nlp = medspacy.load()
        # Add ConText for negation detection
        from medspacy.context import ConTextComponent
        if "medspacy_context" not in _nlp.pipe_names:
            _nlp.add_pipe("medspacy_context")
    return _nlp

def _get_embedding_model():
    """Lazy load embedding model via HuggingFace."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        from config.settings import embedding_config
        _embedding_model = SentenceTransformer(embedding_config.model_name)
    return _embedding_model


class ClinicalCodeExtractor:
    """
    Main extraction engine for clinical text to billing codes.

    Usage:
        extractor = ClinicalCodeExtractor(db_connection)
        clinical_objects = extractor.batch_extract(claims_df)
    """

    def __init__(self, db_connection=None):
        """
        Initialize extractor with database connection for code lookups.

        Args:
            db_connection: SQLAlchemy engine or connection for pgvector queries
        """
        self.db = db_connection
        self.nlp = None
        self.embedding_model = None
        self._code_embeddings_loaded = False

    def _ensure_models_loaded(self):
        """Lazy load NLP models on first use."""
        if self.nlp is None:
            print("📦 Loading medspaCy pipeline...")
            self.nlp = _get_nlp()
            print("✅ medspaCy loaded")

        if self.embedding_model is None:
            print("📦 Loading BioClinical ModernBERT...")
            self.embedding_model = _get_embedding_model()
            print("✅ Embedding model loaded")

    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract clinical entities from text with context analysis.

        Args:
            text: Clinical note text

        Returns:
            Dict with 'conditions' and 'procedures' lists
        """
        self._ensure_models_loaded()

        doc = self.nlp(text)

        entities = {
            'conditions': [],
            'procedures': []
        }

        for ent in doc.ents:
            entity_data = {
                'term': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'is_negated': getattr(ent._, 'is_negated', False),
                'is_historical': getattr(ent._, 'is_historical', False),
                'is_family': getattr(ent._, 'is_family', False),
                'is_hypothetical': getattr(ent._, 'is_hypothetical', False)
            }

            # Categorize by entity type
            if ent.label_ in ['PROBLEM', 'DIAGNOSIS', 'DISEASE']:
                entities['conditions'].append(entity_data)
            elif ent.label_ in ['PROCEDURE', 'TREATMENT', 'TEST']:
                entities['procedures'].append(entity_data)

        return entities

    def match_code_by_embedding(
        self,
        term: str,
        code_type: str = 'ICD-10',
        top_k: int = 5,
        threshold: float = 0.85
    ) -> List[Tuple[str, str, float]]:
        """
        Find matching billing code using semantic similarity.

        Args:
            term: Clinical term to match
            code_type: 'ICD-10' or 'HCPCS'
            top_k: Number of candidates to return
            threshold: Minimum confidence threshold

        Returns:
            List of (code, description, confidence) tuples
        """
        self._ensure_models_loaded()

        if self.db is None:
            print("⚠️  No database connection - cannot perform code matching")
            return []

        # Generate embedding for the term
        query_embedding = self.embedding_model.encode(term)

        # Query pgvector for similar codes
        from sqlalchemy import text

        query = text("""
            SELECT
                code_id,
                short_description,
                1 - (embedding <=> CAST(:embedding AS vector)) as similarity
            FROM code_embeddings
            WHERE code_type = :code_type
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        with self.db.connect() as conn:
            result = conn.execute(query, {
                'embedding': query_embedding.tolist(),
                'code_type': code_type,
                'top_k': top_k
            })

            matches = []
            for row in result:
                if row.similarity >= threshold:
                    matches.append((row.code_id, row.short_description, row.similarity))

            return matches

    def process_claim(self, claim_id: str, clinical_text: str) -> 'ClinicalCodeObject':
        """
        Process a single claim's clinical text and create ClinicalCodeObject.

        Args:
            claim_id: Unique claim identifier
            clinical_text: Combined clinical notes text

        Returns:
            ClinicalCodeObject with extracted codes and metadata
        """
        from src.features.clinical_code_object import ClinicalCodeObject, ExtractedEntity
        from config.settings import nlp_config

        clinical_object = ClinicalCodeObject(claim_id=claim_id)

        # Extract entities
        entities = self.extract_entities(clinical_text)

        # Process conditions -> ICD-10 codes
        for condition in entities['conditions']:
            if condition['is_negated']:
                # Skip negated conditions but log them
                continue

            matches = self.match_code_by_embedding(
                condition['term'],
                code_type='ICD-10',
                threshold=nlp_config.confidence_threshold
            )

            if matches:
                best_match = matches[0]
                clinical_object.extracted_diagnoses.append(ExtractedEntity(
                    term=condition['term'],
                    code=best_match[0],
                    code_type='ICD-10',
                    confidence=best_match[2],
                    source_span=clinical_text[max(0, condition['start']-50):min(len(clinical_text), condition['end']+50)],
                    is_negated=condition['is_negated'],
                    is_historical=condition['is_historical'],
                    is_family_history=condition['is_family']
                ))

        # Process procedures -> HCPCS codes
        for procedure in entities['procedures']:
            matches = self.match_code_by_embedding(
                procedure['term'],
                code_type='HCPCS',
                threshold=nlp_config.confidence_threshold
            )

            if matches:
                best_match = matches[0]
                clinical_object.extracted_procedures.append(ExtractedEntity(
                    term=procedure['term'],
                    code=best_match[0],
                    code_type='HCPCS',
                    confidence=best_match[2],
                    source_span=clinical_text[max(0, procedure['start']-50):min(len(clinical_text), procedure['end']+50)]
                ))

        return clinical_object

    def batch_extract(
        self,
        df: pd.DataFrame,
        text_column: str = 'visit_notes',
        claim_id_column: str = 'claim_id',
        condition_column: str = 'condition'
    ) -> List['ClinicalCodeObject']:
        """
        Process multiple claims in batch.

        Args:
            df: DataFrame with claims data
            text_column: Column containing clinical notes
            claim_id_column: Column containing claim IDs
            condition_column: Optional column with structured condition text

        Returns:
            List of ClinicalCodeObject instances
        """
        from config.settings import processing_config

        results = []
        total = min(len(df), processing_config.max_claims_per_run)

        print(f"🔄 Processing {total} claims...")

        for idx, row in df.head(total).iterrows():
            # Combine text sources
            clinical_text = str(row.get(text_column, ''))
            if condition_column and condition_column in row:
                clinical_text = f"{row[condition_column]}. {clinical_text}"

            try:
                clinical_object = self.process_claim(
                    claim_id=str(row[claim_id_column]),
                    clinical_text=clinical_text
                )
                results.append(clinical_object)

                if (idx + 1) % 100 == 0:
                    print(f"  📊 Processed {idx + 1}/{total} claims")

            except Exception as e:
                print(f"  ⚠️  Error processing claim {row[claim_id_column]}: {e}")
                continue

        print(f"✅ Completed: {len(results)} claims processed successfully")
        return results
