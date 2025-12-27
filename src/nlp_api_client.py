"""
Unified API Client for Medical NLP Operations
==============================================
Uses HuggingFace Inference API + Claude instead of local models.

Memory usage: ~10MB (vs ~2GB for local spaCy/medspaCy/scispaCy)

Architecture:
- Medical NER: HuggingFace Inference API (d4data/biomedical-ner-all)
- Embeddings: HuggingFace Inference API (BAAI/bge-small-en-v1.5)
- Context Analysis: Claude API or rule-based fallback
- Text Processing: Lightweight regex (no spaCy)
"""

import os
import re
import time
import requests
from typing import List, Dict, Any, Optional, Union
import json


class MedicalNLPClient:
    """
    API-based medical NLP - replaces local spaCy/medspaCy/scispaCy.

    Benefits:
    - Memory: ~10MB vs ~2GB for local models
    - Startup: Instant vs 30-60 seconds
    - Updates: Automatic via API
    - Scalability: Runs on any platform

    Usage:
        client = get_nlp_client()
        entities = client.extract_medical_entities("Patient has diabetes")
        entities_with_context = client.analyze_clinical_context(text, entities)
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        anthropic_key: Optional[str] = None
    ):
        """
        Initialize the API client.

        Args:
            hf_token: HuggingFace API token (defaults to HF_TOKEN env var)
            anthropic_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")

        # HuggingFace Inference API configuration
        self.hf_base_url = "https://api-inference.huggingface.co"

        # Model selection
        # Using microsoft/BiomedNLP-PubMedBERT for embeddings (clinically validated)
        # Using d4data/biomedical-ner-all for NER (254K downloads, proven)
        self.embedding_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        self.ner_model = "d4data/biomedical-ner-all"

        self.headers = {}
        if self.hf_token:
            self.headers["Authorization"] = f"Bearer {self.hf_token}"

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Get embeddings via HuggingFace Inference API.

        Replaces: sentence-transformers local model

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for API calls

        Returns:
            List of 768-dimensional embedding vectors
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._get_embeddings_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        url = f"{self.hf_base_url}/models/{self.embedding_model}"

        response = requests.post(
            url,
            headers=self.headers,
            json={"inputs": texts},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            # Handle both single and batch responses
            if isinstance(result, list):
                if isinstance(result[0], list) and isinstance(result[0][0], float):
                    # Already list of embeddings
                    return result
                elif isinstance(result[0], float):
                    # Single embedding
                    return [result]
            return result
        elif response.status_code == 503:
            # Model loading - wait and retry
            print("  ⏳ Model loading (first use)... waiting 20s")
            time.sleep(20)
            return self._get_embeddings_batch(texts)
        else:
            raise Exception(f"Embedding API error {response.status_code}: {response.text}")

    # =========================================================================
    # MEDICAL NER (Named Entity Recognition)
    # =========================================================================

    def extract_medical_entities(
        self,
        text: str,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Extract medical entities via HuggingFace NER API.

        Replaces: medspaCy + scispaCy local models

        Returns entities with types:
        - DISEASE: Medical conditions, diagnoses
        - CHEMICAL: Medications, drugs
        - GENE_OR_GENE_PRODUCT: Genes, proteins
        - ORGANISM: Bacteria, viruses

        Args:
            text: Clinical text to analyze
            min_score: Minimum confidence score (0-1)

        Returns:
            List of entities with text, label, score, start, end
        """
        url = f"{self.hf_base_url}/models/{self.ner_model}"

        response = requests.post(
            url,
            headers=self.headers,
            json={"inputs": text},
            timeout=30
        )

        if response.status_code == 200:
            raw_entities = response.json()

            # Normalize to common format
            entities = []
            for e in raw_entities:
                score = e.get("score", 0.0)
                if score >= min_score:
                    entities.append({
                        "text": e.get("word", "").replace("##", ""),  # Clean subword tokens
                        "label": e.get("entity_group", e.get("entity", "UNKNOWN")),
                        "score": score,
                        "start": e.get("start", 0),
                        "end": e.get("end", 0)
                    })

            return entities
        elif response.status_code == 503:
            # Model loading - wait and retry
            print("  ⏳ NER model loading... waiting 20s")
            time.sleep(20)
            return self.extract_medical_entities(text, min_score)
        else:
            # Fallback to rule-based NER
            print(f"  ⚠️  NER API unavailable ({response.status_code}), using fallback")
            return self._fallback_entity_extraction(text)

    def _fallback_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Simple rule-based entity extraction (fallback)."""
        entities = []

        # Common medical terms patterns
        disease_patterns = [
            r'\b(diabetes|hypertension|pneumonia|COPD|asthma|cancer|stroke)\b',
            r'\b(infection|inflammation|failure|disease|syndrome)\b',
        ]

        for pattern in disease_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(0),
                    "label": "DISEASE",
                    "score": 0.6,  # Lower score for rule-based
                    "start": match.start(),
                    "end": match.end()
                })

        return entities

    # =========================================================================
    # CLINICAL CONTEXT ANALYSIS
    # =========================================================================

    def analyze_clinical_context(
        self,
        text: str,
        entities: List[Dict]
    ) -> List[Dict]:
        """
        Analyze clinical context (negation, uncertainty, etc.).

        Replaces: medspaCy ConText algorithm

        Uses Claude API if available, otherwise rule-based fallback.

        Returns entities with context attributes:
        - is_negated: bool - Condition is negated
        - is_uncertain: bool - Possible/uncertain diagnosis
        - is_historical: bool - Past medical history
        - is_family_history: bool - Family member's condition

        Args:
            text: Clinical text
            entities: Entities from extract_medical_entities()

        Returns:
            Entities with context attributes added
        """
        if self.anthropic_key and len(entities) > 0:
            try:
                return self._claude_context_analysis(text, entities)
            except Exception as e:
                print(f"  ⚠️  Claude API failed ({e}), using rule-based fallback")

        # Fallback to rule-based
        return self._rule_based_context_analysis(text, entities)

    def _claude_context_analysis(
        self,
        text: str,
        entities: List[Dict]
    ) -> List[Dict]:
        """Analyze context using Claude API."""
        try:
            from anthropic import Anthropic
        except ImportError:
            print("  ⚠️  anthropic package not installed, using rule-based fallback")
            return self._rule_based_context_analysis(text, entities)

        client = Anthropic(api_key=self.anthropic_key)

        entity_texts = [e["text"] for e in entities]

        prompt = f"""Analyze the following clinical text and determine the context for each medical entity.

Clinical Text:
{text}

Entities to analyze: {', '.join(entity_texts)}

For each entity, determine:
1. is_negated: Is this condition/finding negated? (e.g., "no fever", "denies pain", "without complications")
2. is_uncertain: Is this uncertain/possible? (e.g., "possible pneumonia", "rule out MI", "suspected")
3. is_historical: Is this from patient's history? (e.g., "history of diabetes", "past medical history of")
4. is_family_history: Is this about family member? (e.g., "mother had cancer", "family history of")

Respond ONLY with valid JSON in this exact format:
{{"entities": [{{"text": "entity1", "is_negated": true, "is_uncertain": false, "is_historical": false, "is_family_history": false}}]}}"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response and merge with original entities
        try:
            response_text = response.content[0].text
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())
            context_map = {e["text"]: e for e in result.get("entities", [])}

            for entity in entities:
                ctx = context_map.get(entity["text"], {})
                entity["is_negated"] = ctx.get("is_negated", False)
                entity["is_uncertain"] = ctx.get("is_uncertain", False)
                entity["is_historical"] = ctx.get("is_historical", False)
                entity["is_family_history"] = ctx.get("is_family_history", False)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  ⚠️  Failed to parse Claude response: {e}")
            return self._rule_based_context_analysis(text, entities)

        return entities

    def _rule_based_context_analysis(
        self,
        text: str,
        entities: List[Dict]
    ) -> List[Dict]:
        """Simple rule-based context analysis (fallback)."""
        # Negation patterns
        negation_patterns = [
            r'\b(no|not|without|denies|denied|negative|absent|ruled out)\b',
            r'\b(never|none|neither|nor)\b',
            r'\b(free of|clear of)\b',
        ]

        # Uncertainty patterns
        uncertainty_patterns = [
            r'\b(possible|possibly|probable|probably|suspected|likely)\b',
            r'\b(rule out|r/o|query|uncertain|unclear)\b',
            r'\b(may have|might have|could have)\b',
        ]

        # History patterns
        history_patterns = [
            r'\b(history of|h/o|hx|previous|prior|past)\b',
            r'\b(previously|formerly)\b',
        ]

        # Family history patterns
        family_patterns = [
            r'\b(family history|mother|father|sister|brother|parent)\b',
            r'\b(maternal|paternal|sibling)\b',
        ]

        text_lower = text.lower()

        for entity in entities:
            start = max(0, entity.get("start", 0) - 50)  # Look 50 chars before
            end = entity.get("end", 0) + 20  # Look 20 chars after
            context = text_lower[start:end]

            entity["is_negated"] = any(
                re.search(p, context) for p in negation_patterns
            )
            entity["is_uncertain"] = any(
                re.search(p, context) for p in uncertainty_patterns
            )
            entity["is_historical"] = any(
                re.search(p, context) for p in history_patterns
            )
            entity["is_family_history"] = any(
                re.search(p, context) for p in family_patterns
            )

        return entities


# =========================================================================
# SINGLETON PATTERN
# =========================================================================

_client = None


def get_nlp_client(
    hf_token: Optional[str] = None,
    anthropic_key: Optional[str] = None
) -> MedicalNLPClient:
    """
    Get or create the NLP client singleton.

    Args:
        hf_token: HuggingFace API token (optional, uses env var)
        anthropic_key: Anthropic API key (optional, uses env var)

    Returns:
        MedicalNLPClient instance
    """
    global _client
    if _client is None:
        _client = MedicalNLPClient(hf_token=hf_token, anthropic_key=anthropic_key)
    return _client


# =========================================================================
# COMPATIBILITY LAYER
# =========================================================================

def encode(texts: Union[str, List[str]]) -> List[List[float]]:
    """
    Drop-in replacement for sentence_transformers.encode()

    Usage:
        from src.nlp_api_client import encode
        embeddings = encode(["text1", "text2"])
    """
    client = get_nlp_client()
    return client.get_embeddings(texts)
