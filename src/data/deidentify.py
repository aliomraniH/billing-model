"""
HIPAA-compliant de-identification for clinical text.

This module uses Microsoft Presidio to detect and anonymize Protected Health
Information (PHI) in clinical notes, including:
- Patient names
- Dates
- Locations
- Phone numbers
- Medical record numbers
- Provider identifiers
"""

import re
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError:
    print("WARNING: Presidio not installed. Install with:")
    print("  pip install presidio-analyzer presidio-anonymizer")
    AnalyzerEngine = None
    AnonymizerEngine = None


class ClinicalDeidentifier:
    """
    De-identify clinical text using Presidio.

    This class handles HIPAA-compliant de-identification by detecting and
    replacing 18 types of PHI identifiers with placeholder tokens.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize de-identification engine.

        Args:
            language: Language code (default: "en")
        """
        if AnalyzerEngine is None:
            raise ImportError("Presidio is required. Install with: "
                            "pip install presidio-analyzer presidio-anonymizer")

        self.language = language
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # Add custom clinical recognizers
        self._add_custom_recognizers()

    def _add_custom_recognizers(self) -> None:
        """Add custom recognizers for medical-specific PHI."""

        # Medical Record Number (MRN) pattern
        # Typical formats: MRN123456, MR-123456, 12345678
        mrn_pattern = Pattern(
            name="mrn_pattern",
            regex=r"\b(?:MRN|MR|Medical Record)[\s#:-]*(\d{6,10})\b",
            score=0.85
        )

        mrn_recognizer = PatternRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            patterns=[mrn_pattern],
            context=["patient", "record", "chart"]
        )

        # Patient ID pattern
        patient_id_pattern = Pattern(
            name="patient_id_pattern",
            regex=r"\b(?:Patient ID|PID|Pat)[\s#:-]*([A-Z0-9]{6,12})\b",
            score=0.80
        )

        patient_id_recognizer = PatternRecognizer(
            supported_entity="PATIENT_ID",
            patterns=[patient_id_pattern],
            context=["patient", "identification"]
        )

        # National Provider Identifier (NPI)
        # 10-digit number
        npi_pattern = Pattern(
            name="npi_pattern",
            regex=r"\b(?:NPI|Provider ID)[\s#:-]*(\d{10})\b",
            score=0.90
        )

        npi_recognizer = PatternRecognizer(
            supported_entity="NPI",
            patterns=[npi_pattern],
            context=["provider", "physician", "doctor"]
        )

        # Add all custom recognizers
        self.analyzer.registry.add_recognizer(mrn_recognizer)
        self.analyzer.registry.add_recognizer(patient_id_recognizer)
        self.analyzer.registry.add_recognizer(npi_recognizer)

    def deidentify_text(
        self,
        text: str,
        entities_to_detect: Optional[List[str]] = None
    ) -> str:
        """
        De-identify clinical text by replacing PHI with tokens.

        Args:
            text: Clinical text to de-identify
            entities_to_detect: List of entity types to detect (None = all)

        Returns:
            De-identified text with PHI replaced by tokens
        """
        if not text or not text.strip():
            return text

        # Detect PHI
        if entities_to_detect is None:
            # Default: detect common HIPAA identifiers
            entities_to_detect = [
                "PERSON",
                "DATE_TIME",
                "LOCATION",
                "PHONE_NUMBER",
                "EMAIL_ADDRESS",
                "US_SSN",
                "MEDICAL_RECORD_NUMBER",
                "PATIENT_ID",
                "NPI"
            ]

        analyzer_results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=entities_to_detect
        )

        # Define replacement operators
        # Use consistent tokens for each entity type
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "[PATIENT]"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATE]"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
            "MEDICAL_RECORD_NUMBER": OperatorConfig("replace", {"new_value": "[MRN]"}),
            "PATIENT_ID": OperatorConfig("replace", {"new_value": "[PATIENT_ID]"}),
            "NPI": OperatorConfig("replace", {"new_value": "[NPI]"}),
        }

        # Anonymize
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators
        )

        return anonymized_result.text

    def batch_deidentify(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """
        De-identify multiple texts.

        Args:
            texts: List of clinical texts
            show_progress: Show progress bar

        Returns:
            List of de-identified texts
        """
        deidentified = []

        iterator = tqdm(texts, desc="De-identifying") if show_progress else texts

        for text in iterator:
            deidentified.append(self.deidentify_text(text))

        return deidentified

    def deidentify_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: List[str]
    ) -> pd.DataFrame:
        """
        De-identify specific columns in a DataFrame.

        Args:
            df: Input DataFrame
            text_columns: List of column names to de-identify

        Returns:
            DataFrame with de-identified columns
        """
        df_copy = df.copy()

        for col in text_columns:
            if col in df_copy.columns:
                print(f"De-identifying column: {col}")
                df_copy[col] = self.batch_deidentify(
                    df_copy[col].fillna('').astype(str).tolist()
                )

        return df_copy

    def get_phi_entities(
        self,
        text: str,
        entities_to_detect: Optional[List[str]] = None
    ) -> List[Dict[str, any]]:
        """
        Get detected PHI entities without anonymizing.

        Useful for auditing and quality checks.

        Args:
            text: Clinical text
            entities_to_detect: List of entity types to detect

        Returns:
            List of detected entities with metadata
        """
        if entities_to_detect is None:
            entities_to_detect = [
                "PERSON", "DATE_TIME", "LOCATION", "PHONE_NUMBER",
                "EMAIL_ADDRESS", "US_SSN", "MEDICAL_RECORD_NUMBER",
                "PATIENT_ID", "NPI"
            ]

        analyzer_results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=entities_to_detect
        )

        entities = []
        for result in analyzer_results:
            entities.append({
                'entity_type': result.entity_type,
                'text': text[result.start:result.end],
                'start': result.start,
                'end': result.end,
                'score': result.score
            })

        return entities


def simple_deidentify(text: str) -> str:
    """
    Simple rule-based de-identification without Presidio.

    This is a fallback for when Presidio is not installed.
    Less accurate but doesn't require dependencies.

    Args:
        text: Clinical text

    Returns:
        De-identified text
    """
    if not text:
        return text

    # Replace dates (MM/DD/YYYY, DD-MM-YYYY, etc.)
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '[DATE]', text)

    # Replace phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Replace SSN patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

    # Replace MRN patterns
    text = re.sub(r'\b(?:MRN|MR)[\s#:-]*\d{6,10}\b', '[MRN]', text, flags=re.IGNORECASE)

    # Replace numeric IDs that look like patient IDs
    text = re.sub(r'\b(?:Patient ID|PID)[\s#:-]*[A-Z0-9]{6,12}\b', '[PATIENT_ID]', text, flags=re.IGNORECASE)

    return text


def main():
    """Demo de-identification."""
    print("\n" + "=" * 70)
    print(" " * 25 + "DE-IDENTIFICATION DEMO")
    print("=" * 70 + "\n")

    # Sample clinical text with PHI
    sample_text = """
    Patient Name: John Smith
    MRN: 12345678
    DOB: 01/15/1965
    Visit Date: 03/20/2024

    Chief Complaint: Follow-up for type 2 diabetes

    HPI: Mr. Smith is a 59-year-old male with a history of type 2 diabetes,
    hypertension, and hyperlipidemia. He was last seen on 02/15/2024.
    Current A1c is 7.2%, improved from 8.1%.

    Phone: 555-123-4567
    Email: john.smith@example.com

    Provider: Dr. Jane Doe
    NPI: 1234567890
    """

    print("Original Text:")
    print("=" * 60)
    print(sample_text)

    # Try Presidio de-identification
    try:
        deidentifier = ClinicalDeidentifier()

        deidentified = deidentifier.deidentify_text(sample_text)

        print("\n\nDe-identified Text (Presidio):")
        print("=" * 60)
        print(deidentified)

        # Show detected entities
        entities = deidentifier.get_phi_entities(sample_text)

        print("\n\nDetected PHI Entities:")
        print("=" * 60)
        for entity in entities:
            print(f"  {entity['entity_type']:25s} | {entity['text']:20s} | Score: {entity['score']:.2f}")

    except ImportError:
        print("\n\nPresidio not available, using simple de-identification:")
        print("=" * 60)
        deidentified = simple_deidentify(sample_text)
        print(deidentified)

    print("\n\n" + "=" * 70)
    print("IMPORTANT: Always verify de-identification results manually!")
    print("Automated tools may miss context-specific identifiers.")
    print("=" * 70)


if __name__ == "__main__":
    main()
