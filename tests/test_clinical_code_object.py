"""Tests for clinical code object data structures."""

import json
import pytest

from src.features.clinical_code_object import (
    ExtractedEntity,
    CodeSuggestion,
    ValidationFlag,
    ClinicalCodeObject
)


class TestExtractedEntity:
    """Test ExtractedEntity dataclass."""

    def test_create_entity(self):
        """Test entity creation."""
        entity = ExtractedEntity(
            term="diabetes",
            code="E119",
            code_type="ICD-10",
            confidence=0.95,
            source_span="Patient has diabetes",
            is_negated=False,
            is_historical=False
        )

        assert entity.term == "diabetes"
        assert entity.code == "E119"
        assert entity.confidence == 0.95

    def test_is_billable_positive(self):
        """Test billable entity."""
        entity = ExtractedEntity(
            term="diabetes",
            code="E119",
            code_type="ICD-10",
            confidence=0.95,
            source_span="Patient has diabetes"
        )

        assert entity.is_billable() is True

    def test_is_billable_negated(self):
        """Test negated entity is not billable."""
        entity = ExtractedEntity(
            term="chest pain",
            code="R079",
            code_type="ICD-10",
            confidence=0.90,
            source_span="Patient denies chest pain",
            is_negated=True
        )

        assert entity.is_billable() is False

    def test_is_billable_historical(self):
        """Test historical entity is not billable."""
        entity = ExtractedEntity(
            term="MI",
            code="I2101",
            code_type="ICD-10",
            confidence=0.92,
            source_span="History of MI",
            is_historical=True
        )

        assert entity.is_billable() is False

    def test_serialization(self):
        """Test entity serialization."""
        entity = ExtractedEntity(
            term="hypertension",
            code="I10",
            code_type="ICD-10",
            confidence=0.88,
            source_span="Patient has hypertension"
        )

        # To dict
        entity_dict = entity.to_dict()
        assert entity_dict['term'] == "hypertension"
        assert entity_dict['code'] == "I10"

        # From dict
        restored = ExtractedEntity.from_dict(entity_dict)
        assert restored.term == entity.term
        assert restored.code == entity.code


class TestClinicalCodeObject:
    """Test ClinicalCodeObject."""

    def test_create_object(self):
        """Test object creation."""
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Patient presents with diabetes",
            condition_text="Type 2 diabetes"
        )

        assert obj.claim_id == "TEST001"
        assert obj.diagnosis_count == 0
        assert obj.procedure_count == 0

    def test_add_diagnosis(self):
        """Test adding diagnosis."""
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Diabetes",
            condition_text="DM2"
        )

        entity = ExtractedEntity(
            term="diabetes",
            code="E119",
            code_type="ICD-10",
            confidence=0.95,
            source_span="Patient has diabetes"
        )

        obj.add_diagnosis(entity)

        assert obj.diagnosis_count == 1
        assert len(obj.billable_diagnoses) == 1

    def test_add_procedure(self):
        """Test adding procedure."""
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="ECG performed",
            condition_text=""
        )

        entity = ExtractedEntity(
            term="ECG",
            code="93000",
            code_type="HCPCS",
            confidence=0.90,
            source_span="ECG performed"
        )

        obj.add_procedure(entity)

        assert obj.procedure_count == 1
        assert len(obj.billable_procedures) == 1

    def test_billable_diagnoses_filter(self):
        """Test billable diagnoses filtering."""
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Test",
            condition_text="Test"
        )

        # Add billable diagnosis
        obj.add_diagnosis(ExtractedEntity(
            term="diabetes",
            code="E119",
            code_type="ICD-10",
            confidence=0.95,
            source_span="diabetes"
        ))

        # Add negated diagnosis
        obj.add_diagnosis(ExtractedEntity(
            term="chest pain",
            code="R079",
            code_type="ICD-10",
            confidence=0.90,
            source_span="no chest pain",
            is_negated=True
        ))

        # Should only return non-negated
        assert len(obj.billable_diagnoses) == 1
        assert obj.billable_diagnoses[0].code == "E119"

    def test_nlp_code_set(self):
        """Test NLP code set property."""
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Test",
            condition_text="Test"
        )

        obj.add_diagnosis(ExtractedEntity(
            term="diabetes",
            code="E119",
            code_type="ICD-10",
            confidence=0.95,
            source_span="diabetes"
        ))

        obj.add_diagnosis(ExtractedEntity(
            term="hypertension",
            code="I10",
            code_type="ICD-10",
            confidence=0.90,
            source_span="hypertension"
        ))

        code_set = obj.nlp_code_set
        assert code_set == {"E119", "I10"}

    def test_json_serialization(self):
        """Test JSON round-trip."""
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Diabetes follow-up",
            condition_text="Type 2 DM"
        )

        obj.add_diagnosis(ExtractedEntity(
            term="diabetes",
            code="E119",
            code_type="ICD-10",
            confidence=0.95,
            source_span="diabetes"
        ))

        # To JSON
        json_str = obj.to_json()
        assert isinstance(json_str, str)

        # From JSON
        restored = ClinicalCodeObject.from_json(json_str)
        assert restored.claim_id == obj.claim_id
        assert restored.diagnosis_count == obj.diagnosis_count
        assert restored.billable_diagnoses[0].code == "E119"

    def test_add_flag(self):
        """Test adding validation flag."""
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Test",
            condition_text="Test"
        )

        flag = ValidationFlag(
            flag_type="revenue_leakage",
            code="E119",
            severity="high",
            confidence=0.95,
            rationale="Code documented but not billed"
        )

        obj.add_flag(flag)

        assert obj.has_flags is True
        assert len(obj.flagged_codes) == 1
        assert len(obj.high_severity_flags) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
