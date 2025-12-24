"""Tests for gap analyzer."""

import pytest
import pandas as pd

from src.features.gap_analyzer import GapAnalyzer, FlagType
from src.features.clinical_code_object import ClinicalCodeObject, ExtractedEntity


class TestGapAnalyzer:
    """Test gap analyzer functionality."""

    def test_revenue_leakage_detection(self):
        """Test detection of documented but not billed codes."""
        analyzer = GapAnalyzer()

        # Create clinical object with diagnosis
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Patient has diabetes and hypertension",
            condition_text="DM2, HTN"
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

        # Billed codes missing I10
        billed_codes = {"E119"}

        flags = analyzer.analyze_claim(obj, billed_codes)

        # Should detect I10 as revenue leakage
        revenue_flags = [f for f in flags if f.flag_type == FlagType.REVENUE_LEAKAGE.value]
        assert len(revenue_flags) == 1
        assert revenue_flags[0].code == "I10"

    def test_compliance_risk_detection(self):
        """Test detection of billed but not documented codes."""
        analyzer = GapAnalyzer()

        # Create clinical object
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Diabetes follow-up",
            condition_text="DM2"
        )

        obj.add_diagnosis(ExtractedEntity(
            term="diabetes",
            code="E119",
            code_type="ICD-10",
            confidence=0.95,
            source_span="diabetes"
        ))

        # Billed codes include extra code not in documentation
        billed_codes = {"E119", "J449"}  # J449 not documented

        flags = analyzer.analyze_claim(obj, billed_codes)

        # Should detect J449 as compliance risk
        compliance_flags = [f for f in flags if f.flag_type == FlagType.COMPLIANCE_RISK.value]
        assert len(compliance_flags) >= 1
        assert any(f.code == "J449" for f in compliance_flags)

    def test_missing_7th_character(self):
        """Test detection of ICD-10 codes missing 7th character."""
        analyzer = GapAnalyzer()

        # S-code (injury) without 7th character
        assert analyzer._is_missing_7th("S52") is True

        # T-code (poisoning) without 7th character
        assert analyzer._is_missing_7th("T36") is True

        # Complete S-code with 7th character
        assert analyzer._is_missing_7th("S52501A") is False

        # Non-S/T code
        assert analyzer._is_missing_7th("E119") is False

    def test_parent_code_hierarchy(self):
        """Test ICD-10 hierarchy parent code detection."""
        analyzer = GapAnalyzer()

        # E11.65 has parents E11.6 and E11
        parents = analyzer._get_parent_codes("E1165")

        assert "E116" in parents
        assert "E11" in parents
        assert "E1" in parents

    def test_batch_analyze(self):
        """Test batch analysis of multiple claims."""
        analyzer = GapAnalyzer()

        # Create multiple clinical objects
        objs = []

        obj1 = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Diabetes",
            condition_text="DM2"
        )
        obj1.add_diagnosis(ExtractedEntity(
            term="diabetes", code="E119", code_type="ICD-10",
            confidence=0.95, source_span="diabetes"
        ))
        objs.append(obj1)

        obj2 = ClinicalCodeObject(
            claim_id="TEST002",
            visit_notes="Hypertension",
            condition_text="HTN"
        )
        obj2.add_diagnosis(ExtractedEntity(
            term="hypertension", code="I10", code_type="ICD-10",
            confidence=0.90, source_span="hypertension"
        ))
        objs.append(obj2)

        # Billed codes
        billed_df = pd.DataFrame([
            {'claim_id': 'TEST001', 'codes': 'E119,I10'},  # Extra code
            {'claim_id': 'TEST002', 'codes': 'I10'}
        ])

        # Run batch analysis
        flags_df = analyzer.batch_analyze(objs, billed_df)

        assert len(flags_df) > 0
        assert 'claim_id' in flags_df.columns
        assert 'flag_type' in flags_df.columns

    def test_generate_report(self):
        """Test report generation."""
        analyzer = GapAnalyzer()

        # Create sample flags dataframe
        flags_df = pd.DataFrame([
            {
                'claim_id': 'TEST001',
                'flag_type': FlagType.REVENUE_LEAKAGE.value,
                'code': 'E119',
                'severity': 'high',
                'confidence': 0.95,
                'rationale': 'Test'
            },
            {
                'claim_id': 'TEST001',
                'flag_type': FlagType.COMPLIANCE_RISK.value,
                'code': 'I10',
                'severity': 'high',
                'confidence': 0.90,
                'rationale': 'Test'
            }
        ])

        report = analyzer.generate_report(flags_df)

        assert report['total_flags'] == 2
        assert report['high_severity_count'] == 2
        assert report['claims_with_flags'] == 1

    def test_calculate_leakage_metrics(self):
        """Test leakage metrics calculation."""
        analyzer = GapAnalyzer()

        # Create clinical objects
        obj = ClinicalCodeObject(
            claim_id="TEST001",
            visit_notes="Diabetes and hypertension",
            condition_text="DM2, HTN"
        )
        obj.add_diagnosis(ExtractedEntity(
            term="diabetes", code="E119", code_type="ICD-10",
            confidence=0.95, source_span="diabetes"
        ))
        obj.add_diagnosis(ExtractedEntity(
            term="hypertension", code="I10", code_type="ICD-10",
            confidence=0.90, source_span="hypertension"
        ))

        objs = [obj]

        # Billed codes (missing I10)
        billed_df = pd.DataFrame([
            {'claim_id': 'TEST001', 'codes': 'E119'}
        ])

        metrics = analyzer.calculate_leakage_metrics(objs, billed_df)

        assert metrics['total_nlp_codes'] == 2
        assert metrics['total_billed_codes'] == 1
        assert metrics['missing_codes'] == 1  # I10 is missing
        assert metrics['leakage_ratio'] == 1.0  # 1 missing / 1 billed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
