"""
Gap Analysis: Compare NLP-extracted codes against billed codes.
Identifies revenue leakage (under-coding) and compliance risks (over-coding).
"""
import pandas as pd
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class FlagType(Enum):
    """Types of validation flags."""
    REVENUE_LEAKAGE = "revenue_leakage"        # Documented but not billed
    COMPLIANCE_RISK = "compliance_risk"         # Billed but not documented
    SPECIFICITY_MISMATCH = "specificity_mismatch"  # Parent code vs child
    MISSING_EXTENSION = "missing_extension"     # Missing 7th character


@dataclass
class GapAnalysisResult:
    """Result of gap analysis for a single claim."""
    claim_id: str
    flags: List[Dict]
    revenue_impact: float  # Estimated dollar impact
    risk_score: float      # 0-1 compliance risk score

    @property
    def has_issues(self) -> bool:
        return len(self.flags) > 0


class GapAnalyzer:
    """
    Analyzes gaps between clinical documentation and billing codes.

    Gap Types:
    1. Revenue Leakage: NLP found code, but not in billed codes
    2. Compliance Risk: Billed code has no NLP/documentation support
    3. Specificity Mismatch: Billed parent code when child is documented
    4. Missing Extension: ICD-10 code missing required 7th character
    """

    def __init__(self, db_connection=None):
        """
        Initialize gap analyzer.

        Args:
            db_connection: Database connection for code hierarchy lookups
        """
        self.db = db_connection
        self._hierarchy_cache = {}

    def _get_code_parent(self, code: str) -> str:
        """Get parent code in ICD-10 hierarchy."""
        if len(code) > 3:
            return code[:3]
        return code

    def _requires_7th_character(self, code: str) -> bool:
        """Check if ICD-10 code requires 7th character extension."""
        # Injury codes (S/T), Pregnancy (O), External causes (V-Y)
        if code and code[0] in ['S', 'T', 'O', 'V', 'W', 'X', 'Y']:
            return len(code) < 7
        return False

    def analyze_claim(
        self,
        clinical_object: 'ClinicalCodeObject',
        billed_codes: List[str],
        confidence_threshold: float = 0.90
    ) -> GapAnalysisResult:
        """
        Analyze a single claim for coding gaps.

        Args:
            clinical_object: ClinicalCodeObject with NLP-extracted codes
            billed_codes: List of codes actually billed
            confidence_threshold: Minimum confidence for flagging

        Returns:
            GapAnalysisResult with identified issues
        """
        flags = []
        billed_set = set(billed_codes)

        # Get high-confidence NLP codes
        nlp_diagnoses = {
            e.code for e in clinical_object.billable_diagnoses
            if e.confidence >= confidence_threshold
        }
        nlp_procedures = {
            e.code for e in clinical_object.billable_procedures
            if e.confidence >= confidence_threshold
        }
        nlp_codes = nlp_diagnoses | nlp_procedures

        # 1. Revenue Leakage: In NLP but not billed
        for code in nlp_codes - billed_set:
            entity = next(
                (e for e in clinical_object.extracted_diagnoses + clinical_object.extracted_procedures
                 if e.code == code),
                None
            )
            flags.append({
                'type': FlagType.REVENUE_LEAKAGE.value,
                'code': code,
                'confidence': entity.confidence if entity else 0.0,
                'evidence': entity.term if entity else 'Unknown',
                'message': f"Documented '{entity.term if entity else code}' (confidence: {entity.confidence:.0%}) not found in billed codes",
                'priority': 'HIGH' if code.startswith('E') or code.startswith('I') else 'MEDIUM'  # HCC-relevant codes
            })
            clinical_object.missing_codes.append(code)

        # 2. Compliance Risk: Billed but not in NLP
        suggested_set = set(clinical_object.suggested_codes)
        for code in billed_set - nlp_codes:
            # Check if it's a consensus code (acceptable)
            if code in suggested_set:
                continue

            flags.append({
                'type': FlagType.COMPLIANCE_RISK.value,
                'code': code,
                'message': f"Billed code {code} has no supporting documentation",
                'priority': 'HIGH'
            })
            clinical_object.unsupported_codes.append(code)

        # 3. Specificity Mismatch: Parent billed, child documented
        for nlp_code in nlp_codes:
            parent = self._get_code_parent(nlp_code)
            if parent in billed_set and nlp_code not in billed_set:
                flags.append({
                    'type': FlagType.SPECIFICITY_MISMATCH.value,
                    'billed_code': parent,
                    'suggested_code': nlp_code,
                    'message': f"More specific code {nlp_code} documented, but parent {parent} billed",
                    'priority': 'MEDIUM'
                })

        # 4. Missing 7th Character
        for code in billed_codes:
            if self._requires_7th_character(code):
                flags.append({
                    'type': FlagType.MISSING_EXTENSION.value,
                    'code': code,
                    'message': f"Code {code} requires 7th character extension (A/D/S)",
                    'priority': 'HIGH'
                })

        # Update clinical object flags
        clinical_object.flags['potential_under_coding'] = any(
            f['type'] == FlagType.REVENUE_LEAKAGE.value for f in flags
        )
        clinical_object.flags['potential_over_coding'] = any(
            f['type'] == FlagType.COMPLIANCE_RISK.value for f in flags
        )
        clinical_object.flags['specificity_mismatch'] = any(
            f['type'] == FlagType.SPECIFICITY_MISMATCH.value for f in flags
        )
        clinical_object.flags['missing_7th_character'] = any(
            f['type'] == FlagType.MISSING_EXTENSION.value for f in flags
        )

        # Calculate risk score
        high_priority_count = sum(1 for f in flags if f.get('priority') == 'HIGH')
        risk_score = min(1.0, high_priority_count * 0.25)

        return GapAnalysisResult(
            claim_id=clinical_object.claim_id,
            flags=flags,
            revenue_impact=len([f for f in flags if f['type'] == FlagType.REVENUE_LEAKAGE.value]) * 150,  # Estimated
            risk_score=risk_score
        )

    def batch_analyze(
        self,
        clinical_objects: List['ClinicalCodeObject'],
        billed_df: pd.DataFrame,
        claim_id_column: str = 'claim_id',
        codes_column: str = 'codes'
    ) -> pd.DataFrame:
        """
        Batch analyze multiple claims.

        Args:
            clinical_objects: List of ClinicalCodeObjects
            billed_df: DataFrame with billed codes per claim
            claim_id_column: Column with claim IDs
            codes_column: Column with billed codes (list or comma-separated)

        Returns:
            DataFrame with analysis results
        """
        # Build claim_id to billed codes mapping
        def parse_codes(codes):
            if isinstance(codes, list):
                return codes
            if isinstance(codes, str):
                return [c.strip() for c in codes.split(',')]
            return []

        billed_map = {
            row[claim_id_column]: parse_codes(row[codes_column])
            for _, row in billed_df.iterrows()
        }

        results = []
        for obj in clinical_objects:
            billed_codes = billed_map.get(obj.claim_id, [])
            result = self.analyze_claim(obj, billed_codes)
            results.append({
                'claim_id': result.claim_id,
                'flag_count': len(result.flags),
                'revenue_leakage_flags': sum(1 for f in result.flags if f['type'] == FlagType.REVENUE_LEAKAGE.value),
                'compliance_risk_flags': sum(1 for f in result.flags if f['type'] == FlagType.COMPLIANCE_RISK.value),
                'revenue_impact': result.revenue_impact,
                'risk_score': result.risk_score,
                'flags_json': result.flags
            })

        return pd.DataFrame(results)
