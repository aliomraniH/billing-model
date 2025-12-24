"""
Gap analysis and validation for medical billing codes.

This module identifies potential coding errors including:
- Revenue leakage: Documented but not billed
- Compliance risk: Billed but not documented
- Specificity mismatches: Parent code billed, child code documented
- Missing extensions: ICD-10 codes missing required 7th character
"""

from enum import Enum
from typing import List, Set, Dict, Optional, Tuple
from collections import Counter

import pandas as pd
from tqdm import tqdm

from src.features.clinical_code_object import ClinicalCodeObject, ValidationFlag
from src.features.cluster_consensus import ClusterConsensus


class FlagType(Enum):
    """Types of validation flags."""

    REVENUE_LEAKAGE = "revenue_leakage"
    COMPLIANCE_RISK = "compliance_risk"
    SPECIFICITY_MISMATCH = "specificity_mismatch"
    MISSING_EXTENSION = "missing_extension"
    UNBILLED_PROCEDURE = "unbilled_procedure"
    MISSING_CONSENSUS = "missing_consensus"


class Severity(Enum):
    """Severity levels for flags."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapAnalyzer:
    """
    Analyze gaps between documented and billed codes.

    This class performs comprehensive validation to identify:
    1. Revenue leakage opportunities
    2. Compliance risk areas
    3. Coding quality issues
    """

    def __init__(self):
        """Initialize gap analyzer."""
        # ICD-10 codes requiring 7th character (injuries, external causes)
        self.codes_requiring_7th = {'S', 'T', 'V', 'W', 'X', 'Y'}

        # ICD-10 hierarchy patterns (simplified)
        self.icd10_hierarchy_cache: Dict[str, Set[str]] = {}

    def _get_parent_codes(self, code: str) -> Set[str]:
        """
        Get all parent codes in ICD-10 hierarchy.

        For example, E11.65 has parents: E11.6, E11

        Args:
            code: ICD-10 code

        Returns:
            Set of parent codes
        """
        if code in self.icd10_hierarchy_cache:
            return self.icd10_hierarchy_cache[code]

        parents = set()

        # Remove decimal if present
        code_clean = code.replace('.', '')

        # Generate parent codes
        for i in range(len(code_clean) - 1, 0, -1):
            parent = code_clean[:i]
            parents.add(parent)

        self.icd10_hierarchy_cache[code] = parents
        return parents

    def _requires_7th_character(self, code: str) -> bool:
        """
        Check if ICD-10 code requires a 7th character.

        Args:
            code: ICD-10 code

        Returns:
            True if 7th character is required
        """
        if not code:
            return False

        # Check if code starts with S, T, V, W, X, Y
        return code[0] in self.codes_requiring_7th

    def _is_missing_7th(self, code: str) -> bool:
        """
        Check if code is missing required 7th character.

        Args:
            code: ICD-10 code

        Returns:
            True if 7th character is required but missing
        """
        code_clean = code.replace('.', '')

        if self._requires_7th_character(code_clean):
            # Most injury/external cause codes need 7 characters
            # (simplified check - actual rules are more complex)
            return len(code_clean) < 7

        return False

    def analyze_claim(
        self,
        clinical_obj: ClinicalCodeObject,
        billed_codes: Set[str],
        cluster_consensus: Optional[ClusterConsensus] = None
    ) -> List[ValidationFlag]:
        """
        Analyze a single claim for coding gaps.

        Args:
            clinical_obj: Clinical object with NLP-extracted codes
            billed_codes: Set of actually billed codes
            cluster_consensus: Optional consensus model for additional validation

        Returns:
            List of validation flags
        """
        flags = []

        nlp_codes = clinical_obj.nlp_code_set

        # 1. REVENUE LEAKAGE: Codes in NLP but not billed
        missing_codes = nlp_codes - billed_codes

        for code in missing_codes:
            # Find the entity that generated this code
            all_entities = clinical_obj.billable_diagnoses + clinical_obj.billable_procedures

            for entity in all_entities:
                if entity.code == code:
                    flag = ValidationFlag(
                        flag_type=FlagType.REVENUE_LEAKAGE.value,
                        code=code,
                        severity=Severity.HIGH.value,
                        confidence=entity.confidence,
                        rationale=f"Code '{code}' documented ('{entity.term}') but not billed. "
                                 f"Potential revenue loss."
                    )
                    flags.append(flag)
                    break

        # 2. COMPLIANCE RISK: Codes billed but not in NLP
        unbilled_nlp_codes = billed_codes - nlp_codes

        for code in unbilled_nlp_codes:
            # Check if it's a parent of a documented code
            is_parent = any(
                code in self._get_parent_codes(nlp_code)
                for nlp_code in nlp_codes
            )

            if is_parent:
                flag = ValidationFlag(
                    flag_type=FlagType.SPECIFICITY_MISMATCH.value,
                    code=code,
                    severity=Severity.MEDIUM.value,
                    confidence=0.8,
                    rationale=f"Non-specific code '{code}' billed, but more specific "
                             f"code documented. Consider using specific code."
                )
            else:
                flag = ValidationFlag(
                    flag_type=FlagType.COMPLIANCE_RISK.value,
                    code=code,
                    severity=Severity.HIGH.value,
                    confidence=0.9,
                    rationale=f"Code '{code}' billed but not found in clinical documentation. "
                             f"Compliance risk."
                )

            flags.append(flag)

        # 3. MISSING 7TH CHARACTER: ICD-10 codes that need extension
        all_codes = nlp_codes | billed_codes

        for code in all_codes:
            if self._is_missing_7th(code):
                flag = ValidationFlag(
                    flag_type=FlagType.MISSING_EXTENSION.value,
                    code=code,
                    severity=Severity.MEDIUM.value,
                    confidence=1.0,
                    rationale=f"Code '{code}' requires 7th character extension. "
                             f"Incomplete code."
                )
                flags.append(flag)

        # 4. UNBILLED PROCEDURES: Procedures in NLP but not billed
        procedure_codes = {p.code for p in clinical_obj.billable_procedures}
        unbilled_procedures = procedure_codes - billed_codes

        for code in unbilled_procedures:
            for proc in clinical_obj.billable_procedures:
                if proc.code == code:
                    flag = ValidationFlag(
                        flag_type=FlagType.UNBILLED_PROCEDURE.value,
                        code=code,
                        severity=Severity.HIGH.value,
                        confidence=proc.confidence,
                        rationale=f"Procedure '{proc.term}' ({code}) documented but not billed. "
                                 f"Revenue opportunity."
                    )
                    flags.append(flag)
                    break

        # 5. MISSING CONSENSUS CODES: High-prevalence cluster codes not present
        if cluster_consensus and clinical_obj.cluster_id is not None:
            consensus_codes = cluster_consensus.get_consensus_codes(clinical_obj.cluster_id)

            # Get high-confidence consensus codes not in claim
            for code, prevalence in consensus_codes.items():
                if code not in nlp_codes and code not in billed_codes:
                    if prevalence >= 0.8:  # Very high prevalence
                        flag = ValidationFlag(
                            flag_type=FlagType.MISSING_CONSENSUS.value,
                            code=code,
                            severity=Severity.MEDIUM.value,
                            confidence=prevalence,
                            rationale=f"Code '{code}' present in {prevalence:.0%} of similar "
                                     f"cases but missing from this claim. Review clinical notes."
                        )
                        flags.append(flag)

        return flags

    def batch_analyze(
        self,
        clinical_objects: List[ClinicalCodeObject],
        billed_df: pd.DataFrame,
        claim_id_col: str = 'claim_id',
        codes_col: str = 'codes',
        cluster_consensus: Optional[ClusterConsensus] = None
    ) -> pd.DataFrame:
        """
        Analyze multiple claims.

        Args:
            clinical_objects: List of clinical objects
            billed_df: DataFrame with billed codes (claim_id, codes columns)
            claim_id_col: Name of claim ID column
            codes_col: Name of codes column (comma-separated or list)
            cluster_consensus: Optional consensus model

        Returns:
            DataFrame with all validation flags
        """
        print("\n" + "=" * 60)
        print("Running Gap Analysis")
        print("=" * 60)

        # Create lookup for billed codes
        billed_lookup = {}
        for _, row in billed_df.iterrows():
            claim_id = str(row[claim_id_col])
            codes = row[codes_col]

            if isinstance(codes, str):
                code_set = set(c.strip() for c in codes.split(',') if c.strip())
            elif isinstance(codes, list):
                code_set = set(codes)
            else:
                code_set = set()

            billed_lookup[claim_id] = code_set

        # Analyze each claim
        all_flags = []

        for obj in tqdm(clinical_objects, desc="Analyzing claims"):
            billed_codes = billed_lookup.get(obj.claim_id, set())

            flags = self.analyze_claim(obj, billed_codes, cluster_consensus)

            # Add flags to object
            for flag in flags:
                obj.add_flag(flag)

            # Add to results
            for flag in flags:
                all_flags.append({
                    'claim_id': obj.claim_id,
                    'flag_type': flag.flag_type,
                    'code': flag.code,
                    'severity': flag.severity,
                    'confidence': flag.confidence,
                    'rationale': flag.rationale
                })

        flags_df = pd.DataFrame(all_flags)

        if len(flags_df) > 0:
            # Summary
            print(f"\nFound {len(flags_df)} validation flags:")
            print(flags_df['flag_type'].value_counts().to_string())

            print(f"\nBy severity:")
            print(flags_df['severity'].value_counts().to_string())
        else:
            print("\nNo validation flags found")

        return flags_df

    def generate_report(self, flags_df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate summary report from validation flags.

        Args:
            flags_df: DataFrame with validation flags

        Returns:
            Dictionary with report statistics
        """
        if len(flags_df) == 0:
            return {
                'total_flags': 0,
                'by_type': {},
                'by_severity': {},
                'claims_with_flags': 0
            }

        report = {
            'total_flags': len(flags_df),
            'by_type': flags_df['flag_type'].value_counts().to_dict(),
            'by_severity': flags_df['severity'].value_counts().to_dict(),
            'claims_with_flags': flags_df['claim_id'].nunique(),
            'avg_flags_per_claim': len(flags_df) / flags_df['claim_id'].nunique(),
            'high_severity_count': (flags_df['severity'] == 'high').sum(),
            'revenue_leakage_codes': flags_df[
                flags_df['flag_type'] == FlagType.REVENUE_LEAKAGE.value
            ]['code'].nunique(),
            'compliance_risk_codes': flags_df[
                flags_df['flag_type'] == FlagType.COMPLIANCE_RISK.value
            ]['code'].nunique()
        }

        return report

    def calculate_leakage_metrics(
        self,
        clinical_objects: List[ClinicalCodeObject],
        billed_df: pd.DataFrame,
        claim_id_col: str = 'claim_id',
        codes_col: str = 'codes'
    ) -> Dict[str, float]:
        """
        Calculate revenue leakage metrics.

        Args:
            clinical_objects: List of clinical objects
            billed_df: DataFrame with billed codes
            claim_id_col: Name of claim ID column
            codes_col: Name of codes column

        Returns:
            Dictionary with leakage metrics
        """
        total_nlp_codes = 0
        total_billed_codes = 0
        total_missing = 0
        total_unbilled = 0

        billed_lookup = {}
        for _, row in billed_df.iterrows():
            claim_id = str(row[claim_id_col])
            codes = row[codes_col]

            if isinstance(codes, str):
                code_set = set(c.strip() for c in codes.split(',') if c.strip())
            else:
                code_set = set(codes) if codes else set()

            billed_lookup[claim_id] = code_set

        for obj in clinical_objects:
            nlp_codes = obj.nlp_code_set
            billed_codes = billed_lookup.get(obj.claim_id, set())

            total_nlp_codes += len(nlp_codes)
            total_billed_codes += len(billed_codes)
            total_missing += len(nlp_codes - billed_codes)
            total_unbilled += len(billed_codes - nlp_codes)

        return {
            'total_nlp_codes': total_nlp_codes,
            'total_billed_codes': total_billed_codes,
            'missing_codes': total_missing,
            'unbilled_codes': total_unbilled,
            'leakage_ratio': total_missing / total_billed_codes if total_billed_codes > 0 else 0,
            'compliance_risk_ratio': total_unbilled / total_billed_codes if total_billed_codes > 0 else 0,
            'avg_nlp_per_claim': total_nlp_codes / len(clinical_objects),
            'avg_billed_per_claim': total_billed_codes / len(clinical_objects)
        }


def main():
    """Demo gap analysis on sample data."""
    from src.features.clinical_code_object import ExtractedEntity

    print("\n" + "=" * 70)
    print(" " * 25 + "GAP ANALYZER DEMO")
    print("=" * 70 + "\n")

    # Create sample clinical objects
    clinical_objects = []

    # Claim 1: Revenue leakage (documented but not billed)
    obj1 = ClinicalCodeObject(
        claim_id="TEST001",
        visit_notes="Diabetes and hypertension follow-up",
        condition_text="DM2, HTN"
    )
    obj1.add_diagnosis(ExtractedEntity(
        term="diabetes", code="E119", code_type="ICD-10",
        confidence=0.95, source_span="diabetes"
    ))
    obj1.add_diagnosis(ExtractedEntity(
        term="hypertension", code="I10", code_type="ICD-10",
        confidence=0.90, source_span="hypertension"
    ))
    clinical_objects.append(obj1)

    # Claim 2: Compliance risk (billed but not documented)
    obj2 = ClinicalCodeObject(
        claim_id="TEST002",
        visit_notes="Routine follow-up",
        condition_text="Routine visit"
    )
    obj2.add_diagnosis(ExtractedEntity(
        term="diabetes", code="E119", code_type="ICD-10",
        confidence=0.95, source_span="diabetes"
    ))
    clinical_objects.append(obj2)

    # Billed codes
    billed_df = pd.DataFrame([
        {'claim_id': 'TEST001', 'codes': 'E119'},  # Missing I10
        {'claim_id': 'TEST002', 'codes': 'E119,I10,J449'}  # Extra codes
    ])

    # Run analysis
    analyzer = GapAnalyzer()
    flags_df = analyzer.batch_analyze(clinical_objects, billed_df)

    print("\n" + "=" * 60)
    print("VALIDATION FLAGS")
    print("=" * 60)
    print(flags_df.to_string(index=False))

    # Generate report
    report = analyzer.generate_report(flags_df)
    print("\n" + "=" * 60)
    print("REPORT SUMMARY")
    print("=" * 60)
    for key, value in report.items():
        print(f"  {key}: {value}")

    # Calculate leakage metrics
    metrics = analyzer.calculate_leakage_metrics(clinical_objects, billed_df)
    print("\n" + "=" * 60)
    print("LEAKAGE METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
