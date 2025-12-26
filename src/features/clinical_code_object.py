"""
ClinicalCodeObject: Core data structure for NLP-derived billing intelligence.
Each claim gets one ClinicalCodeObject containing extracted codes and validation flags.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json

@dataclass
class ExtractedEntity:
    """Single extracted clinical entity with code mapping."""
    term: str                           # Original text span
    code: str                           # ICD-10 or HCPCS code
    code_type: str                      # 'ICD-10' or 'HCPCS'
    confidence: float                   # 0.0-1.0 confidence score
    source_span: str                    # Context from original text
    is_negated: bool = False            # ConText negation detection
    is_historical: bool = False         # Historical vs current condition
    is_family_history: bool = False     # Family history flag

    def is_billable(self) -> bool:
        """Check if entity is billable (not negated, not historical)."""
        return not self.is_negated and not self.is_family_history

@dataclass
class ClinicalCodeObject:
    """
    Container for all NLP-derived intelligence for a single claim.
    Used for code suggestion, cluster consensus, and gap analysis.
    """
    claim_id: str
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Extracted entities
    extracted_diagnoses: List[ExtractedEntity] = field(default_factory=list)
    extracted_procedures: List[ExtractedEntity] = field(default_factory=list)

    # Cluster consensus suggestions
    suggested_codes: List[str] = field(default_factory=list)
    cluster_id: Optional[int] = None

    # Validation flags
    flags: Dict[str, bool] = field(default_factory=lambda: {
        'potential_under_coding': False,
        'potential_over_coding': False,
        'documentation_gap': False,
        'specificity_mismatch': False,
        'missing_7th_character': False
    })

    # Gap analysis results
    missing_codes: List[str] = field(default_factory=list)      # Revenue leakage
    unsupported_codes: List[str] = field(default_factory=list)  # Compliance risk

    @property
    def billable_diagnoses(self) -> List[ExtractedEntity]:
        """Get only billable diagnosis entities."""
        return [e for e in self.extracted_diagnoses if e.is_billable()]

    @property
    def billable_procedures(self) -> List[ExtractedEntity]:
        """Get only billable procedure entities."""
        return [e for e in self.extracted_procedures if e.is_billable()]

    def to_dict(self) -> Dict:
        """Serialize to dictionary for database storage."""
        return {
            'claim_id': self.claim_id,
            'processing_timestamp': self.processing_timestamp.isoformat(),
            'extracted_diagnoses': [
                {
                    'term': e.term,
                    'code': e.code,
                    'code_type': e.code_type,
                    'confidence': e.confidence,
                    'is_negated': e.is_negated,
                    'is_historical': e.is_historical
                }
                for e in self.extracted_diagnoses
            ],
            'extracted_procedures': [
                {
                    'term': e.term,
                    'code': e.code,
                    'code_type': e.code_type,
                    'confidence': e.confidence
                }
                for e in self.extracted_procedures
            ],
            'suggested_codes': self.suggested_codes,
            'flags': self.flags,
            'missing_codes': self.missing_codes,
            'unsupported_codes': self.unsupported_codes
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ClinicalCodeObject':
        """Deserialize from dictionary."""
        obj = cls(claim_id=data['claim_id'])
        obj.processing_timestamp = datetime.fromisoformat(data.get('processing_timestamp', datetime.utcnow().isoformat()))

        for diag in data.get('extracted_diagnoses', []):
            obj.extracted_diagnoses.append(ExtractedEntity(
                term=diag['term'],
                code=diag['code'],
                code_type=diag['code_type'],
                confidence=diag['confidence'],
                source_span=diag.get('source_span', ''),
                is_negated=diag.get('is_negated', False),
                is_historical=diag.get('is_historical', False)
            ))

        for proc in data.get('extracted_procedures', []):
            obj.extracted_procedures.append(ExtractedEntity(
                term=proc['term'],
                code=proc['code'],
                code_type=proc['code_type'],
                confidence=proc['confidence'],
                source_span=proc.get('source_span', '')
            ))

        obj.suggested_codes = data.get('suggested_codes', [])
        obj.flags = data.get('flags', obj.flags)
        obj.missing_codes = data.get('missing_codes', [])
        obj.unsupported_codes = data.get('unsupported_codes', [])

        return obj
