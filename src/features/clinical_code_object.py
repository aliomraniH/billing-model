"""Core data structures for clinical code extraction and validation."""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Any


@dataclass
class ExtractedEntity:
    """Represents a single extracted clinical entity with billing code mapping."""

    term: str
    code: str
    code_type: str  # 'ICD-10' or 'HCPCS'
    confidence: float
    source_span: str
    is_negated: bool = False
    is_historical: bool = False
    is_hypothetical: bool = False
    is_family_history: bool = False

    def is_billable(self) -> bool:
        """Check if entity represents a billable condition/procedure."""
        return not (self.is_negated or self.is_historical or
                   self.is_hypothetical or self.is_family_history)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedEntity':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class CodeSuggestion:
    """Represents a suggested billing code with confidence and rationale."""

    code: str
    code_type: str
    description: str
    confidence: float
    source: str  # 'nlp', 'cluster_consensus', 'manual'
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeSuggestion':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class ValidationFlag:
    """Represents a potential coding issue detected during validation."""

    flag_type: str  # 'REVENUE_LEAKAGE', 'COMPLIANCE_RISK', etc.
    code: str
    severity: str  # 'high', 'medium', 'low'
    confidence: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationFlag':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class ClinicalCodeObject:
    """
    Central data structure holding all clinical code extraction and validation data.

    This object represents a single medical claim with:
    - Original clinical documentation
    - NLP-extracted diagnoses and procedures
    - Cluster assignment and consensus suggestions
    - Validation flags for potential coding errors
    """

    claim_id: str
    visit_notes: str
    condition_text: str
    diagnoses: List[ExtractedEntity] = field(default_factory=list)
    procedures: List[ExtractedEntity] = field(default_factory=list)
    cluster_id: Optional[int] = None
    suggested_codes: List[CodeSuggestion] = field(default_factory=list)
    flags: Dict[str, Any] = field(default_factory=dict)
    flagged_codes: List[ValidationFlag] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def billable_diagnoses(self) -> List[ExtractedEntity]:
        """
        Return only billable diagnoses (non-negated, non-historical).

        Returns:
            List of ExtractedEntity objects that should be billed.
        """
        return [d for d in self.diagnoses if d.is_billable()]

    @property
    def billable_procedures(self) -> List[ExtractedEntity]:
        """
        Return only billable procedures (non-negated, non-historical).

        Returns:
            List of ExtractedEntity objects that should be billed.
        """
        return [p for p in self.procedures if p.is_billable()]

    @property
    def nlp_code_set(self) -> Set[str]:
        """
        Return set of all NLP-derived billing codes.

        Returns:
            Set of code strings from billable diagnoses and procedures.
        """
        codes = set()
        codes.update(d.code for d in self.billable_diagnoses)
        codes.update(p.code for p in self.billable_procedures)
        return codes

    @property
    def diagnosis_count(self) -> int:
        """Return count of billable diagnoses."""
        return len(self.billable_diagnoses)

    @property
    def procedure_count(self) -> int:
        """Return count of billable procedures."""
        return len(self.billable_procedures)

    @property
    def has_flags(self) -> bool:
        """Check if any validation flags exist."""
        return len(self.flagged_codes) > 0

    @property
    def high_severity_flags(self) -> List[ValidationFlag]:
        """Return only high severity validation flags."""
        return [f for f in self.flagged_codes if f.severity == 'high']

    def add_diagnosis(self, entity: ExtractedEntity) -> None:
        """Add a diagnosis entity."""
        if entity.code_type != 'ICD-10':
            raise ValueError(f"Diagnosis must be ICD-10, got {entity.code_type}")
        self.diagnoses.append(entity)

    def add_procedure(self, entity: ExtractedEntity) -> None:
        """Add a procedure entity."""
        if entity.code_type != 'HCPCS':
            raise ValueError(f"Procedure must be HCPCS, got {entity.code_type}")
        self.procedures.append(entity)

    def add_suggestion(self, suggestion: CodeSuggestion) -> None:
        """Add a code suggestion."""
        self.suggested_codes.append(suggestion)

    def add_flag(self, flag: ValidationFlag) -> None:
        """Add a validation flag."""
        self.flagged_codes.append(flag)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the object.
        """
        return {
            'claim_id': self.claim_id,
            'visit_notes': self.visit_notes,
            'condition_text': self.condition_text,
            'diagnoses': [d.to_dict() for d in self.diagnoses],
            'procedures': [p.to_dict() for p in self.procedures],
            'cluster_id': self.cluster_id,
            'suggested_codes': [s.to_dict() for s in self.suggested_codes],
            'flags': self.flags,
            'flagged_codes': [f.to_dict() for f in self.flagged_codes],
            'metadata': self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.

        Args:
            indent: Number of spaces for indentation (default: 2)

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClinicalCodeObject':
        """
        Create instance from dictionary.

        Args:
            data: Dictionary containing object data

        Returns:
            ClinicalCodeObject instance
        """
        # Convert nested structures
        diagnoses = [ExtractedEntity.from_dict(d) for d in data.get('diagnoses', [])]
        procedures = [ExtractedEntity.from_dict(p) for p in data.get('procedures', [])]
        suggested_codes = [CodeSuggestion.from_dict(s) for s in data.get('suggested_codes', [])]
        flagged_codes = [ValidationFlag.from_dict(f) for f in data.get('flagged_codes', [])]

        return cls(
            claim_id=data['claim_id'],
            visit_notes=data['visit_notes'],
            condition_text=data['condition_text'],
            diagnoses=diagnoses,
            procedures=procedures,
            cluster_id=data.get('cluster_id'),
            suggested_codes=suggested_codes,
            flags=data.get('flags', {}),
            flagged_codes=flagged_codes,
            metadata=data.get('metadata', {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'ClinicalCodeObject':
        """
        Create instance from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            ClinicalCodeObject instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"ClinicalCodeObject(claim_id='{self.claim_id}', "
                f"diagnoses={self.diagnosis_count}, "
                f"procedures={self.procedure_count}, "
                f"cluster_id={self.cluster_id}, "
                f"flags={len(self.flagged_codes)})")
