"""
Cluster-based consensus code suggestion using weak supervision.

This module implements a cluster consensus algorithm that identifies
commonly co-occurring codes within patient clusters to suggest additional
billing codes that may have been missed.

Mathematical Framework:
    For cluster C_k, prevalence P of code x:
    P(x | C_k) = Σ 1(x ∈ Claim_i) / N

    If P(x | C_k) > τ (threshold), x is a "Consensus Code"
"""

from collections import defaultdict, Counter
from typing import List, Dict, Set, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import CONSENSUS_THRESHOLD, MIN_CLUSTER_SIZE
from src.features.clinical_code_object import ClinicalCodeObject, CodeSuggestion


class ClusterConsensus:
    """
    Weak supervision engine using cluster consensus for code suggestion.

    This class learns patterns of co-occurring codes within patient clusters
    and suggests codes that are commonly present in similar patients.
    """

    def __init__(
        self,
        consensus_threshold: float = CONSENSUS_THRESHOLD,
        min_cluster_size: int = MIN_CLUSTER_SIZE
    ):
        """
        Initialize cluster consensus model.

        Args:
            consensus_threshold: Minimum prevalence for consensus code (0-1)
            min_cluster_size: Minimum cluster size to compute consensus
        """
        self.consensus_threshold = consensus_threshold
        self.min_cluster_size = min_cluster_size

        # Cluster code statistics
        self.cluster_code_counts: Dict[int, Counter] = {}
        self.cluster_sizes: Dict[int, int] = {}
        self.cluster_profiles: Dict[int, Dict[str, float]] = {}

        self.is_fitted = False

    def fit(self, clinical_objects: List[ClinicalCodeObject]) -> 'ClusterConsensus':
        """
        Fit the consensus model on clinical objects with cluster assignments.

        Args:
            clinical_objects: List of ClinicalCodeObject with cluster_id set

        Returns:
            self (for method chaining)
        """
        print("\n" + "=" * 60)
        print("Fitting Cluster Consensus Model")
        print("=" * 60)

        # Group objects by cluster
        clusters: Dict[int, List[ClinicalCodeObject]] = defaultdict(list)

        for obj in clinical_objects:
            if obj.cluster_id is not None:
                clusters[obj.cluster_id].append(obj)

        print(f"Found {len(clusters)} clusters")

        # Compute code prevalence for each cluster
        for cluster_id, cluster_objs in tqdm(clusters.items(), desc="Computing consensus"):
            cluster_size = len(cluster_objs)
            self.cluster_sizes[cluster_id] = cluster_size

            # Count codes in cluster
            code_counter = Counter()

            for obj in cluster_objs:
                # Add all NLP-derived codes
                code_counter.update(obj.nlp_code_set)

            self.cluster_code_counts[cluster_id] = code_counter

            # Compute prevalence (proportion of claims with code)
            if cluster_size >= self.min_cluster_size:
                prevalences = {
                    code: count / cluster_size
                    for code, count in code_counter.items()
                }
                self.cluster_profiles[cluster_id] = prevalences
            else:
                self.cluster_profiles[cluster_id] = {}

        # Summary statistics
        total_consensus_codes = sum(
            sum(1 for prev in profile.values() if prev >= self.consensus_threshold)
            for profile in self.cluster_profiles.values()
        )

        print(f"\nCluster Statistics:")
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Clusters meeting min size: {len(self.cluster_profiles)}")
        print(f"  Total consensus codes: {total_consensus_codes}")
        print(f"  Avg cluster size: {np.mean(list(self.cluster_sizes.values())):.1f}")

        self.is_fitted = True
        return self

    def get_consensus_codes(self, cluster_id: int) -> Dict[str, float]:
        """
        Get consensus codes for a cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Dict mapping code to prevalence (only codes above threshold)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if cluster_id not in self.cluster_profiles:
            return {}

        profile = self.cluster_profiles[cluster_id]

        # Return only codes above consensus threshold
        return {
            code: prev
            for code, prev in profile.items()
            if prev >= self.consensus_threshold
        }

    def get_prevalence(self, cluster_id: int, code: str) -> float:
        """
        Get prevalence of a specific code in a cluster.

        Args:
            cluster_id: Cluster identifier
            code: Billing code

        Returns:
            Prevalence (0-1), or 0 if code not found
        """
        if cluster_id not in self.cluster_profiles:
            return 0.0

        return self.cluster_profiles[cluster_id].get(code, 0.0)

    def suggest(
        self,
        clinical_obj: ClinicalCodeObject,
        max_suggestions: int = 5
    ) -> List[CodeSuggestion]:
        """
        Suggest additional codes based on cluster consensus.

        Args:
            clinical_obj: Clinical object with cluster_id
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of CodeSuggestion objects
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if clinical_obj.cluster_id is None:
            return []

        # Get consensus codes for cluster
        consensus_codes = self.get_consensus_codes(clinical_obj.cluster_id)

        # Filter out codes already in NLP set
        existing_codes = clinical_obj.nlp_code_set
        missing_codes = {
            code: prev
            for code, prev in consensus_codes.items()
            if code not in existing_codes
        }

        # Sort by prevalence
        sorted_codes = sorted(
            missing_codes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_suggestions]

        # Create suggestions
        suggestions = []
        for code, prevalence in sorted_codes:
            # Determine code type (simplified)
            code_type = 'ICD-10' if code[0].isalpha() else 'HCPCS'

            suggestion = CodeSuggestion(
                code=code,
                code_type=code_type,
                description=f"Cluster consensus code (prevalence: {prevalence:.1%})",
                confidence=prevalence,
                source='cluster_consensus',
                rationale=f"Present in {prevalence:.1%} of similar claims in cluster {clinical_obj.cluster_id}"
            )
            suggestions.append(suggestion)

        return suggestions

    def batch_suggest(
        self,
        clinical_objects: List[ClinicalCodeObject],
        max_suggestions: int = 5
    ) -> List[ClinicalCodeObject]:
        """
        Add cluster consensus suggestions to multiple clinical objects.

        Args:
            clinical_objects: List of clinical objects
            max_suggestions: Maximum suggestions per claim

        Returns:
            Updated clinical objects with suggestions added
        """
        print(f"\nGenerating cluster consensus suggestions...")

        for obj in tqdm(clinical_objects, desc="Suggesting codes"):
            suggestions = self.suggest(obj, max_suggestions)

            for sugg in suggestions:
                obj.add_suggestion(sugg)

        total_suggestions = sum(
            len([s for s in obj.suggested_codes if s.source == 'cluster_consensus'])
            for obj in clinical_objects
        )

        print(f"Added {total_suggestions} cluster consensus suggestions")

        return clinical_objects

    def get_cluster_profile(self, cluster_id: int) -> pd.DataFrame:
        """
        Get detailed profile of codes in a cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            DataFrame with code, count, prevalence, is_consensus columns
        """
        if cluster_id not in self.cluster_code_counts:
            return pd.DataFrame()

        counts = self.cluster_code_counts[cluster_id]
        size = self.cluster_sizes[cluster_id]

        data = []
        for code, count in counts.most_common():
            prevalence = count / size
            is_consensus = prevalence >= self.consensus_threshold

            data.append({
                'code': code,
                'count': count,
                'prevalence': prevalence,
                'is_consensus': is_consensus
            })

        return pd.DataFrame(data)

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all clusters.

        Returns:
            DataFrame with cluster statistics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        data = []
        for cluster_id in sorted(self.cluster_sizes.keys()):
            size = self.cluster_sizes[cluster_id]
            num_codes = len(self.cluster_code_counts[cluster_id])
            consensus_codes = self.get_consensus_codes(cluster_id)
            num_consensus = len(consensus_codes)

            data.append({
                'cluster_id': cluster_id,
                'size': size,
                'total_codes': num_codes,
                'consensus_codes': num_consensus,
                'meets_min_size': size >= self.min_cluster_size
            })

        return pd.DataFrame(data)


def integrate_with_clustering(
    clinical_objects: List[ClinicalCodeObject],
    cluster_labels: np.ndarray,
    consensus_model: Optional[ClusterConsensus] = None
) -> Tuple[List[ClinicalCodeObject], ClusterConsensus]:
    """
    Helper function to integrate clustering results with consensus model.

    Args:
        clinical_objects: List of clinical objects
        cluster_labels: Array of cluster labels from sklearn
        consensus_model: Optional pre-initialized model

    Returns:
        Tuple of (updated clinical objects, fitted consensus model)
    """
    print("\n" + "=" * 60)
    print("Integrating Clustering with Consensus Model")
    print("=" * 60)

    # Assign cluster IDs
    for obj, label in zip(clinical_objects, cluster_labels):
        obj.cluster_id = int(label)

    # Initialize and fit consensus model
    if consensus_model is None:
        consensus_model = ClusterConsensus()

    consensus_model.fit(clinical_objects)

    # Generate suggestions
    consensus_model.batch_suggest(clinical_objects)

    return clinical_objects, consensus_model


def main():
    """Demo cluster consensus on sample data."""
    from src.features.clinical_code_object import ExtractedEntity

    print("\n" + "=" * 70)
    print(" " * 20 + "CLUSTER CONSENSUS DEMO")
    print("=" * 70 + "\n")

    # Create sample clinical objects
    clinical_objects = []

    # Cluster 0: Diabetes patients
    for i in range(20):
        obj = ClinicalCodeObject(
            claim_id=f"DM_{i:03d}",
            visit_notes="Diabetes follow-up",
            condition_text="Type 2 DM",
            cluster_id=0
        )

        # Most have E11.9 (diabetes)
        obj.add_diagnosis(ExtractedEntity(
            term="diabetes", code="E119", code_type="ICD-10",
            confidence=0.95, source_span="diabetes"
        ))

        # Many have I10 (hypertension)
        if i < 15:
            obj.add_diagnosis(ExtractedEntity(
                term="hypertension", code="I10", code_type="ICD-10",
                confidence=0.90, source_span="hypertension"
            ))

        clinical_objects.append(obj)

    # Cluster 1: Heart failure patients
    for i in range(15):
        obj = ClinicalCodeObject(
            claim_id=f"HF_{i:03d}",
            visit_notes="Heart failure follow-up",
            condition_text="CHF",
            cluster_id=1
        )

        # All have I50 (heart failure)
        obj.add_diagnosis(ExtractedEntity(
            term="heart failure", code="I50", code_type="ICD-10",
            confidence=0.92, source_span="heart failure"
        ))

        # Most have I10 (hypertension)
        if i < 12:
            obj.add_diagnosis(ExtractedEntity(
                term="hypertension", code="I10", code_type="ICD-10",
                confidence=0.90, source_span="hypertension"
            ))

        clinical_objects.append(obj)

    # Fit model
    model = ClusterConsensus(consensus_threshold=0.7)
    model.fit(clinical_objects)

    # Show cluster profiles
    print("\n" + "=" * 60)
    print("CLUSTER PROFILES")
    print("=" * 60)

    for cluster_id in [0, 1]:
        print(f"\nCluster {cluster_id}:")
        profile_df = model.get_cluster_profile(cluster_id)
        print(profile_df.to_string(index=False))

    # Test suggestion
    print("\n" + "=" * 60)
    print("TESTING SUGGESTIONS")
    print("=" * 60)

    # Create new diabetes patient without hypertension
    test_obj = ClinicalCodeObject(
        claim_id="TEST_001",
        visit_notes="Diabetes follow-up",
        condition_text="Type 2 DM",
        cluster_id=0
    )
    test_obj.add_diagnosis(ExtractedEntity(
        term="diabetes", code="E119", code_type="ICD-10",
        confidence=0.95, source_span="diabetes"
    ))

    suggestions = model.suggest(test_obj)
    print(f"\nSuggestions for new diabetes patient:")
    for sugg in suggestions:
        print(f"  {sugg.code}: {sugg.rationale}")


if __name__ == "__main__":
    main()
