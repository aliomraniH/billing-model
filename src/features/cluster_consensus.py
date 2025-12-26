"""
Cluster Consensus: Weak supervision algorithm for code suggestion.
Uses collective coding patterns within similar patient clusters to suggest codes.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Set
from collections import Counter
from config.settings import nlp_config


class ClusterConsensus:
    """
    Implements weak supervision through cluster-based code consensus.

    Algorithm:
    1. Group claims by cluster (from existing clustering in notebooks 05-06)
    2. Calculate code frequency within each cluster
    3. Codes appearing in >70% of cluster members = "Consensus Codes"
    4. Suggest missing consensus codes to individual claims
    """

    def __init__(self, consensus_threshold: float = None):
        """
        Initialize cluster consensus analyzer.

        Args:
            consensus_threshold: Minimum prevalence for consensus (default: 0.70)
        """
        self.threshold = consensus_threshold or nlp_config.consensus_threshold
        self.cluster_profiles: Dict[int, Dict] = {}

    def build_cluster_profiles(
        self,
        df: pd.DataFrame,
        cluster_column: str = 'cluster_id',
        codes_column: str = 'extracted_codes'
    ) -> Dict[int, Dict]:
        """
        Build code frequency profiles for each cluster.

        Args:
            df: DataFrame with cluster assignments and extracted codes
            cluster_column: Column containing cluster IDs
            codes_column: Column containing list of codes (or JSON string)

        Returns:
            Dict mapping cluster_id to profile dict
        """
        print(f"📊 Building cluster profiles (threshold: {self.threshold:.0%})...")

        for cluster_id in df[cluster_column].unique():
            cluster_df = df[df[cluster_column] == cluster_id]
            cluster_size = len(cluster_df)

            # Aggregate all codes in cluster
            all_codes = []
            for codes in cluster_df[codes_column]:
                if isinstance(codes, str):
                    import json
                    try:
                        codes = json.loads(codes)
                    except:
                        codes = [c.strip() for c in codes.split(',')]
                if isinstance(codes, list):
                    all_codes.extend(codes)

            # Calculate frequency
            code_counts = Counter(all_codes)

            # Identify consensus codes (above threshold)
            consensus_codes = {
                code: count / cluster_size
                for code, count in code_counts.items()
                if count / cluster_size >= self.threshold
            }

            self.cluster_profiles[cluster_id] = {
                'size': cluster_size,
                'total_codes': len(code_counts),
                'consensus_codes': consensus_codes,
                'all_code_frequencies': {
                    code: count / cluster_size
                    for code, count in code_counts.items()
                }
            }

        print(f"✅ Built profiles for {len(self.cluster_profiles)} clusters")
        return self.cluster_profiles

    def get_suggestions(
        self,
        claim_codes: List[str],
        cluster_id: int
    ) -> List[Dict]:
        """
        Get code suggestions for a claim based on cluster consensus.

        Args:
            claim_codes: Codes already present/extracted for the claim
            cluster_id: Cluster this claim belongs to

        Returns:
            List of suggestion dicts with code and rationale
        """
        if cluster_id not in self.cluster_profiles:
            return []

        profile = self.cluster_profiles[cluster_id]
        claim_code_set = set(claim_codes)

        suggestions = []
        for code, prevalence in profile['consensus_codes'].items():
            if code not in claim_code_set:
                suggestions.append({
                    'code': code,
                    'prevalence': prevalence,
                    'cluster_size': profile['size'],
                    'rationale': f"Present in {prevalence:.0%} of similar claims in cluster {cluster_id}"
                })

        # Sort by prevalence (highest first)
        suggestions.sort(key=lambda x: x['prevalence'], reverse=True)
        return suggestions

    def apply_suggestions(
        self,
        clinical_objects: List['ClinicalCodeObject'],
        df: pd.DataFrame,
        cluster_column: str = 'cluster_id'
    ) -> List['ClinicalCodeObject']:
        """
        Apply consensus suggestions to ClinicalCodeObjects.

        Args:
            clinical_objects: List of ClinicalCodeObject instances
            df: DataFrame with cluster assignments (must have matching claim_id)
            cluster_column: Column containing cluster IDs

        Returns:
            Updated ClinicalCodeObjects with suggestions
        """
        # Create claim_id to cluster_id mapping
        claim_to_cluster = df.set_index('claim_id')[cluster_column].to_dict()

        updated_count = 0
        for obj in clinical_objects:
            cluster_id = claim_to_cluster.get(obj.claim_id)
            if cluster_id is None:
                continue

            obj.cluster_id = cluster_id

            # Get current codes from the object
            current_codes = [e.code for e in obj.billable_diagnoses]
            current_codes += [e.code for e in obj.billable_procedures]

            # Get suggestions
            suggestions = self.get_suggestions(current_codes, cluster_id)

            if suggestions:
                obj.suggested_codes = [s['code'] for s in suggestions[:5]]  # Top 5
                updated_count += 1

        print(f"✅ Applied suggestions to {updated_count} claims")
        return clinical_objects
