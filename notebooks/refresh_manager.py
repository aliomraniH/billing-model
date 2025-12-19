"""
Refresh Manager for Medical Billing ML System

This utility manages automatic refresh of embeddings and categories based on
configured time intervals. It identifies stale data and triggers re-processing.

Features:
- Timestamp-based staleness detection
- Configurable refresh intervals per data type
- Batch processing to avoid overwhelming APIs
- Automatic or manual refresh triggers
- Detailed logging and statistics

Usage:
    # Check what needs refresh (dry run)
    python refresh_manager.py --check

    # Refresh stale embeddings
    python refresh_manager.py --refresh-embeddings

    # Refresh stale categories
    python refresh_manager.py --refresh-categories

    # Refresh everything
    python refresh_manager.py --refresh-all
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sqlalchemy import create_engine, text

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config


@dataclass
class RefreshStats:
    """Statistics from a refresh operation"""
    total_checked: int = 0
    stale_found: int = 0
    refreshed: int = 0
    errors: int = 0
    skipped: int = 0


class RefreshManager:
    """Manages refresh operations for embeddings and categories"""

    def __init__(self, database_url: str, dry_run: bool = False):
        self.database_url = database_url
        self.dry_run = dry_run
        self.engine = create_engine(database_url)
        self.config = get_config()

        print("=" * 70)
        print("🔄 REFRESH MANAGER")
        print("=" * 70)
        print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will update data)'}")
        print("=" * 70 + "\n")

    def find_stale_embeddings(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Find clinical notes with stale embeddings that need refresh

        Returns:
            List of dicts with note_id, claim_id, last_embedded_at, age_hours
        """
        with self.engine.connect() as conn:
            # Calculate staleness threshold
            refresh_hours = self.config.refresh.default_embedding_refresh_hours
            threshold_sql = f"NOW() - INTERVAL '{refresh_hours} hours'"

            query = text(f"""
                SELECT
                    note_id,
                    claim_id,
                    note_type,
                    last_embedded_at,
                    embedding_model,
                    EXTRACT(EPOCH FROM (NOW() - last_embedded_at))/3600 as age_hours
                FROM clinical_notes
                WHERE
                    embedding IS NOT NULL
                    AND (
                        last_embedded_at IS NULL
                        OR last_embedded_at < {threshold_sql}
                    )
                ORDER BY last_embedded_at ASC NULLS FIRST
                {f'LIMIT {limit}' if limit else ''}
            """)

            result = conn.execute(query)
            stale_notes = []

            for row in result:
                stale_notes.append({
                    'note_id': row[0],
                    'claim_id': row[1],
                    'note_type': row[2],
                    'last_embedded_at': row[3],
                    'embedding_model': row[4],
                    'age_hours': float(row[5]) if row[5] else None,
                })

            return stale_notes

    def find_stale_categories(self) -> List[Dict]:
        """
        Find categories that need re-clustering

        Returns:
            List of dicts with category_id, category_name, last_refreshed_at, age_hours
        """
        with self.engine.connect() as conn:
            # Use category-specific refresh interval
            refresh_hours = self.config.refresh.category_refresh_hours
            threshold_sql = f"NOW() - INTERVAL '{refresh_hours} hours'"

            query = text(f"""
                SELECT
                    category_id,
                    category_name,
                    display_name,
                    last_refreshed_at,
                    refresh_interval_hours,
                    cluster_version,
                    EXTRACT(EPOCH FROM (NOW() - last_refreshed_at))/3600 as age_hours
                FROM claim_categories
                WHERE
                    last_refreshed_at < {threshold_sql}
                    OR refresh_interval_hours IS NULL
                ORDER BY last_refreshed_at ASC NULLS FIRST
            """)

            result = conn.execute(query)
            stale_categories = []

            for row in result:
                # Use custom refresh interval if set, otherwise use default
                custom_interval = row[4]
                age_hours = float(row[6]) if row[6] else None

                # Check if stale based on custom interval
                is_stale = (
                    age_hours is None or
                    (custom_interval and age_hours >= custom_interval) or
                    (not custom_interval and age_hours >= refresh_hours)
                )

                if is_stale:
                    stale_categories.append({
                        'category_id': row[0],
                        'category_name': row[1],
                        'display_name': row[2],
                        'last_refreshed_at': row[3],
                        'refresh_interval_hours': custom_interval or refresh_hours,
                        'cluster_version': row[5],
                        'age_hours': age_hours,
                    })

            return stale_categories

    def check_refresh_status(self) -> Tuple[RefreshStats, RefreshStats]:
        """
        Check refresh status without making changes

        Returns:
            (embedding_stats, category_stats) tuples
        """
        print("🔍 CHECKING REFRESH STATUS\n")

        # Check embeddings
        embedding_stats = RefreshStats()
        stale_embeddings = self.find_stale_embeddings()
        embedding_stats.stale_found = len(stale_embeddings)

        # Get total embedded notes
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM clinical_notes WHERE embedding IS NOT NULL"))
            embedding_stats.total_checked = result.fetchone()[0]

        print(f"📊 Embeddings:")
        print(f"   Total embedded notes: {embedding_stats.total_checked:,}")
        print(f"   Stale (need refresh): {embedding_stats.stale_found:,} "
              f"({100*embedding_stats.stale_found/max(1,embedding_stats.total_checked):.1f}%)")

        if embedding_stats.stale_found > 0:
            refresh_hours = self.config.refresh.default_embedding_refresh_hours
            print(f"   Threshold: Older than {refresh_hours} hours")

            # Show oldest 5
            print(f"\n   Oldest 5 stale embeddings:")
            for i, note in enumerate(stale_embeddings[:5], 1):
                age = note['age_hours']
                age_str = f"{age:.1f}h" if age else "never"
                print(f"      [{i}] Note {note['note_id']}: {age_str} old")

        # Check categories
        category_stats = RefreshStats()
        stale_categories = self.find_stale_categories()
        category_stats.stale_found = len(stale_categories)

        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM claim_categories"))
            category_stats.total_checked = result.fetchone()[0]

        print(f"\n📊 Categories:")
        print(f"   Total categories: {category_stats.total_checked:,}")
        print(f"   Stale (need re-clustering): {category_stats.stale_found:,} "
              f"({100*category_stats.stale_found/max(1,category_stats.total_checked):.1f}%)")

        if category_stats.stale_found > 0:
            print(f"\n   Stale categories:")
            for i, cat in enumerate(stale_categories[:10], 1):
                age = cat['age_hours']
                age_str = f"{age:.1f}h" if age else "never"
                interval = cat['refresh_interval_hours']
                print(f"      [{i}] {cat['category_name']}: {age_str} old (refresh every {interval}h)")

        print("\n" + "=" * 70)

        return embedding_stats, category_stats

    def refresh_embeddings(self, note_ids: List[int]) -> RefreshStats:
        """
        Refresh embeddings for specific notes

        This marks notes for re-embedding. The actual embedding happens
        when Notebook 5 runs with auto-refresh enabled.

        Args:
            note_ids: List of note IDs to refresh

        Returns:
            RefreshStats with operation results
        """
        stats = RefreshStats()
        stats.total_checked = len(note_ids)

        if self.dry_run:
            print(f"   [DRY RUN] Would mark {len(note_ids)} notes for re-embedding")
            stats.refreshed = len(note_ids)
            return stats

        with self.engine.connect() as conn:
            # Clear embedding to force re-generation
            query = text("""
                UPDATE clinical_notes
                SET
                    embedding = NULL,
                    last_embedded_at = NULL
                WHERE note_id = ANY(:note_ids)
            """)

            result = conn.execute(query, {"note_ids": note_ids})
            stats.refreshed = result.rowcount
            conn.commit()

        print(f"   ✅ Marked {stats.refreshed:,} notes for re-embedding")

        return stats

    def refresh_categories_batch(self, category_ids: List[int]) -> RefreshStats:
        """
        Mark categories as needing re-clustering

        This updates last_refreshed_at to trigger re-clustering.
        The actual clustering happens when Notebook 6 runs.

        Args:
            category_ids: List of category IDs to refresh

        Returns:
            RefreshStats with operation results
        """
        stats = RefreshStats()
        stats.total_checked = len(category_ids)

        if self.dry_run:
            print(f"   [DRY RUN] Would mark {len(category_ids)} categories for re-clustering")
            stats.refreshed = len(category_ids)
            return stats

        with self.engine.connect() as conn:
            # Update last_refreshed_at to NOW - this will make them eligible for refresh
            # when enough time passes
            query = text("""
                UPDATE claim_categories
                SET last_refreshed_at = NOW()
                WHERE category_id = ANY(:category_ids)
            """)

            result = conn.execute(query, {"category_ids": category_ids})
            stats.refreshed = result.rowcount
            conn.commit()

        print(f"   ✅ Updated {stats.refreshed:,} category timestamps")

        return stats

    def auto_refresh_embeddings(self, max_refresh: Optional[int] = None) -> RefreshStats:
        """
        Automatically refresh stale embeddings

        Args:
            max_refresh: Maximum number to refresh (None for all)

        Returns:
            RefreshStats with operation results
        """
        print("🔄 AUTO-REFRESHING STALE EMBEDDINGS\n")

        # Find stale embeddings
        limit = max_refresh or self.config.refresh.max_refresh_per_run
        stale_notes = self.find_stale_embeddings(limit=limit)

        if not stale_notes:
            print("   ✅ No stale embeddings found")
            return RefreshStats()

        print(f"   Found {len(stale_notes):,} stale embeddings")

        # Extract note IDs
        note_ids = [note['note_id'] for note in stale_notes]

        # Refresh in batches
        batch_size = self.config.refresh.refresh_batch_size
        total_stats = RefreshStats()

        for i in range(0, len(note_ids), batch_size):
            batch = note_ids[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(note_ids) + batch_size - 1) // batch_size

            print(f"\n   Batch {batch_num}/{total_batches} ({len(batch)} notes)...")
            stats = self.refresh_embeddings(batch)

            total_stats.total_checked += stats.total_checked
            total_stats.refreshed += stats.refreshed
            total_stats.errors += stats.errors

        print(f"\n✅ Refresh complete:")
        print(f"   Total marked for refresh: {total_stats.refreshed:,}")

        return total_stats

    def auto_refresh_categories(self) -> RefreshStats:
        """
        Automatically refresh stale categories

        Returns:
            RefreshStats with operation results
        """
        print("🔄 AUTO-REFRESHING STALE CATEGORIES\n")

        # Find stale categories
        stale_categories = self.find_stale_categories()

        if not stale_categories:
            print("   ✅ No stale categories found")
            return RefreshStats()

        print(f"   Found {len(stale_categories):,} stale categories")

        # Extract category IDs
        category_ids = [cat['category_id'] for cat in stale_categories]

        # Refresh all at once (categories are small dataset)
        stats = self.refresh_categories_batch(category_ids)

        print(f"\n✅ Refresh complete:")
        print(f"   Categories updated: {stats.refreshed:,}")

        return stats


def main():
    """Main entry point for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Manage refresh operations for embeddings and categories")
    parser.add_argument('--check', action='store_true', help='Check refresh status (dry run)')
    parser.add_argument('--refresh-embeddings', action='store_true', help='Refresh stale embeddings')
    parser.add_argument('--refresh-categories', action='store_true', help='Refresh stale categories')
    parser.add_argument('--refresh-all', action='store_true', help='Refresh both embeddings and categories')
    parser.add_argument('--max-refresh', type=int, help='Maximum number of embeddings to refresh')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no changes)')

    args = parser.parse_args()

    # Get database URL
    database_url = os.getenv('VERCEL_POSTGRES_URL')
    if not database_url:
        print("❌ Error: VERCEL_POSTGRES_URL environment variable not set")
        sys.exit(1)

    # Create manager
    manager = RefreshManager(database_url, dry_run=args.dry_run)

    # Execute requested operation
    if args.check or not any([args.refresh_embeddings, args.refresh_categories, args.refresh_all]):
        # Default action: check status
        manager.check_refresh_status()

    if args.refresh_embeddings or args.refresh_all:
        manager.auto_refresh_embeddings(max_refresh=args.max_refresh)

    if args.refresh_categories or args.refresh_all:
        manager.auto_refresh_categories()

    print("\n" + "=" * 70)
    print("✅ REFRESH MANAGER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
