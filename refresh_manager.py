"""
Refresh Manager for Medical Billing ML System

Manages the refresh/update cycle for embeddings, categories, and clusters.
Tracks what needs to be refreshed based on timestamp and configuration.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from config import ModelConfig


class RefreshManager:
    """Manages refresh cycles for embeddings and categories"""

    def __init__(self, db_url: Optional[str] = None, config: Optional[ModelConfig] = None, dry_run: bool = False):
        """
        Initialize the refresh manager

        Args:
            db_url: Database connection URL (uses VERCEL_POSTGRES_URL if not provided)
            config: ModelConfig instance (loads default if not provided)
            dry_run: If True, only report what would be refreshed without making changes
        """
        self.db_url = db_url or os.getenv('VERCEL_POSTGRES_URL')
        if not self.db_url:
            raise ValueError("Database URL must be provided or set via VERCEL_POSTGRES_URL")

        self.config = config or ModelConfig.load()
        self.dry_run = dry_run
        self.engine = create_engine(self.db_url)

    def find_stale_embeddings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find notes with stale embeddings that need to be refreshed

        Args:
            limit: Maximum number of stale embeddings to return

        Returns:
            List of dictionaries containing note information
        """
        refresh_hours = self.config.refresh.default_embedding_refresh_hours

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
                    embedding_model IS NOT NULL
                    AND (
                        last_embedded_at IS NULL
                        OR last_embedded_at < NOW() - INTERVAL '{refresh_hours} hours'
                    )
                ORDER BY last_embedded_at ASC NULLS FIRST
                LIMIT :limit
            """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            rows = result.fetchall()

            return [
                {
                    'note_id': row[0],
                    'claim_id': row[1],
                    'note_type': row[2],
                    'last_embedded_at': row[3],
                    'embedding_model': row[4],
                    'age_hours': float(row[5]) if row[5] else None
                }
                for row in rows
            ]

    def find_stale_categories(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find categories that need refresh based on their refresh interval

        Args:
            limit: Maximum number of categories to return

        Returns:
            List of dictionaries containing category information
        """
        query = text("""
                SELECT
                    category_id,
                    category_name,
                    last_refreshed_at,
                    refresh_interval_hours,
                    EXTRACT(EPOCH FROM (NOW() - last_refreshed_at))/3600 as age_hours
                FROM claim_categories
                WHERE
                    last_refreshed_at IS NULL
                    OR (
                        refresh_interval_hours IS NOT NULL
                        AND last_refreshed_at < NOW() - (refresh_interval_hours || ' hours')::INTERVAL
                    )
                ORDER BY last_refreshed_at ASC NULLS FIRST
                LIMIT :limit
            """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            rows = result.fetchall()

            return [
                {
                    'category_id': row[0],
                    'category_name': row[1],
                    'last_refreshed_at': row[2],
                    'refresh_interval_hours': row[3],
                    'age_hours': float(row[4]) if row[4] else None
                }
                for row in rows
            ]

    def get_refresh_stats(self) -> Dict[str, Any]:
        """
        Get statistics about refresh status across the system

        Returns:
            Dictionary with refresh statistics
        """
        stats_query = text("""
                SELECT
                    COUNT(*) as total_notes,
                    COUNT(embedding_model) as embedded_notes,
                    COUNT(CASE
                        WHEN last_embedded_at IS NOT NULL
                        AND last_embedded_at < NOW() - INTERVAL '12 hours'
                        THEN 1
                    END) as stale_embeddings,
                    AVG(EXTRACT(EPOCH FROM (NOW() - last_embedded_at))/3600) as avg_embedding_age_hours
                FROM clinical_notes
                WHERE embedding_model IS NOT NULL
            """)

        category_query = text("""
                SELECT
                    COUNT(*) as total_categories,
                    COUNT(last_refreshed_at) as refreshed_categories,
                    COUNT(CASE
                        WHEN last_refreshed_at IS NOT NULL
                        AND refresh_interval_hours IS NOT NULL
                        AND last_refreshed_at < NOW() - (refresh_interval_hours || ' hours')::INTERVAL
                        THEN 1
                    END) as stale_categories
                FROM claim_categories
            """)

        with self.engine.connect() as conn:
            # Get embedding stats
            result = conn.execute(stats_query)
            row = result.fetchone()
            embedding_stats = {
                'total_notes': row[0],
                'embedded_notes': row[1],
                'stale_embeddings': row[2],
                'avg_embedding_age_hours': float(row[3]) if row[3] else 0
            }

            # Get category stats
            result = conn.execute(category_query)
            row = result.fetchone()
            category_stats = {
                'total_categories': row[0],
                'refreshed_categories': row[1],
                'stale_categories': row[2]
            }

            return {
                'embeddings': embedding_stats,
                'categories': category_stats,
                'config': {
                    'embedding_refresh_hours': self.config.refresh.default_embedding_refresh_hours,
                    'category_refresh_hours': self.config.refresh.category_refresh_hours,
                    'auto_refresh_enabled': self.config.refresh.auto_refresh_enabled
                }
            }

    def update_embedding_timestamp(self, note_ids: List[int], embedding_model: str) -> int:
        """
        Update the embedding timestamp for specific notes

        Args:
            note_ids: List of note IDs to update
            embedding_model: The embedding model used

        Returns:
            Number of rows updated
        """
        if self.dry_run:
            return len(note_ids)

        if not note_ids:
            return 0

        update_query = text("""
                UPDATE clinical_notes
                SET
                    last_embedded_at = NOW(),
                    embedding_model = :model
                WHERE note_id = ANY(:note_ids)
            """)

        with self.engine.connect() as conn:
            result = conn.execute(
                update_query,
                {"model": embedding_model, "note_ids": note_ids}
            )
            conn.commit()
            return result.rowcount

    def update_category_timestamp(self, category_ids: List[int]) -> int:
        """
        Update the refresh timestamp for specific categories

        Args:
            category_ids: List of category IDs to update

        Returns:
            Number of rows updated
        """
        if self.dry_run:
            return len(category_ids)

        if not category_ids:
            return 0

        update_query = text("""
                UPDATE claim_categories
                SET last_refreshed_at = NOW()
                WHERE category_id = ANY(:category_ids)
            """)

        with self.engine.connect() as conn:
            result = conn.execute(update_query, {"category_ids": category_ids})
            conn.commit()
            return result.rowcount

    def get_refresh_config_rules(self) -> List[Dict[str, Any]]:
        """
        Get all refresh configuration rules from the database

        Returns:
            List of refresh configuration rules
        """
        query = text("""
                SELECT
                    config_name,
                    refresh_interval_hours,
                    enabled,
                    last_run_at,
                    description
                FROM embedding_refresh_config
                ORDER BY config_name
            """)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

            return [
                {
                    'config_name': row[0],
                    'refresh_interval_hours': row[1],
                    'enabled': row[2],
                    'last_run_at': row[3],
                    'description': row[4]
                }
                for row in rows
            ]

    def print_status(self):
        """Print a formatted status report"""
        print("=" * 70)
        print("🔄 REFRESH MANAGER")
        print("=" * 70)
        if self.dry_run:
            print("Mode: DRY RUN (no changes)")
        else:
            print("Mode: LIVE (will make changes)")
        print("=" * 70)
        print()

        # Get statistics
        stats = self.get_refresh_stats()

        print("📊 Embedding Status:")
        print(f"   Total notes: {stats['embeddings']['total_notes']:,}")
        print(f"   Embedded notes: {stats['embeddings']['embedded_notes']:,}")
        print(f"   Stale embeddings: {stats['embeddings']['stale_embeddings']:,}")
        print(f"   Average age: {stats['embeddings']['avg_embedding_age_hours']:.1f} hours")
        print()

        print("🏷️  Category Status:")
        print(f"   Total categories: {stats['categories']['total_categories']}")
        print(f"   Refreshed categories: {stats['categories']['refreshed_categories']}")
        print(f"   Stale categories: {stats['categories']['stale_categories']}")
        print()

        print("⚙️  Configuration:")
        print(f"   Embedding refresh interval: {stats['config']['embedding_refresh_hours']} hours")
        print(f"   Category refresh interval: {stats['config']['category_refresh_hours']} hours")
        print(f"   Auto-refresh: {'enabled' if stats['config']['auto_refresh_enabled'] else 'disabled'}")
        print()

        # Get refresh config rules
        rules = self.get_refresh_config_rules()
        if rules:
            print("📋 Refresh Rules:")
            for rule in rules:
                status = "✅ enabled" if rule['enabled'] else "❌ disabled"
                print(f"   {rule['config_name']}: {rule['refresh_interval_hours']}h ({status})")
            print()

        print("=" * 70)


# Convenience function
def get_refresh_manager(dry_run: bool = False) -> RefreshManager:
    """
    Get a RefreshManager instance

    Args:
        dry_run: If True, manager will not make any changes

    Returns:
        RefreshManager instance
    """
    return RefreshManager(dry_run=dry_run)


# Example usage
if __name__ == "__main__":
    manager = get_refresh_manager(dry_run=True)
    manager.print_status()

    print("\n🔍 Finding stale embeddings (top 5):")
    stale = manager.find_stale_embeddings(limit=5)
    for i, note in enumerate(stale, 1):
        age = f"{note['age_hours']:.1f}h" if note['age_hours'] else "never"
        print(f"   {i}. Note {note['note_id']} ({note['note_type']}): {age} old")

    print("\n🔍 Finding stale categories (top 5):")
    stale_cats = manager.find_stale_categories(limit=5)
    for i, cat in enumerate(stale_cats, 1):
        age = f"{cat['age_hours']:.1f}h" if cat['age_hours'] else "never"
        print(f"   {i}. {cat['category_name']}: {age} old")
