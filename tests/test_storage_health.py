"""
Storage Health Check Tests
===========================
Unit tests for storage connectivity and health checks.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.cloud_cache import CloudModelCache, BlobStorage, get_model_with_cache


class TestCloudModelCache:
    """Test CloudModelCache functionality."""

    def test_initialization_without_tokens(self):
        """Test initialization when no cloud tokens available."""
        with patch.dict(os.environ, {}, clear=True):
            cache = CloudModelCache()
            assert cache.use_blob is False
            assert cache.use_kv is False

    def test_initialization_with_tokens(self):
        """Test initialization with cloud tokens."""
        with patch.dict(os.environ, {
            'BLOB_READ_WRITE_TOKEN': 'test_blob_token',
            'KV_REST_API_URL': 'https://test-kv.vercel.com'
        }):
            cache = CloudModelCache()
            assert cache.use_blob is True
            assert cache.use_kv is True

    def test_get_cache_key(self):
        """Test cache key generation."""
        cache = CloudModelCache()
        key1 = cache._get_cache_key("model_name")
        key2 = cache._get_cache_key("model_name")
        key3 = cache._get_cache_key("different_model")

        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key
        assert len(key1) == 32  # MD5 hash length

    def test_local_cache_fallback(self):
        """Test local filesystem cache fallback."""
        with patch.dict(os.environ, {}, clear=True):
            cache = CloudModelCache()

            # Should fall back to local storage
            metadata = {"version": "1.0", "model": "test"}
            success = cache.store_model_metadata("test_model", metadata)
            assert success is True

            # Retrieve from local cache
            retrieved = cache.get_model_metadata("test_model")
            assert retrieved is not None
            assert retrieved["version"] == "1.0"


class TestBlobStorage:
    """Test Vercel Blob storage functionality."""

    def test_initialization_without_token(self):
        """Test that BlobStorage requires token."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="BLOB_READ_WRITE_TOKEN"):
                BlobStorage()

    def test_initialization_with_token(self):
        """Test successful initialization with token."""
        with patch.dict(os.environ, {'BLOB_READ_WRITE_TOKEN': 'test_token'}):
            blob = BlobStorage()
            assert blob.token == 'test_token'

    @patch('requests.put')
    def test_put_success(self, mock_put):
        """Test successful blob upload."""
        mock_put.return_value = Mock(status_code=200)

        with patch.dict(os.environ, {'BLOB_READ_WRITE_TOKEN': 'test_token'}):
            blob = BlobStorage()
            success = blob.put("test_key", b"test_data")

            assert success is True
            mock_put.assert_called_once()

    @patch('requests.get')
    def test_get_success(self, mock_get):
        """Test successful blob download."""
        mock_get.return_value = Mock(status_code=200, content=b"test_data")

        with patch.dict(os.environ, {'BLOB_READ_WRITE_TOKEN': 'test_token'}):
            blob = BlobStorage()
            data = blob.get("test_key")

            assert data == b"test_data"
            mock_get.assert_called_once()


class TestDatabaseHealth:
    """Test database connectivity checks."""

    @pytest.mark.skipif(not os.getenv("VERCEL_POSTGRES_URL"),
                        reason="Requires VERCEL_POSTGRES_URL")
    def test_postgres_connection(self):
        """Test actual Postgres connection (integration test)."""
        from sqlalchemy import create_engine, text

        engine = create_engine(os.getenv("VERCEL_POSTGRES_URL"))

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1

    @pytest.mark.skipif(not os.getenv("VERCEL_POSTGRES_URL"),
                        reason="Requires VERCEL_POSTGRES_URL")
    def test_pgvector_extension(self):
        """Test pgvector extension availability."""
        from sqlalchemy import create_engine, text

        engine = create_engine(os.getenv("VERCEL_POSTGRES_URL"))

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """))
            assert result.fetchone()[0] is True


class TestClinicalCodeObject:
    """Test ClinicalCodeObject serialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from src.features.clinical_code_object import ClinicalCodeObject, ExtractedEntity

        obj = ClinicalCodeObject(claim_id="TEST-001")
        obj.extracted_diagnoses.append(ExtractedEntity(
            term="diabetes",
            code="E11.9",
            code_type="ICD-10",
            confidence=0.95,
            source_span="patient has diabetes"
        ))

        data = obj.to_dict()
        assert data["claim_id"] == "TEST-001"
        assert len(data["extracted_diagnoses"]) == 1
        assert data["extracted_diagnoses"][0]["code"] == "E11.9"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        from src.features.clinical_code_object import ClinicalCodeObject

        data = {
            "claim_id": "TEST-002",
            "extracted_diagnoses": [
                {
                    "term": "hypertension",
                    "code": "I10",
                    "code_type": "ICD-10",
                    "confidence": 0.92,
                    "source_span": "patient with hypertension"
                }
            ],
            "extracted_procedures": [],
            "suggested_codes": ["I10", "E11.9"],
            "flags": {},
            "missing_codes": [],
            "unsupported_codes": []
        }

        obj = ClinicalCodeObject.from_dict(data)
        assert obj.claim_id == "TEST-002"
        assert len(obj.extracted_diagnoses) == 1
        assert obj.extracted_diagnoses[0].code == "I10"
        assert len(obj.suggested_codes) == 2


class TestGapAnalyzer:
    """Test gap analysis functionality."""

    def test_flag_revenue_leakage(self):
        """Test revenue leakage detection."""
        from src.features.gap_analyzer import GapAnalyzer
        from src.features.clinical_code_object import ClinicalCodeObject, ExtractedEntity

        analyzer = GapAnalyzer()

        obj = ClinicalCodeObject(claim_id="TEST-003")
        obj.extracted_diagnoses.append(ExtractedEntity(
            term="diabetes",
            code="E11.9",
            code_type="ICD-10",
            confidence=0.95,
            source_span="patient has diabetes"
        ))

        billed_codes = []  # Nothing billed

        result = analyzer.analyze_claim(obj, billed_codes, confidence_threshold=0.90)

        assert result.claim_id == "TEST-003"
        assert len(result.flags) > 0
        assert any(f['type'] == 'revenue_leakage' for f in result.flags)
        assert obj.flags['potential_under_coding'] is True

    def test_flag_compliance_risk(self):
        """Test compliance risk detection."""
        from src.features.gap_analyzer import GapAnalyzer
        from src.features.clinical_code_object import ClinicalCodeObject

        analyzer = GapAnalyzer()

        obj = ClinicalCodeObject(claim_id="TEST-004")
        # No extracted diagnoses

        billed_codes = ["E11.9"]  # Code billed but not documented

        result = analyzer.analyze_claim(obj, billed_codes)

        assert any(f['type'] == 'compliance_risk' for f in result.flags)
        assert obj.flags['potential_over_coding'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
