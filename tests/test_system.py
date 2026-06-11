import unittest
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock heavy packages at the import level so tests can run without dependencies
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.http'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['datasets'] = MagicMock()

# Now import the modules and app safely
import app.search_engine
import app.synthesizer
from fastapi.testclient import TestClient

with patch("app.search_engine.SearchEngine.__init__", return_value=None), \
     patch("app.synthesizer.ReviewSynthesizer.__init__", return_value=None):
    from app.main import app



class TestSystemAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.search_engine")
    @patch("app.main.synthesizer")
    def test_health_check_healthy(self, mock_synthesizer, mock_search_engine):
        """Test the /health endpoint under healthy conditions."""
        # Mock search engine client collection check
        mock_search_engine.qdrant_client = MagicMock()
        mock_search_engine.qdrant_client.get_collections.return_value = MagicMock()

        # Mock LLM check (Ollama fallback)
        with patch("app.main.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            response = self.client.get("/health")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["qdrant_connected"])
            self.assertTrue(data["llm_connected"])

    @patch("app.main.search_engine")
    @patch("app.main.synthesizer")
    def test_health_check_unhealthy_db(self, mock_synthesizer, mock_search_engine):
        """Test the /health endpoint when the database is unreachable."""
        # Mock database failure
        mock_search_engine.qdrant_client = MagicMock()
        mock_search_engine.qdrant_client.get_collections.side_effect = Exception("DB Connection Refused")

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["status"], "unhealthy")
        self.assertFalse(data["qdrant_connected"])

    @patch("app.main.search_engine")
    @patch("app.main.synthesizer")
    def test_search_endpoint_success(self, mock_synthesizer, mock_search_engine):
        """Test the /search endpoint. Validates that Stage 1 and 2 run, and top result is synthesized."""
        # 1. Mock search engine results
        mock_search_engine.search.return_value = [
            {
                "parent_asin": "B001",
                "title": "Mock Product 1",
                "description": "Description 1",
                "reviews": ["Good review 1", "Bad review 1"],
                "re_ranked_score": 0.95
            },
            {
                "parent_asin": "B002",
                "title": "Mock Product 2",
                "description": "Description 2",
                "reviews": ["Good review 2"],
                "re_ranked_score": 0.82
            }
        ]

        # 2. Mock synthesizer review synthesis response (only called for first index)
        mock_synthesizer.synthesize_reviews = AsyncMock(return_value="- Pro: High quality.\n- Con: Overpriced.")

        # 3. Call the /search endpoint
        payload = {"query": "smart home gadgets", "top_k": 2}
        response = self.client.post("/search", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(len(data), 2)
        
        # Verify top result structure (has summary)
        self.assertEqual(data[0]["product_id"], "B001")
        self.assertEqual(data[0]["title"], "Mock Product 1")
        self.assertEqual(data[0]["re_ranked_score"], 0.95)
        self.assertEqual(data[0]["review_summary"], "- Pro: High quality.\n- Con: Overpriced.")

        # Verify second result structure (has NO summary)
        self.assertEqual(data[1]["product_id"], "B002")
        self.assertEqual(data[1]["title"], "Mock Product 2")
        self.assertEqual(data[1]["re_ranked_score"], 0.82)
        self.assertIsNone(data[1]["review_summary"])

    def test_search_endpoint_validation(self):
        """Test search endpoint query parameters validation (min lengths, limits)."""
        # Test empty query
        response = self.client.post("/search", json={"query": "", "top_k": 10})
        self.assertEqual(response.status_code, 422)

        # Test invalid top_k (less than 1)
        response = self.client.post("/search", json={"query": "test", "top_k": 0})
        self.assertEqual(response.status_code, 422)

        # Test invalid top_k (more than 100)
        response = self.client.post("/search", json={"query": "test", "top_k": 101})
        self.assertEqual(response.status_code, 422)
