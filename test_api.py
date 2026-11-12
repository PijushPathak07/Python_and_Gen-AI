import tempfile
import unittest
from unittest.mock import patch

from config import Config

# Use an isolated temporary SQLite database for deterministic offline testing
TEMP_DB = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
TEMP_DB.close()
Config.DATABASE_PATH = TEMP_DB.name

from app import app  # noqa: E402
from database import init_db  # noqa: E402


MOCK_ANALYSIS = {
    "company_name": "Reliance Industries",
    "ticker": "RELIANCE",
    "quarter": "Q3 FY26",
    "financial_performance": "Revenue grew 14% YoY with record EBITDA margins.",
    "market_dynamics": "Strong domestic retail and telecom subscriber additions.",
    "expansion_plans": "Scaling 5G enterprise solutions and new energy giga-factories.",
    "environmental_risks": "Transitioning toward 100 GW renewable energy target.",
    "regulatory_or_policy_changes": "Compliant with updated telecom spectrum norms.",
    "overall_sentiment": "Bullish",
    "confidence_score": 88,
    "risk_score": 24,
    "radar_scores": {
        "revenue_growth": 85,
        "profitability": 82,
        "market_position": 90,
        "expansion_momentum": 88,
        "regulatory_safety": 80,
        "esg_sustainability": 76,
    },
    "key_kpis": [
        {"metric": "Revenue Growth", "value": "14% YoY", "impact": "Positive"}
    ],
    "executive_takeaways": ["Strong retail expansion", "Record telecom ARPU"],
    "nlp_metrics": {
        "word_count": 15,
        "estimated_reading_minutes": 1,
        "financial_keyword_hits": {"revenue": 1, "ebitda": 1},
        "total_financial_signals": 2,
    },
}


class FinSightAPITestSuite(unittest.TestCase):
    def setUp(self):
        init_db(Config.DATABASE_PATH)
        self.client = app.test_client()
        self.client.testing = True

    def test_1_health_endpoint(self):
        """Verify /health returns 200 OK and service metadata."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("cached_reports_count", data)

    def test_2_dashboard_home_renders(self):
        """Verify GET / renders the FinSight AI HTML interface."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"FinSight AI", res.data)

    def test_3_missing_fields_returns_400(self):
        """Verify missing transcript_text returns 400 Bad Request via Pydantic."""
        res = self.client.post(
            "/earnings_transcript_summary", json={"company_name": "Reliance"}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Missing required field", res.get_json()["error"])

    def test_4_empty_transcript_returns_400(self):
        """Verify whitespace-only transcript is rejected with 400 via Pydantic validator."""
        res = self.client.post(
            "/earnings_transcript_summary",
            json={"company_name": "Reliance", "transcript_text": "    "},
        )
        self.assertEqual(res.status_code, 400)

    def test_5_token_limit_exceeded_returns_400(self):
        """Verify transcripts exceeding Config.MAX_TOKENS trigger Pydantic validation error."""
        oversized_text = "A" * (Config.MAX_TOKENS + 50)
        res = self.client.post(
            "/earnings_transcript_summary",
            json={"company_name": "Reliance", "transcript_text": oversized_text},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("exceeds maximum length", res.get_json()["error"])

    @patch("app.analyze_earnings_transcript")
    def test_6_analyze_and_sha256_cache_hit(self, mock_analyze):
        """Verify first request calls analyzer and second identical request hits SQLite cache."""
        mock_analyze.return_value = dict(MOCK_ANALYSIS)
        payload = {
            "company_name": "Reliance Industries",
            "ticker": "RELIANCE",
            "quarter": "Q3 FY26",
            "transcript_text": "Revenue and EBITDA grew strongly in Q3 FY26.",
        }

        res1 = self.client.post("/earnings_transcript_summary", json=payload)
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()
        self.assertFalse(data1["cached"])
        self.assertEqual(mock_analyze.call_count, 1)

        res2 = self.client.post("/earnings_transcript_summary", json=payload)
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        self.assertTrue(data2["cached"])
        self.assertEqual(mock_analyze.call_count, 1)

    @patch("app.analyze_earnings_transcript")
    def test_7_history_and_compare_endpoints(self, mock_analyze):
        """Verify /history and /compare endpoints work across saved reports."""
        mock_analyze.return_value = dict(MOCK_ANALYSIS)

        r1 = self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Company Alpha",
                "transcript_text": "Alpha transcript unique text one.",
            },
        ).get_json()

        r2 = self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Company Beta",
                "transcript_text": "Beta transcript unique text two.",
            },
        ).get_json()

        hist = self.client.get("/history").get_json()
        self.assertGreaterEqual(hist["count"], 2)

        comp = self.client.get(f"/compare?id1={r1['id']}&id2={r2['id']}")
        self.assertEqual(comp.status_code, 200)
        self.assertIn("radar_score_deltas", comp.get_json())

    @patch("app.analyze_earnings_transcript")
    def test_8_rag_full_text_search_endpoint(self, mock_analyze):
        """Verify GET /search retrieves ranked context snippets across transcripts."""
        mock_analyze.return_value = dict(MOCK_ANALYSIS)
        self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Apex Cloud",
                "ticker": "APEX",
                "quarter": "Q4 FY26",
                "transcript_text": "We are investing $150M in sovereign GPU inference clusters and carbon tax credits.",
            },
        )

        # Missing query parameter -> 400
        bad_res = self.client.get("/search")
        self.assertEqual(bad_res.status_code, 400)

        # Valid RAG search query -> 200 with context snippet
        search_res = self.client.get("/search?q=sovereign GPU")
        self.assertEqual(search_res.status_code, 200)
        payload = search_res.get_json()
        self.assertGreaterEqual(payload["count"], 1)
        self.assertIn("sovereign GPU", payload["results"][0]["matched_snippets"][0])

    def test_9_openapi_and_swagger_docs(self):
        """Verify /openapi.json and /apidocs endpoints serve valid API documentation."""
        spec_res = self.client.get("/openapi.json")
        self.assertEqual(spec_res.status_code, 200)
        spec_data = spec_res.get_json()
        self.assertEqual(spec_data["openapi"], "3.0.3")
        self.assertIn("/earnings_transcript_summary", spec_data["paths"])

        docs_res = self.client.get("/apidocs")
        self.assertEqual(docs_res.status_code, 200)
        self.assertIn(b"SwaggerUIBundle", docs_res.data)


if __name__ == "__main__":
    unittest.main()