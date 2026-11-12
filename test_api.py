import tempfile
import unittest
from unittest.mock import MagicMock, patch

from config import Config

# Use an isolated temporary SQLite database for deterministic offline testing
TEMP_DB = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
TEMP_DB.close()
Config.DATABASE_PATH = TEMP_DB.name

import app as app_module  # noqa: E402
from app import app, reset_rate_limiter  # noqa: E402
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
        reset_rate_limiter()
        app_module.RATE_LIMIT_MAX_REQUESTS = 60
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

    @patch("app.analyze_earnings_transcript")
    def test_10_concurrent_batch_analyze_endpoint(self, mock_analyze):
        """Verify POST /batch_analyze processes multiple transcripts concurrently."""
        mock_analyze.return_value = dict(MOCK_ANALYSIS)
        batch_payload = {
            "items": [
                {
                    "company_name": "Batch Corp One",
                    "ticker": "BC1",
                    "quarter": "Q1 FY26",
                    "transcript_text": "Batch Corp One delivered 18% revenue growth.",
                },
                {
                    "company_name": "Batch Corp Two",
                    "ticker": "BC2",
                    "quarter": "Q1 FY26",
                    "transcript_text": "Batch Corp Two expanded operating margins by 200 bps.",
                },
            ]
        }
        res = self.client.post("/batch_analyze", json=batch_payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["batch_count"], 2)
        self.assertEqual(len(data["results"]), 2)

    @patch("app.analyze_earnings_transcript")
    def test_11_export_markdown_and_csv_endpoints(self, mock_analyze):
        """Verify GET /export/<id> exports both Markdown (.md) and CSV (.csv) institutional reports."""
        mock_analyze.return_value = dict(MOCK_ANALYSIS)
        created = self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Exportable Tech",
                "transcript_text": "Exportable Tech achieved strong free cash flow.",
            },
        ).get_json()

        md_res = self.client.get(f"/export/{created['id']}?format=md")
        self.assertEqual(md_res.status_code, 200)
        self.assertIn("text/markdown", md_res.content_type)
        self.assertIn(b"# Exportable Tech", md_res.data)

        csv_res = self.client.get(f"/export/{created['id']}?format=csv")
        self.assertEqual(csv_res.status_code, 200)
        self.assertIn("text/csv", csv_res.content_type)
        self.assertIn(b"Overall Sentiment", csv_res.data)

    def test_12_system_telemetry_metrics_endpoint(self):
        """Verify GET /metrics returns aggregate system telemetry and token savings."""
        res = self.client.get("/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("total_reports_cached", data)
        self.assertIn("sentiment_breakdown", data)
        self.assertIn("estimated_api_tokens_saved", data)

    @patch("app.analyze_earnings_transcript")
    def test_13_grounded_qa_ask_endpoint(self, mock_analyze):
        """Verify POST /ask/<id> returns grounded answers with verbatim quote citations."""
        mock_analyze.return_value = dict(MOCK_ANALYSIS)
        created = self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Citation Corp",
                "ticker": "CITE",
                "quarter": "Q3 FY26",
                "transcript_text": (
                    "We delivered record quarterly revenue of $1.42 billion, up 24% YoY. "
                    "Operating margin expanded by 320 basis points to 28.4%."
                ),
            },
        ).get_json()

        # Ask a specific question about revenue and margin
        qa_res = self.client.post(
            f"/ask/{created['id']}",
            json={"question": "What was the quarterly revenue and operating margin?"},
        )
        self.assertEqual(qa_res.status_code, 200)
        qa_data = qa_res.get_json()
        self.assertEqual(qa_data["company_name"], "Citation Corp")
        self.assertIn("$1.42 billion", qa_data["answer"])
        self.assertGreaterEqual(len(qa_data["citations"]), 1)

        # Verify 404 for non-existent report ID
        not_found_res = self.client.post(
            "/ask/99999", json={"question": "What was the revenue?"}
        )
        self.assertEqual(not_found_res.status_code, 404)

    @patch("app.analyze_earnings_transcript")
    def test_14_forensic_red_flag_audit_endpoint(self, mock_analyze):
        """Verify GET /forensics/<id> detects hedging ratios and forensic risk signals."""
        mock_analyze.return_value = dict(MOCK_ANALYSIS)
        created = self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Forensic Corp",
                "ticker": "FRNS",
                "quarter": "Q4 FY26",
                "transcript_text": (
                    "We delivered record revenue growth. However, mid-market budgets "
                    "remain scrutinized due to macroeconomic headwinds and non-gaap restructuring."
                ),
            },
        ).get_json()

        res = self.client.get(f"/forensics/{created['id']}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["company_name"], "Forensic Corp")
        self.assertIn("transparency_score", data)
        self.assertGreaterEqual(data["hedging_word_count"], 1)
        self.assertGreaterEqual(len(data["red_flags"]), 1)

    def test_15_rate_limiting_and_latency_headers_middleware(self):
        """Verify X-Process-Time-Ms headers and 429 Too Many Requests enforcement."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertIn("X-Process-Time-Ms", res.headers)
        self.assertIn("X-RateLimit-Limit", res.headers)
        self.assertIn("X-RateLimit-Remaining", res.headers)

        # Temporarily lower threshold to 2 requests to verify 429 enforcement
        app_module.RATE_LIMIT_MAX_REQUESTS = 2
        reset_rate_limiter()
        self.client.get("/health")
        self.client.get("/health")
        blocked = self.client.get("/health")
        self.assertEqual(blocked.status_code, 429)
        self.assertIn("Rate limit exceeded", blocked.get_json()["error"])

    @patch("app.analyze_earnings_transcript")
    def test_16_sector_peer_benchmark_matrix_endpoint(self, mock_analyze):
        """Verify GET /benchmark ranks multiple peers, computes composite scores, and returns sector averages."""
        peer_a = dict(MOCK_ANALYSIS)
        peer_a["risk_score"] = 20
        peer_b = dict(MOCK_ANALYSIS)
        peer_b["risk_score"] = 45
        mock_analyze.side_effect = [peer_a, peer_b]

        r1 = self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Alpha Semiconductor",
                "ticker": "ALPH",
                "quarter": "Q3 FY26",
                "transcript_text": "Alpha Semiconductor delivered record AI chip revenue.",
            },
        ).get_json()

        r2 = self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Beta Cloud Corp",
                "ticker": "BETA",
                "quarter": "Q3 FY26",
                "transcript_text": "Beta Cloud Corp experienced moderate enterprise growth.",
            },
        ).get_json()

        res = self.client.get(f"/benchmark?ids={r1['id']},{r2['id']}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["peer_count"], 2)
        self.assertEqual(data["sector_leader"], "Alpha Semiconductor")
        self.assertEqual(data["lowest_risk_peer"], "Alpha Semiconductor")
        self.assertIn("composite_score", data["sector_averages"])
        self.assertEqual(data["peers"][0]["rank"], 1)

    @patch("app.analyze_earnings_transcript")
    def test_17_guidance_credibility_tracker_endpoint(self, mock_analyze):
        """Verify GET /credibility audits prior-quarter promises against current-quarter reality."""
        mock_analyze.return_value = dict(MOCK_ANALYSIS)

        q2_res = self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Apex Cloud Systems",
                "ticker": "APEX",
                "quarter": "Q2 FY26",
                "transcript_text": (
                    "We plan to launch two sovereign cloud regions in Frankfurt and "
                    "expect revenue growth to exceed 20% next quarter."
                ),
            },
        ).get_json()

        q3_res = self.client.post(
            "/earnings_transcript_summary",
            json={
                "company_name": "Apex Cloud Systems",
                "ticker": "APEX",
                "quarter": "Q3 FY26",
                "transcript_text": (
                    "We delivered 24% revenue growth and completed the launch of our "
                    "Frankfurt sovereign cloud region ahead of schedule."
                ),
            },
        ).get_json()

        # Verify 400 when parameters are missing
        bad_res = self.client.get("/credibility")
        self.assertEqual(bad_res.status_code, 400)

        # Verify 200 and structured promise audit
        cred_res = self.client.get(
            f"/credibility?prev_id={q2_res['id']}&curr_id={q3_res['id']}"
        )
        self.assertEqual(cred_res.status_code, 200)
        data = cred_res.get_json()
        self.assertEqual(data["company_name"], "Apex Cloud Systems")
        self.assertGreaterEqual(data["credibility_score"], 75)
        self.assertIn(
            data["execution_grade"],
            ["High Credibility", "Moderate Execution", "Execution Deficit"],
        )
        self.assertGreaterEqual(len(data["promises_audited"]), 1)
        self.assertEqual(data["promises_audited"][0]["status"], "Delivered")

    @patch("sec_connector.requests.get")
    def test_18_live_sec_edgar_connector_endpoint(self, mock_requests_get):
        """Verify GET /sec_filings/<ticker> queries SEC EDGAR endpoints and parses 10-K/10-Q filings."""
        mock_tickers_resp = MagicMock()
        mock_tickers_resp.raise_for_status.return_value = None
        mock_tickers_resp.json.return_value = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        }

        mock_submissions_resp = MagicMock()
        mock_submissions_resp.raise_for_status.return_value = None
        mock_submissions_resp.json.return_value = {
            "name": "Apple Inc.",
            "sic": "3571",
            "sicDescription": "Electronic Computers",
            "exchanges": ["Nasdaq"],
            "fiscalYearEnd": "0928",
            "stateOfIncorporation": "CA",
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "4"],
                    "filingDate": ["2025-11-01", "2025-08-01", "2025-07-15"],
                    "reportDate": ["2025-09-28", "2025-06-28", "2025-07-15"],
                    "accessionNumber": [
                        "0000320193-25-000106",
                        "0000320193-25-000077",
                        "0000320193-25-000050",
                    ],
                    "primaryDocument": [
                        "aapl-20250928.htm",
                        "aapl-20250628.htm",
                        "form4.xml",
                    ],
                }
            },
        }

        mock_requests_get.side_effect = [mock_tickers_resp, mock_submissions_resp]

        res = self.client.get("/sec_filings/AAPL")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertEqual(data["cik"], "0000320193")
        self.assertEqual(data["company_name"], "Apple Inc.")
        self.assertEqual(len(data["recent_filings"]), 2)
        self.assertEqual(data["recent_filings"][0]["form"], "10-K")
        self.assertIn("https://www.sec.gov/Archives/edgar/data/", data["recent_filings"][0]["document_url"])


if __name__ == "__main__":
    unittest.main()