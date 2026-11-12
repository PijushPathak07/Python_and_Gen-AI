import csv
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from pydantic import ValidationError

from analyzer import analyze_earnings_transcript
from database import (
    compute_transcript_hash,
    delete_report_by_id,
    get_all_reports,
    get_cached_report,
    get_report_by_id,
    get_system_telemetry,
    init_db,
    save_report,
    search_reports_rag,
)
from pdf_parser import extract_text_from_upload, normalize_transcript_text
from schemas import AnalysisResponse, BatchTranscriptRequest, TranscriptRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("finsight_ai")

app = Flask(__name__)
CORS(app)

init_db()


def validate_input_payload(
    data: Optional[dict],
) -> Tuple[Optional[TranscriptRequest], Optional[str]]:
    """Validate incoming payload using strict Pydantic v2 schema."""
    if not data or not isinstance(data, dict):
        return None, "No data provided"

    required_fields = ["company_name", "transcript_text"]
    for field in required_fields:
        if field not in data:
            return None, f"Missing required field: {field}"

    try:
        validated = TranscriptRequest(**data)
        return validated, None
    except ValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", "Invalid input payload")
        # Strip Pydantic's "Value error, " prefix for clean API error messages
        cleaned_msg = msg.replace("Value error, ", "")
        return None, cleaned_msg


def _process_single_transcript_item(validated_req: TranscriptRequest) -> Dict[str, Any]:
    """Helper to process a single validated TranscriptRequest (used by single & concurrent batch endpoints)."""
    company_name = validated_req.company_name
    ticker = validated_req.ticker
    quarter = validated_req.quarter
    transcript_text = normalize_transcript_text(validated_req.transcript_text)

    transcript_hash = compute_transcript_hash(company_name, transcript_text)
    cached = get_cached_report(transcript_hash)
    if cached:
        logger.info("Serving SHA-256 cached report for %s", company_name)
        return cached

    raw_analysis = analyze_earnings_transcript(
        company_name=company_name,
        transcript_text=transcript_text,
        ticker=ticker,
        quarter=quarter,
    )
    raw_analysis["company_name"] = company_name
    raw_analysis["ticker"] = ticker
    raw_analysis["quarter"] = quarter
    validated_output = AnalysisResponse(**raw_analysis).model_dump()

    report_id = save_report(
        company_name=company_name,
        ticker=ticker,
        quarter=quarter,
        transcript_hash=transcript_hash,
        analysis_data=validated_output,
        transcript_text=transcript_text,
    )
    validated_output["id"] = report_id
    validated_output["cached"] = False
    return validated_output


@app.route("/", methods=["GET"])
def home():
    """Render the FinSight AI Analytics Dashboard."""
    return render_template("index.html")


@app.route("/earnings_transcript_summary", methods=["POST"])
def summarize_transcript():
    """Analyze an earnings call transcript via JSON or file upload with Pydantic validation & SQLite caching."""
    try:
        if request.content_type and "multipart/form-data" in request.content_type:
            company_name = request.form.get("company_name", "")
            ticker = request.form.get("ticker", "")
            quarter = request.form.get("quarter", "")
            transcript_text = request.form.get("transcript_text", "")

            uploaded_file = request.files.get("file")
            if uploaded_file and uploaded_file.filename:
                ok, extracted = extract_text_from_upload(
                    uploaded_file.read(), uploaded_file.filename
                )
                if not ok:
                    return jsonify({"error": extracted}), 400
                transcript_text = extracted

            raw_data = {
                "company_name": company_name,
                "ticker": ticker,
                "quarter": quarter,
                "transcript_text": transcript_text,
            }
        else:
            raw_data = request.get_json(silent=True)

        validated_req, error_message = validate_input_payload(raw_data)
        if error_message or validated_req is None:
            return jsonify({"error": error_message}), 400

        result = _process_single_transcript_item(validated_req)
        return jsonify(result), 200

    except Exception as exc:
        logger.exception("Error processing transcript summary")
        return jsonify({"error": f"Internal server error: {str(exc)}"}), 500


@app.route("/batch_analyze", methods=["POST"])
def batch_analyze_transcripts():
    """Concurrently analyze up to 5 earnings transcripts in parallel using ThreadPoolExecutor."""
    raw_data = request.get_json(silent=True)
    if not raw_data or "items" not in raw_data:
        return jsonify({"error": "Payload must contain an 'items' list of transcripts"}), 400

    try:
        batch_req = BatchTranscriptRequest(**raw_data)
    except ValidationError as exc:
        first_err = exc.errors()[0]
        return jsonify({"error": first_err.get("msg", "Invalid batch payload")}), 400

    with ThreadPoolExecutor(max_workers=min(5, len(batch_req.items))) as executor:
        results = list(executor.map(_process_single_transcript_item, batch_req.items))

    return (
        jsonify(
            {
                "batch_count": len(results),
                "results": results,
            }
        ),
        200,
    )


@app.route("/export/<int:report_id>", methods=["GET"])
def export_report(report_id: int):
    """Export a stored intelligence report as an Executive Markdown (.md) or CSV (.csv) file."""
    report = get_report_by_id(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404

    fmt = request.args.get("format", "md").lower().strip()
    safe_name = report["company_name"].lower().replace(" ", "_")

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Field", "Value"])
        writer.writerow(["Company Name", report.get("company_name", "")])
        writer.writerow(["Ticker", report.get("ticker", "")])
        writer.writerow(["Quarter", report.get("quarter", "")])
        writer.writerow(["Overall Sentiment", report.get("overall_sentiment", "")])
        writer.writerow(["Confidence Score", report.get("confidence_score", "")])
        writer.writerow(["Risk Score", report.get("risk_score", "")])
        writer.writerow(["Financial Performance", report.get("financial_performance", "")])
        writer.writerow(["Market Dynamics", report.get("market_dynamics", "")])
        writer.writerow(["Expansion Plans", report.get("expansion_plans", "")])
        writer.writerow(["Environmental Risks", report.get("environmental_risks", "")])
        writer.writerow(["Regulatory Changes", report.get("regulatory_or_policy_changes", "")])
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={safe_name}_report.csv"},
        )

    # Default to Executive Markdown (.md) brief
    takeaways = "\n".join(f"- {t}" for t in report.get("executive_takeaways", []))
    md_content = f"""# {report.get('company_name')} ({report.get('ticker') or 'N/A'}) — {report.get('quarter') or 'Earnings Brief'}

**Overall Sentiment:** {report.get('overall_sentiment')} | **Confidence:** {report.get('confidence_score')}% | **Risk Index:** {report.get('risk_score')}/100

## Executive Takeaways
{takeaways}

## 5-Pillar Institutional Summary
### 1. Financial Performance
{report.get('financial_performance')}

### 2. Market Dynamics
{report.get('market_dynamics')}

### 3. Expansion Plans
{report.get('expansion_plans')}

### 4. Environmental & ESG Risks
{report.get('environmental_risks')}

### 5. Regulatory & Policy Changes
{report.get('regulatory_or_policy_changes')}
"""
    return Response(
        md_content,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={safe_name}_brief.md"},
    )


@app.route("/metrics", methods=["GET"])
def system_metrics():
    """Return real-time observability metrics and token savings across the SQLite cache."""
    telemetry = get_system_telemetry()
    return jsonify(telemetry), 200


@app.route("/search", methods=["GET"])
def search_transcripts():
    """Full-Text RAG Search endpoint across stored transcripts and intelligence summaries."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    matches = search_reports_rag(query=query, limit=15)
    return (
        jsonify(
            {
                "query": query,
                "count": len(matches),
                "results": matches,
            }
        ),
        200,
    )


@app.route("/history", methods=["GET"])
def list_history():
    """Return recently analyzed company reports from SQLite."""
    reports = get_all_reports(limit=25)
    return jsonify({"reports": reports, "count": len(reports)}), 200


@app.route("/history/<int:report_id>", methods=["GET"])
def get_history_item(report_id: int):
    """Fetch a single saved report by ID."""
    report = get_report_by_id(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    return jsonify(report), 200


@app.route("/history/<int:report_id>", methods=["DELETE"])
def delete_history_item(report_id: int):
    """Delete a saved report by ID."""
    deleted = delete_report_by_id(report_id)
    if not deleted:
        return jsonify({"error": "Report not found"}), 404
    return jsonify({"status": "deleted", "id": report_id}), 200


@app.route("/compare", methods=["GET"])
def compare_reports():
    """Compare two stored company reports side-by-side."""
    id1 = request.args.get("id1", type=int)
    id2 = request.args.get("id2", type=int)
    if not id1 or not id2:
        return jsonify({"error": "Query parameters id1 and id2 are required"}), 400

    r1 = get_report_by_id(id1)
    r2 = get_report_by_id(id2)
    if not r1 or not r2:
        return jsonify({"error": "One or both reports were not found"}), 404

    radar1 = r1.get("radar_scores", {})
    radar2 = r2.get("radar_scores", {})
    deltas = {
        key: int(radar1.get(key, 0)) - int(radar2.get(key, 0))
        for key in set(radar1.keys()) | set(radar2.keys())
    }

    return (
        jsonify(
            {
                "report_1": r1,
                "report_2": r2,
                "sentiment_comparison": {
                    r1["company_name"]: r1.get("overall_sentiment"),
                    r2["company_name"]: r2.get("overall_sentiment"),
                },
                "risk_delta": int(r1.get("risk_score", 0))
                - int(r2.get("risk_score", 0)),
                "radar_score_deltas": deltas,
            }
        ),
        200,
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint with database status."""
    reports = get_all_reports(limit=100)
    return (
        jsonify(
            {
                "status": "healthy",
                "service": "FinSight AI Engine",
                "cached_reports_count": len(reports),
            }
        ),
        200,
    )


@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    """Return the OpenAPI 3.0 specification for FinSight AI."""
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "FinSight AI — Enterprise Earnings Intelligence API",
            "version": "2.0.0",
            "description": (
                "REST API for corporate earnings call analysis, SHA-256 SQLite caching, "
                "full-text RAG search, and quarter-over-quarter risk comparison."
            ),
        },
        "paths": {
            "/earnings_transcript_summary": {
                "post": {
                    "summary": "Analyze an earnings call transcript (JSON or PDF upload)",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["company_name", "transcript_text"],
                                    "properties": {
                                        "company_name": {"type": "string", "example": "Apex Cloud Systems"},
                                        "ticker": {"type": "string", "example": "APEX"},
                                        "quarter": {"type": "string", "example": "Q3 FY26"},
                                        "transcript_text": {"type": "string", "example": "Revenue grew 24% YoY to $1.42B..."},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Structured 5-pillar analysis and 6-factor radar scores"},
                        "400": {"description": "Pydantic validation error"},
                    },
                }
            },
            "/search": {
                "get": {
                    "summary": "Full-text RAG search across stored transcripts",
                    "parameters": [
                        {"name": "q", "in": "query", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "Ranked context snippets"}},
                }
            },
            "/compare": {
                "get": {
                    "summary": "Compare two stored reports side-by-side (QoQ delta)",
                    "parameters": [
                        {"name": "id1", "in": "query", "required": True, "schema": {"type": "integer"}},
                        {"name": "id2", "in": "query", "required": True, "schema": {"type": "integer"}},
                    ],
                    "responses": {"200": {"description": "Radar score deltas and risk comparison"}},
                }
            },
            "/history": {
                "get": {
                    "summary": "List cached earnings intelligence reports",
                    "responses": {"200": {"description": "List of recent reports"}},
                }
            },
            "/health": {
                "get": {
                    "summary": "Service health check and cache count",
                    "responses": {"200": {"description": "Healthy status"}},
                }
            },
        },
    }
    return jsonify(spec), 200


@app.route("/apidocs", methods=["GET"])
def swagger_ui():
    """Render interactive Swagger UI documentation powered by /openapi.json."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>FinSight AI — Swagger API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
</head>
<body style="margin:0; background:#fafafa;">
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => {
      SwaggerUIBundle({
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true
      });
    };
  </script>
</body>
</html>"""
    return html, 200


if __name__ == "__main__":
    app.run(debug=True)