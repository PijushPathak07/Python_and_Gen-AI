import logging
from typing import Optional, Tuple
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from analyzer import analyze_earnings_transcript
from config import Config
from database import (
    compute_transcript_hash,
    delete_report_by_id,
    get_all_reports,
    get_cached_report,
    get_report_by_id,
    init_db,
    save_report,
)
from pdf_parser import extract_text_from_upload, normalize_transcript_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("finsight_ai")

app = Flask(__name__)
CORS(app)

init_db()


def validate_input(data: Optional[dict]) -> Tuple[bool, Optional[str]]:
    """Validate incoming payload for required fields and token limits."""
    if not data:
        return False, "No data provided"

    required_fields = ["company_name", "transcript_text"]
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    if not str(data["company_name"]).strip():
        return False, "Company name cannot be empty"

    transcript = str(data["transcript_text"]).strip()
    if not transcript:
        return False, "Transcript text cannot be empty"

    if len(transcript) > Config.MAX_TOKENS:
        return (
            False,
            f"Transcript text exceeds maximum length of {Config.MAX_TOKENS} characters",
        )

    return True, None


@app.route("/", methods=["GET"])
def home():
    """Render the FinSight AI Analytics Dashboard."""
    return render_template("index.html")


@app.route("/earnings_transcript_summary", methods=["POST"])
def summarize_transcript():
    """Analyze an earnings call transcript via JSON or file upload with SQLite caching."""
    try:
        if request.content_type and "multipart/form-data" in request.content_type:
            company_name = request.form.get("company_name", "").strip()
            ticker = request.form.get("ticker", "").strip().upper()
            quarter = request.form.get("quarter", "").strip()
            transcript_text = request.form.get("transcript_text", "").strip()

            uploaded_file = request.files.get("file")
            if uploaded_file and uploaded_file.filename:
                ok, extracted = extract_text_from_upload(
                    uploaded_file.read(), uploaded_file.filename
                )
                if not ok:
                    return jsonify({"error": extracted}), 400
                transcript_text = extracted

            data = {
                "company_name": company_name,
                "ticker": ticker,
                "quarter": quarter,
                "transcript_text": transcript_text,
            }
        else:
            data = request.get_json(silent=True)

        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({"error": error_message}), 400

        company_name = data["company_name"].strip()
        ticker = str(data.get("ticker", "")).strip().upper()
        quarter = str(data.get("quarter", "")).strip()
        transcript_text = normalize_transcript_text(data["transcript_text"])

        transcript_hash = compute_transcript_hash(company_name, transcript_text)
        cached = get_cached_report(transcript_hash)
        if cached:
            logger.info("Serving SHA-256 cached report for %s", company_name)
            return jsonify(cached), 200

        analysis = analyze_earnings_transcript(
            company_name=company_name,
            transcript_text=transcript_text,
            ticker=ticker,
            quarter=quarter,
        )
        report_id = save_report(
            company_name=company_name,
            ticker=ticker,
            quarter=quarter,
            transcript_hash=transcript_hash,
            analysis_data=analysis,
        )
        analysis["id"] = report_id
        analysis["cached"] = False
        return jsonify(analysis), 200

    except Exception as exc:
        logger.exception("Error processing transcript summary")
        return jsonify({"error": f"Internal server error: {str(exc)}"}), 500


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


if __name__ == "__main__":
    app.run(debug=True)