import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from config import Config


def compute_transcript_hash(company_name: str, transcript_text: str) -> str:
    """Compute a deterministic SHA-256 hash for content-addressable caching."""
    normalized = f"{company_name.strip().lower()}::{' '.join(transcript_text.split())}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a SQLite connection with row factory enabled."""
    path = db_path or Config.DATABASE_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the SQLite schema for storing earnings reports."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT NOT NULL,
                ticker TEXT DEFAULT '',
                quarter TEXT DEFAULT '',
                transcript_hash TEXT NOT NULL UNIQUE,
                analysis_json TEXT NOT NULL,
                overall_sentiment TEXT NOT NULL,
                confidence_score INTEGER NOT NULL,
                risk_score INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_transcript_hash ON reports(transcript_hash)"
        )
        conn.commit()


def get_cached_report(
    transcript_hash: str, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Fetch a previously analyzed report by its SHA-256 hash (0 API tokens)."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM reports WHERE transcript_hash = ?", (transcript_hash,)
        ).fetchone()
        if not row:
            return None
        payload = json.loads(row["analysis_json"])
        payload["id"] = row["id"]
        payload["cached"] = True
        payload["created_at"] = row["created_at"]
        return payload


def save_report(
    company_name: str,
    ticker: str,
    quarter: str,
    transcript_hash: str,
    analysis_data: Dict[str, Any],
    db_path: Optional[str] = None,
) -> int:
    """Persist an analyzed earnings report to SQLite and return its row ID."""
    created_at = datetime.now(timezone.utc).isoformat()
    overall_sentiment = analysis_data.get("overall_sentiment", "Neutral")
    confidence_score = int(analysis_data.get("confidence_score", 75))
    risk_score = int(analysis_data.get("risk_score", 35))

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO reports (
                company_name, ticker, quarter, transcript_hash,
                analysis_json, overall_sentiment, confidence_score,
                risk_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_name,
                ticker,
                quarter,
                transcript_hash,
                json.dumps(analysis_data),
                overall_sentiment,
                confidence_score,
                risk_score,
                created_at,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_all_reports(
    limit: int = 20, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve a summary list of recent earnings analyses."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, company_name, ticker, quarter, overall_sentiment,
                   confidence_score, risk_score, created_at
            FROM reports
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_report_by_id(
    report_id: int, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Retrieve the full JSON analysis for a specific report ID."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM reports WHERE id = ?", (report_id,)
        ).fetchone()
        if not row:
            return None
        payload = json.loads(row["analysis_json"])
        payload["id"] = row["id"]
        payload["cached"] = True
        payload["created_at"] = row["created_at"]
        return payload


def delete_report_by_id(report_id: int, db_path: Optional[str] = None) -> bool:
    """Delete a stored report by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        conn.commit()
        return cursor.rowcount > 0