# FinSight AI — Enterprise Earnings Call Intelligence & Risk Engine

![CI Status](https://github.com/PijushPathak07/Python_and_Gen-AI/actions/workflows/ci.yml/badge.svg)
![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![Pydantic v2](https://img.shields.io/badge/validation-Pydantic%20v2-emerald)
![Docker Ready](https://img.shields.io/badge/container-Docker%20%2B%20Gunicorn-sky)

FinSight AI is a production-grade financial intelligence platform built with Python, Flask, Pydantic v2, SQLite, and Google Gemini. It ingests corporate earnings call transcripts (via raw text or PDF uploads) to extract structured institutional research across 5 core analytical pillars, multi-factor risk/growth radar scores, full-text RAG search snippets, and quarter-over-quarter (QoQ) comparative risk deltas.

## Key Engineering Highlights

* **80% API Latency & Quota Reduction:** Replaced sequential LLM calls with a single structured JSON schema extraction pass.
* **Content-Addressable SHA-256 Caching:** Incoming transcripts are normalized and hashed in a local SQLite database (`finsight.db`). Repeated queries return in `< 5ms`, consuming **0** API tokens.
* **Strict Pydantic v2 Schema Contracts (`schemas.py`):** Enforces rigorous input sanitization, token limits, and output structure validation.
* **Full-Text RAG Search Engine (`/search`):** Indexes raw corporate filings in SQLite and retrieves ranked context-window snippets across companies.
* **Interactive Swagger / OpenAPI 3.0 Docs (`/apidocs`):** Live browser-based API documentation and testing interface.
* **Docker & CI/CD Ready:** Includes a multi-worker `Dockerfile` (`gunicorn`), `docker-compose.yml`, and automated GitHub Actions matrix testing across Python 3.10 and 3.11.

## System Architecture

```text
+-------------------------------------------------------------------+
|                   FinSight AI Web Dashboard                       |
|   (Tailwind CSS + Chart.js Radar + RAG Search + QoQ Comparator)   |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|           Flask REST API + Pydantic v2 Layer (app.py)             |
|  POST /earnings_transcript_summary | GET /search | GET /compare   |
|  GET /history                      | GET /apidocs| GET /health    |
+----------------+---------------------------------+----------------+
                 |                                 |
      [1. SHA-256 Cache Lookup]         [2. Cache Miss: Single-Shot JSON]
                 |                                 |
                 v                                 v
+--------------------------------+  +-------------------------------+
|   SQLite Persistence & RAG     |  |  Hybrid NLP + Gemini Engine   |
|         (database.py)          |  |  (analyzer.py / pdf_parser.py)|
| - Content-Addressable Caching  |  | - Local Keyword Signal Scan   |
| - Ranked Context Snippet Search|  | - Structured JSON Extraction  |
+--------------------------------+  +-------------------------------+
```

## Codebase Overview

* **`app.py`**: Core Flask application routing, OpenAPI 3.0 specification (`/openapi.json`), and Swagger UI (`/apidocs`).
* **`schemas.py`**: Pydantic v2 data models (`TranscriptRequest`, `RadarScores`, `AnalysisResponse`) enforcing strict API validation.
* **`analyzer.py`**: Dual-engine analysis logic (Gemini API Structured JSON + Local Heuristic NLP Fallback).
* **`database.py`**: SQLite schema management, SHA-256 hashing, and full-text RAG search (`search_reports_rag`).
* **`pdf_parser.py`**: Secure `.pdf` and `.txt` file ingestion and text normalization utilities.
* **`config.py`**: Centralized environment configuration.
* **`test_api.py`**: 9 deterministic offline unit tests utilizing `unittest.mock` and isolated temporary SQLite databases.
* **`Dockerfile` & `docker-compose.yml`**: Production containerization running Gunicorn with non-root security.

## 🔌 API Endpoints

* **`POST /earnings_transcript_summary`**: Analyzes an earnings transcript via JSON or `.pdf`/`.txt` upload.
* **`GET /search?q=<keyword>`**: Full-text RAG search returning ranked context snippets across stored filings.
* **`GET /compare?id1=<int>&id2=<int>`**: Computes side-by-side QoQ radar score deltas and risk index shifts.
* **`GET /history`** & **`GET /history/<id>`**: Lists or retrieves cached company analysis reports.
* **`GET /apidocs`** & **`GET /openapi.json`**: Interactive Swagger UI and OpenAPI 3.0 spec.
* **`GET /health`**: Service health check and SQLite cache count.

## Quickstart (Local or Docker)

### Option A: Local Python Setup
```bash
git clone https://github.com/PijushPathak07/Python_and_Gen-AI.git
cd Python_and_Gen-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run the 9 offline unit tests
python -m unittest test_api.py -v

# Start the server
python app.py
```
* **Dashboard:** `http://127.0.0.1:5000`
* **Swagger API Docs:** `http://127.0.0.1:5000/apidocs`

### Option B: Production Docker Container
```bash
docker-compose up --build -d
```

---
**Developer Contact**
*Pijush Pathak* | [pijushpathak94@gmail.com](mailto:pijushpathak94@gmail.com)