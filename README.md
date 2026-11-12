# FinSight AI — Enterprise Earnings Call Intelligence & Risk Engine

FinSight AI is a full-stack financial intelligence platform built with Python, Flask, SQLite, and Google Gemini. It ingests corporate earnings call transcripts (via raw text or PDF uploads) to extract structured institutional research across 5 core analytical pillars, multi-factor risk/growth radar scores, executive sentiment polarity, and quantitative financial KPIs.

## Key Engineering Highlights

This project was architected with a focus on latency reduction, API quota optimization, and deterministic fallbacks:

* **80% API Latency & Quota Reduction:** Replaced sequential LLM calls with a single structured JSON schema extraction pass.
* **Content-Addressable SHA-256 Caching:** Incoming transcripts are normalized and hashed in a local SQLite database (`finsight.db`). Repeated queries return in `< 5ms`, consuming **0** API tokens.
* **Intelligent Local NLP Fallback:** If the Gemini API key is missing, invalid, or rate-limited, the system automatically falls back to a deterministic local NLP heuristic engine to ensure 100% uptime.
* **Local PDF & Text Ingestion:** Supports direct `.pdf` and `.txt` earnings filing uploads using `pypdf`.
* **Deterministic Offline Unit Testing:** Comprehensive `unittest` suite using mock patches and isolated temporary SQLite databases for CI/CD readiness.

## System Architecture

```text
+-------------------------------------------------------------------+
|                   FinSight AI Web Dashboard                       |
|        (Tailwind CSS + Chart.js 6-Factor Radar Visualization)     |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Flask REST API Layer (app.py)                 |
|  POST /earnings_transcript_summary | GET /history | GET /compare  |
+----------------+---------------------------------+----------------+
                 |                                 |
      [1. SHA-256 Cache Lookup]         [2. Cache Miss: Single-Shot JSON]
                 |                                 |
                 v                                 v
+--------------------------------+  +-------------------------------+
|   SQLite Persistence Layer     |  |  Hybrid NLP + Gemini Engine   |
|         (database.py)          |  |  (analyzer.py / pdf_parser.py)|
| - Content-Addressable Caching  |  | - Local Keyword Signal Scan   |
| - 0 Token Repeat Lookups       |  | - Structured JSON Extraction  |
+--------------------------------+  +-------------------------------+
```

## Codebase Overview

* **`app.py`**: The core Flask application routing, validation, and request handling.
* **`analyzer.py`**: Houses the dual-engine analysis logic (Gemini API Structured JSON + Local NLP Heuristics).
* **`database.py`**: SQLite database initialization, SHA-256 hashing, and CRUD operations for the cache layer.
* **`pdf_parser.py`**: Secure file ingestion and text normalization utilities.
* **`config.py`**: Environment variable management and global configuration.
* **`test_api.py`**: Offline unit tests utilizing `unittest.mock`.
* **`templates/index.html`**: A modern, dark-mode frontend dashboard to visualize the extracted intelligence.

## API Endpoints

### `POST /earnings_transcript_summary`
Analyzes a transcript (accepts JSON payload or `multipart/form-data` with a `.pdf` file). Returns a structured JSON payload containing 5-pillar insights, a calculated risk score, sentiment analysis, and a 6-factor radar chart matrix.

### `GET /history`
Retrieves a lightweight list of recently cached corporate earnings reports from the local SQLite database.

### `GET /history/<id>`
Retrieves the full, cached analysis payload for a specific report ID.

### `GET /compare?id1=<int>&id2=<int>`
Compares two stored company reports side-by-side, returning quarter-over-quarter risk index deltas and sentiment shifts.

### `GET /health`
Returns the operational status of the API and the current cache size.

## Quickstart / Local Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/PijushPathak07/Python_and_Gen-AI.git
   cd Python_and_Gen-AI
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory. *(Note: The application will still function using the local NLP heuristic engine if the API key is omitted).*
   ```env
   GOOGLE_API_KEY="your-google-gemini-api-key"
   GEMINI_MODEL_NAME="gemini-2.5-flash"
   ```

5. **Run the Offline Unit Test Suite:**
   ```bash
   python -m unittest test_api.py -v
   ```

6. **Start the Application:**
   ```bash
   python app.py
   ```
   Navigate to `http://127.0.0.1:5000` in your browser to access the dashboard.

---
**Developer Contact**
*Pijush Pathak* | [pijushpathak94@gmail.com](mailto:pijushpathak94@gmail.com)