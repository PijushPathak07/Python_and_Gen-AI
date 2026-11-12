import logging
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger("finsight_ai")

# Official SEC EDGAR Fair-Access User-Agent header requirement
SEC_HEADERS = {
    "User-Agent": "FinSightAI-Enterprise-Engine/2.0 (pijushpathak94@gmail.com)",
    "Accept": "application/json",
}

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik_padded}.json"


def resolve_cik_from_sec(ticker: str, timeout: int = 10) -> Optional[Dict[str, str]]:
    """Query live SEC EDGAR company_tickers.json to resolve a ticker to its 10-digit CIK."""
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        return None

    resp = requests.get(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    for entry in payload.values():
        if str(entry.get("ticker", "")).upper() == clean_ticker:
            raw_cik = str(entry.get("cik_str", "")).strip()
            return {
                "ticker": clean_ticker,
                "company_name": str(entry.get("title", clean_ticker)),
                "cik": raw_cik.zfill(10),
                "cik_int": raw_cik,
            }
    return None


def fetch_live_sec_edgar_profile(ticker: str, max_filings: int = 5) -> Dict[str, Any]:
    """Fetch live company metadata and recent 10-K/10-Q/8-K filings directly from SEC EDGAR."""
    resolved = resolve_cik_from_sec(ticker)
    if not resolved:
        raise LookupError(f"Ticker '{ticker.upper()}' was not found in SEC EDGAR registry.")

    cik_padded = resolved["cik"]
    cik_int = resolved["cik_int"]
    submissions_url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik_padded=cik_padded)

    sub_resp = requests.get(submissions_url, headers=SEC_HEADERS, timeout=10)
    sub_resp.raise_for_status()
    sub_data = sub_resp.json()

    recent = sub_data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    target_forms = {"10-K", "10-Q", "8-K", "20-F", "6-K"}
    recent_filings: List[Dict[str, str]] = []

    for idx, form_type in enumerate(forms):
        if form_type not in target_forms:
            continue
        acc_raw = accession_numbers[idx] if idx < len(accession_numbers) else ""
        acc_no_hyphens = acc_raw.replace("-", "")
        primary_doc = primary_docs[idx] if idx < len(primary_docs) else ""
        filing_date = filing_dates[idx] if idx < len(filing_dates) else ""
        report_date = report_dates[idx] if idx < len(report_dates) else ""

        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_hyphens}/{primary_doc}"
            if acc_no_hyphens and primary_doc
            else f"https://www.sec.gov/edgar/browse/?CIK={cik_padded}"
        )

        recent_filings.append(
            {
                "form": form_type,
                "filing_date": filing_date,
                "report_date": report_date or filing_date,
                "accession_number": acc_raw,
                "document_url": doc_url,
            }
        )
        if len(recent_filings) >= max_filings:
            break

    exchanges = sub_data.get("exchanges", [])
    exchange_str = exchanges[0] if exchanges else "SEC Registered"

    return {
        "ticker": resolved["ticker"],
        "company_name": sub_data.get("name") or resolved["company_name"],
        "cik": cik_padded,
        "sic": str(sub_data.get("sic", "")),
        "sic_description": str(sub_data.get("sicDescription", "")),
        "exchange": str(exchange_str),
        "fiscal_year_end": str(sub_data.get("fiscalYearEnd", "")),
        "state_of_incorporation": str(sub_data.get("stateOfIncorporation", "")),
        "sec_browse_url": f"https://www.sec.gov/edgar/browse/?CIK={cik_padded}",
        "recent_filings": recent_filings,
    }