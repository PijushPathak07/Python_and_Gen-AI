from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field, field_validator
from config import Config


class TranscriptRequest(BaseModel):
    """Strict input validation schema for incoming earnings call transcripts."""

    company_name: str = Field(..., min_length=1, max_length=200)
    ticker: str = Field(default="", max_length=20)
    quarter: str = Field(default="", max_length=40)
    transcript_text: str = Field(..., min_length=1)

    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Company name cannot be empty")
        return cleaned

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("quarter")
    @classmethod
    def validate_quarter(cls, v: str) -> str:
        return v.strip()

    @field_validator("transcript_text")
    @classmethod
    def validate_transcript_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Transcript text cannot be empty")
        if len(cleaned) > Config.MAX_TOKENS:
            raise ValueError(
                f"Transcript text exceeds maximum length of {Config.MAX_TOKENS} characters"
            )
        return cleaned


class RadarScores(BaseModel):
    """6-factor institutional scoring matrix bounded between 0 and 100."""

    revenue_growth: int = Field(default=75, ge=0, le=100)
    profitability: int = Field(default=75, ge=0, le=100)
    market_position: int = Field(default=75, ge=0, le=100)
    expansion_momentum: int = Field(default=75, ge=0, le=100)
    regulatory_safety: int = Field(default=75, ge=0, le=100)
    esg_sustainability: int = Field(default=70, ge=0, le=100)


class KPIItem(BaseModel):
    """Structured financial KPI metric extracted from transcript."""

    metric: str
    value: str
    impact: str = "Neutral"


class AnalysisResponse(BaseModel):
    """Validated output contract for FinSight AI intelligence reports."""

    company_name: str
    ticker: str = ""
    quarter: str = ""
    financial_performance: str
    market_dynamics: str
    expansion_plans: str
    environmental_risks: str
    regulatory_or_policy_changes: str
    overall_sentiment: Literal["Bullish", "Neutral", "Bearish"] = "Neutral"
    confidence_score: int = Field(default=75, ge=0, le=100)
    risk_score: int = Field(default=35, ge=0, le=100)
    radar_scores: RadarScores
    key_kpis: List[KPIItem] = Field(default_factory=list)
    executive_takeaways: List[str] = Field(default_factory=list)
    nlp_metrics: Dict[str, Any] = Field(default_factory=dict)


class BatchTranscriptRequest(BaseModel):
    """Validated schema for concurrent multi-company batch transcript processing."""

    items: List[TranscriptRequest] = Field(..., min_length=1, max_length=5)


class QuestionRequest(BaseModel):
    """Validated schema for asking a grounded question against a stored transcript."""

    question: str = Field(..., min_length=3, max_length=500)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty")
        return cleaned


class GroundedAnswerResponse(BaseModel):
    """Validated output schema for Grounded Q&A with exact verbatim quote citations."""

    report_id: int
    company_name: str
    question: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    confidence_score: int = Field(default=85, ge=0, le=100)
    grounding_mode: str = "deterministic_citation_rag"


class ForensicFlag(BaseModel):
    """Individual forensic or accounting risk signal detected in transcript."""

    category: str
    severity: Literal["Low", "Medium", "High"] = "Medium"
    signal: str
    evidence_sentence: str


class ForensicAuditResponse(BaseModel):
    """Validated output contract for Forensic Red-Flag & Management Hedging Detector."""

    report_id: int
    company_name: str
    transparency_score: int = Field(default=85, ge=0, le=100)
    evasion_risk_level: Literal["Low", "Moderate", "High"] = "Low"
    hedging_word_count: int = 0
    certainty_word_count: int = 0
    hedging_ratio_pct: float = 0.0
    detected_hedging_terms: Dict[str, int] = Field(default_factory=dict)
    red_flags: List[ForensicFlag] = Field(default_factory=list)