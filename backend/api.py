"""
SECTION 5: FASTAPI BACKEND SERVER
====================================
REST API that connects all backend modules:
  POST /analyze/phq9         – Score PHQ-9 responses
  POST /analyze/chat         – Analyze chat message sentiment
  POST /analyze/full         – Full analysis (PHQ-9 + chat history)
  POST /speech/transcribe    – Transcribe audio to text
  GET  /health               – Health check

Run with:
  uvicorn backend.api:app --reload --port 8000
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import tempfile
import logging

# Internal modules
from phq9_scorer        import FuzzyPHQ9Scorer
from genetic_algorithm  import get_default_weights
from sentiment_analyzer import SentimentAnalyzer
from depression_classifier import DepressionClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title      = "Depression Analysis API",
    description= "PHQ-9 + Sentiment + GA/Fuzzy Logic Depression Assessment",
    version    = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module Initialisation ──────────────────────────────────────────────────────
logger.info("Loading GA-optimised weights...")
_weights   = get_default_weights()        # Replace with run_ga() output after training

logger.info("Initialising FuzzyPHQ9Scorer...")
phq9_scorer = FuzzyPHQ9Scorer(weights=_weights)

logger.info("Initialising SentimentAnalyzer...")
sentiment_analyzer = SentimentAnalyzer()

logger.info("Initialising DepressionClassifier...")
classifier = DepressionClassifier()

logger.info("All modules loaded ✓")


# ── Pydantic Request/Response Models ──────────────────────────────────────────

class PHQ9Request(BaseModel):
    """Nine PHQ-9 answers, each 0–3."""
    responses: List[int] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="List of 9 integers, each in [0, 3]",
        examples=[[1, 2, 1, 0, 2, 1, 1, 0, 0]]
    )

    def validate_responses(self):
        for i, r in enumerate(self.responses):
            if r not in (0, 1, 2, 3):
                raise ValueError(f"Response {i+1} must be 0–3, got {r}")


class ChatRequest(BaseModel):
    """Single chat message for sentiment analysis."""
    message: str = Field(..., min_length=1, max_length=5000)


class FullAnalysisRequest(BaseModel):
    """Combined PHQ-9 + full chat history for comprehensive assessment."""
    phq9_responses: List[int] = Field(..., min_length=9, max_length=9)
    chat_history:   List[str] = Field(default=[], description="All user messages in order")
    latest_message: Optional[str] = Field(None, description="Most recent user message")


class PHQ9Response(BaseModel):
    raw_score:      int
    weighted_score: float
    severity_index: float
    severity_label: str
    severity_pct:   str
    memberships:    dict
    weights_used:   list


class ChatResponse(BaseModel):
    sentiment_score:      float
    depression_signal:    str
    has_severe_keywords:  bool
    textblob_polarity:    float
    vader_compound:       float


class FullAnalysisResponse(BaseModel):
    final_score:        float
    final_label:        str
    urgency:            str
    is_crisis:          bool
    color:              str
    phq9_severity_index: float
    phq9_label:         str
    sentiment_score:    float
    sentiment_signal:   str
    summary:            str
    immediate_actions:  list
    lifestyle_tips:     list
    professional_advice: str
    resources:          list
    bot_response:       str
    chat_trajectory:    Optional[str]
    severe_lang_detected: bool
    memberships:        dict


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "modules": ["phq9_scorer", "sentiment_analyzer", "classifier"]}


@app.post("/analyze/phq9", response_model=PHQ9Response)
async def analyze_phq9(req: PHQ9Request):
    """
    Score PHQ-9 responses using Fuzzy Logic + GA weights.
    
    Returns severity index, label, and membership degrees.
    """
    try:
        req.validate_responses()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        result      = phq9_scorer.score(req.responses)
        memberships = phq9_scorer.membership_degrees(result["weighted_score"])
        return PHQ9Response(
            raw_score      = result["raw_score"],
            weighted_score = result["weighted_score"],
            severity_index = result["severity_index"],
            severity_label = result["severity_label"],
            severity_pct   = result["severity_pct"],
            memberships    = memberships,
            weights_used   = _weights,
        )
    except Exception as e:
        logger.error(f"PHQ-9 scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


@app.post("/analyze/chat", response_model=ChatResponse)
async def analyze_chat(req: ChatRequest):
    """
    Analyze a single chat message for depression-related sentiment.
    """
    try:
        result = sentiment_analyzer.analyze(req.message)
        return ChatResponse(
            sentiment_score     = result["sentiment_score"],
            depression_signal   = result["depression_signal"],
            has_severe_keywords = result["has_severe_keywords"],
            textblob_polarity   = result["textblob"]["polarity"],
            vader_compound      = result["vader"]["compound"],
        )
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/full", response_model=FullAnalysisResponse)
async def full_analysis(req: FullAnalysisRequest):
    """
    Complete depression analysis:
      1. Score PHQ-9 with fuzzy logic + GA weights
      2. Analyze latest chat message sentiment
      3. Batch-analyze full chat history
      4. Fuse all signals into final classification
      5. Return recommendations + chatbot response
    """
    # Validate PHQ-9
    for i, r in enumerate(req.phq9_responses):
        if r not in (0, 1, 2, 3):
            raise HTTPException(status_code=422, detail=f"Response {i+1} must be 0–3")

    try:
        # Step 1: PHQ-9 fuzzy scoring
        phq9_result   = phq9_scorer.score(req.phq9_responses)
        memberships   = phq9_scorer.membership_degrees(phq9_result["weighted_score"])
        phq9_result["responses"] = req.phq9_responses

        # Step 2: Latest message sentiment
        latest_msg     = req.latest_message or (req.chat_history[-1] if req.chat_history else "")
        if latest_msg:
            sent_result = sentiment_analyzer.analyze(latest_msg)
        else:
            sent_result = {
                "sentiment_score": 0.5,
                "depression_signal": "Neutral",
                "has_severe_keywords": False,
            }

        # Step 3: Batch analysis of chat history
        batch_result = None
        if req.chat_history:
            batch_result = sentiment_analyzer.batch_analyze(req.chat_history)

        # Step 4 & 5: Classify + generate response
        report      = classifier.classify(phq9_result, sent_result, memberships, batch_result)
        bot_response = classifier.generate_chat_response(report)

        return FullAnalysisResponse(
            final_score         = report.final_score,
            final_label         = report.final_label,
            urgency             = report.urgency,
            is_crisis           = report.is_crisis,
            color               = report.color,
            phq9_severity_index = report.phq9_severity_index,
            phq9_label          = report.phq9_label,
            sentiment_score     = report.sentiment_score,
            sentiment_signal    = report.sentiment_signal,
            summary             = report.summary,
            immediate_actions   = report.immediate_actions,
            lifestyle_tips      = report.lifestyle_tips,
            professional_advice = report.professional_advice,
            resources           = report.resources,
            bot_response        = bot_response,
            chat_trajectory     = report.chat_trajectory,
            severe_lang_detected = report.severe_lang_detected,
            memberships         = memberships,
        )

    except Exception as e:
        logger.error(f"Full analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speech/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe uploaded audio file to text using Google Speech Recognition.
    Accepts .wav, .mp3, .ogg files.
    """
    try:
        import speech_recognition as sr

        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language="en-IN")
        os.remove(tmp_path)

        return {"transcript": text, "success": True}

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "success": False}
        )


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)