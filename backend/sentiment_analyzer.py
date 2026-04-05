"""
SECTION 3: SENTIMENT ANALYSIS ENGINE
======================================
Multi-layer sentiment analysis pipeline combining:
  1. TextBlob  – lexicon-based polarity/subjectivity
  2. VADER     – rule-based, optimised for social/clinical text
  3. HuggingFace Transformers – deep learning (distilbert-base-uncased-finetuned-sst-2-english)
  4. Keyword mapping – clinical depression keyword detection
  5. Score fusion – weighted ensemble of all signals

The final sentiment_score (0–1) maps to a depression_contribution:
  0.0–0.2  → Strong negative affect (high depression signal)
  0.2–0.4  → Negative affect
  0.4–0.6  → Neutral
  0.6–0.8  → Positive affect
  0.8–1.0  → Strong positive affect
"""

import re
import nltk
import numpy as np
from textblob import TextBlob

# Download required NLTK data (run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Attempt to load transformer model (falls back gracefully if unavailable)
try:
    from transformers import pipeline as hf_pipeline
    _transformer_model = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    TRANSFORMER_AVAILABLE = True
except Exception:
    _transformer_model    = None
    TRANSFORMER_AVAILABLE = False
    print("[Warning] Transformer model unavailable. Using TextBlob + VADER only.")


# ── Clinical keyword banks ─────────────────────────────────────────────────────
DEPRESSION_KEYWORDS = {
    "severe": [
        "suicidal", "suicide", "kill myself", "end my life", "no reason to live",
        "worthless", "hopeless", "can't go on", "give up on life", "self-harm",
        "hurt myself", "nothing matters", "dead inside",
    ],
    "high": [
        "hopeless", "empty", "numb", "broken", "useless", "failure", "burden",
        "hate myself", "nobody cares", "no future", "alone", "isolated",
        "crying", "tears", "miserable", "despair", "pointless",
    ],
    "moderate": [
        "sad", "unhappy", "depressed", "anxious", "tired", "exhausted",
        "struggling", "overwhelmed", "lost", "stuck", "unmotivated",
        "can't sleep", "no energy", "not eating", "worried",
    ],
    "mild": [
        "stressed", "low", "not great", "a bit down", "off", "meh",
        "distracted", "not myself", "flat", "blah",
    ],
}

POSITIVE_KEYWORDS = [
    "happy", "good", "great", "better", "hopeful", "motivated", "energised",
    "grateful", "positive", "improving", "calm", "peaceful", "joyful",
]


class SentimentAnalyzer:
    """Multi-modal sentiment analyzer for depression assessment."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # Boost VADER lexicon with clinical terms
        self.vader.lexicon.update({
            "hopeless": -3.5, "worthless": -3.2, "suicidal": -4.0,
            "numb":     -2.5, "broken":    -2.8, "burden":   -3.0,
            "hopeful":   2.5, "grateful":   2.2, "improving": 2.0,
        })

    def _clean_text(self, text: str) -> str:
        """Basic text normalisation."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _textblob_score(self, text: str) -> dict:
        """TextBlob polarity [-1, 1] → normalised [0, 1]."""
        blob      = TextBlob(text)
        polarity  = blob.sentiment.polarity       # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        normalised = (polarity + 1) / 2            # 0 to 1
        return {
            "polarity":     round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
            "normalised":   round(normalised, 4),
        }

    def _vader_score(self, text: str) -> dict:
        """VADER compound score [-1, 1] → normalised [0, 1]."""
        scores   = self.vader.polarity_scores(text)
        compound = scores['compound']
        return {
            "compound":   round(compound, 4),
            "positive":   scores['pos'],
            "neutral":    scores['neu'],
            "negative":   scores['neg'],
            "normalised": round((compound + 1) / 2, 4),
        }

    def _transformer_score(self, text: str) -> dict:
        """HuggingFace transformer score → normalised [0, 1]."""
        if not TRANSFORMER_AVAILABLE:
            return {"normalised": 0.5, "label": "UNAVAILABLE", "confidence": 0.0}

        result     = _transformer_model(text[:512])[0]
        label      = result['label']   # 'POSITIVE' or 'NEGATIVE'
        confidence = result['score']

        # Map to [0,1]: POSITIVE=confidence, NEGATIVE=1-confidence
        normalised = confidence if label == "POSITIVE" else 1.0 - confidence
        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "normalised": round(normalised, 4),
        }

    def _keyword_score(self, text: str) -> dict:
        """
        Keyword-based depression signal.
        Returns score in [0, 1]:
          1.0 = very positive (no depression keywords)
          0.0 = severe depression keywords detected
        """
        text_lower = text.lower()
        found      = {"severe": [], "high": [], "moderate": [], "mild": []}
        positive_hits = []

        for level, keywords in DEPRESSION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    found[level].append(kw)

        for kw in POSITIVE_KEYWORDS:
            if kw in text_lower:
                positive_hits.append(kw)

        # Penalty scoring
        penalty = (
            len(found["severe"])   * 0.40 +
            len(found["high"])     * 0.20 +
            len(found["moderate"]) * 0.10 +
            len(found["mild"])     * 0.05
        )
        # Boost from positive keywords
        boost = len(positive_hits) * 0.05

        raw_score = 1.0 - penalty + boost
        return {
            "score":          round(float(np.clip(raw_score, 0.0, 1.0)), 4),
            "found_keywords": found,
            "positive_hits":  positive_hits,
            "has_severe":     len(found["severe"]) > 0,
        }

    def analyze(self, text: str) -> dict:
        """
        Full multi-modal sentiment analysis pipeline.

        Args:
            text: User's free-text message.
        Returns:
            Comprehensive sentiment report dict.
        """
        if not text or not text.strip():
            return {
                "error":             "Empty input",
                "sentiment_score":   0.5,
                "depression_signal": "Neutral",
            }

        cleaned = self._clean_text(text)

        tb    = self._textblob_score(cleaned)
        vader = self._vader_score(cleaned)
        trans = self._transformer_score(cleaned)
        kw    = self._keyword_score(cleaned)

        # ── Weighted ensemble fusion ────────────────────────────────────────
        # Weights: transformer > vader > keyword > textblob
        # Keyword gets high weight because clinical terms are the most reliable signal
        w_tb    = 0.15
        w_vader = 0.30
        w_trans = 0.35 if TRANSFORMER_AVAILABLE else 0.0
        w_kw    = 0.40 if not TRANSFORMER_AVAILABLE else 0.20

        # Normalise weights
        total_w = w_tb + w_vader + w_trans + w_kw
        fused   = (
            tb["normalised"]    * w_tb    +
            vader["normalised"] * w_vader +
            trans["normalised"] * w_trans +
            kw["score"]         * w_kw
        ) / total_w

        fused = float(np.clip(fused, 0.0, 1.0))

        # Override: if severe keywords detected, cap score at 0.2
        if kw["has_severe"]:
            fused = min(fused, 0.20)

        signal, contribution = self._interpret_signal(fused)

        return {
            "sentiment_score":      round(fused, 4),
            "depression_signal":    signal,
            "depression_contribution": contribution,
            "textblob":             tb,
            "vader":                vader,
            "transformer":          trans,
            "keywords":             kw,
            "has_severe_keywords":  kw["has_severe"],
            "model_available":      TRANSFORMER_AVAILABLE,
        }

    @staticmethod
    def _interpret_signal(score: float) -> tuple:
        """Map fused score to depression signal label and contribution weight."""
        if score < 0.20:
            return "Very High Depression Signal", 0.90
        elif score < 0.35:
            return "High Depression Signal",      0.70
        elif score < 0.50:
            return "Moderate Depression Signal",  0.50
        elif score < 0.65:
            return "Mild Depression Signal",      0.30
        elif score < 0.80:
            return "Low Depression Signal",       0.15
        else:
            return "Minimal Depression Signal",   0.05

    def batch_analyze(self, messages: list) -> dict:
        """
        Analyze a conversation history (list of strings).
        Returns aggregate stats + per-message results.
        """
        results = [self.analyze(msg) for msg in messages if msg.strip()]
        if not results:
            return {"error": "No valid messages"}

        scores     = [r["sentiment_score"] for r in results]
        trajectory = "Worsening" if scores[-1] < scores[0] - 0.1 else \
                     "Improving" if scores[-1] > scores[0] + 0.1 else "Stable"

        return {
            "messages":          results,
            "average_score":     round(float(np.mean(scores)), 4),
            "min_score":         round(float(np.min(scores)), 4),
            "max_score":         round(float(np.max(scores)), 4),
            "trajectory":        trajectory,
            "severe_detected":   any(r["has_severe_keywords"] for r in results),
        }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    test_messages = [
        "I feel completely hopeless and like a burden to everyone around me.",
        "I've been struggling to sleep and feeling really tired all the time.",
        "Things are okay, not great, just kind of going through the motions.",
        "I went for a walk today and actually felt a bit better than yesterday.",
        "I can't stop crying and I don't even know why. Everything feels pointless.",
        "I've been having thoughts of hurting myself and I don't know what to do.",
    ]

    for msg in test_messages:
        result = analyzer.analyze(msg)
        print(f"\nText: {msg[:60]}...")
        print(f"  Score: {result['sentiment_score']}  |  Signal: {result['depression_signal']}")
        print(f"  Severe keywords: {result['has_severe_keywords']}")

    print("\n── Batch Analysis ──────────────────────────────")
    batch = analyzer.batch_analyze(test_messages)
    print(f"Average score: {batch['average_score']}")
    print(f"Trajectory:    {batch['trajectory']}")
    print(f"Severe detected: {batch['severe_detected']}")