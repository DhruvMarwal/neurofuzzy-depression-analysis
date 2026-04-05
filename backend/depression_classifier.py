"""
SECTION 4: DEPRESSION CLASSIFIER & RECOMMENDATION ENGINE
==========================================================
Fuses PHQ-9 fuzzy scores with chat sentiment analysis to produce
a final depression severity assessment and actionable recommendations.

Fusion formula:
  final_score = α × phq9_severity_index + β × sentiment_contribution × 100

  α = 0.65  (PHQ-9 is the clinical gold standard)
  β = 0.35  (sentiment provides supplemental signal)

Output levels:
  0–20   → No/Minimal Depression
  21–40  → Mild Depression
  41–60  → Moderate Depression
  61–80  → Moderately Severe Depression
  81–100 → Severe Depression
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


# ── Recommendation Database ────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "No/Minimal Depression": {
        "summary": (
            "Your responses suggest minimal to no depression. "
            "You appear to be managing well emotionally."
        ),
        "immediate": [
            "Maintain your current healthy habits and routines.",
            "Stay socially connected with friends and family.",
            "Continue regular physical activity — even a 20-minute walk helps.",
        ],
        "lifestyle": [
            "Practice mindfulness or meditation for 5–10 minutes daily.",
            "Maintain a consistent sleep schedule (7–9 hours).",
            "Eat a balanced diet rich in omega-3s (fish, walnuts, flaxseeds).",
            "Limit alcohol and caffeine intake.",
        ],
        "professional": "No immediate professional intervention required. Consider annual mental health check-ins.",
        "resources": [
            "Headspace / Calm app for mindfulness",
            "\"Feeling Good\" by Dr David Burns (self-help CBT book)",
        ],
        "crisis": False,
        "urgency": "Low",
        "color": "#22c55e",
    },
    "Mild Depression": {
        "summary": (
            "You're experiencing mild depression symptoms. "
            "With the right strategies, many people improve significantly."
        ),
        "immediate": [
            "Talk to a trusted friend or family member about how you're feeling.",
            "Establish a daily routine — structure helps reduce depressive symptoms.",
            "Set small, achievable goals each day to build momentum.",
        ],
        "lifestyle": [
            "Exercise at least 30 minutes, 3–4 times a week — it's clinically proven to reduce depression.",
            "Reduce screen time and social media usage, especially before bed.",
            "Spend time outdoors — natural light regulates mood and sleep.",
            "Keep a gratitude journal — write 3 things you're grateful for each day.",
        ],
        "professional": (
            "Consider speaking with a counsellor or therapist. "
            "Cognitive Behavioural Therapy (CBT) is highly effective for mild depression."
        ),
        "resources": [
            "iCBT (internet-based CBT platforms like Woebot or MindShift)",
            "iCall India: 9152987821",
            "Vandrevala Foundation Helpline: 1860-2662-345",
        ],
        "crisis": False,
        "urgency": "Moderate",
        "color": "#eab308",
    },
    "Moderate Depression": {
        "summary": (
            "Your responses indicate moderate depression. "
            "Professional support is strongly recommended and can make a significant difference."
        ),
        "immediate": [
            "Do not isolate yourself — reach out to someone you trust today.",
            "Avoid making major life decisions while feeling this way.",
            "Maintain basic self-care: eating, sleeping, and hygiene.",
        ],
        "lifestyle": [
            "Create a gentle daily schedule — don't over-commit.",
            "Exercise if possible, even light walking is beneficial.",
            "Avoid alcohol and recreational drugs, which worsen depression.",
            "Practice deep breathing or progressive muscle relaxation.",
        ],
        "professional": (
            "Please consult a psychiatrist or clinical psychologist soon. "
            "A combination of therapy (CBT, IPT) and potentially medication may be recommended. "
            "Consider sharing your PHQ-9 results with your doctor."
        ),
        "resources": [
            "iCall India: 9152987821",
            "NIMHANS Bengaluru: 080-46110007",
            "Vandrevala Foundation 24/7: 1860-2662-345",
            "therapists.psychologytoday.com to find a local therapist",
        ],
        "crisis": False,
        "urgency": "High",
        "color": "#f97316",
    },
    "Moderately Severe Depression": {
        "summary": (
            "Your responses suggest moderately severe depression. "
            "Please seek professional help as soon as possible — you deserve support."
        ),
        "immediate": [
            "Contact a mental health professional or your doctor this week.",
            "Share how you're feeling with a trusted person who can support you.",
            "Remove or reduce access to anything that may pose a risk to your safety.",
        ],
        "lifestyle": [
            "Try to maintain basic routines even when it feels difficult.",
            "Accept help from others — you don't have to manage this alone.",
            "Avoid alcohol, which is a depressant and will worsen symptoms.",
        ],
        "professional": (
            "Urgent professional evaluation is strongly recommended. "
            "A psychiatrist can assess whether medication (antidepressants like SSRIs) "
            "combined with psychotherapy (CBT or DBT) would help. "
            "Do not delay seeking help."
        ),
        "resources": [
            "iCall India: 9152987821 (Mon–Sat, 8am–10pm)",
            "Vandrevala Foundation 24/7: 1860-2662-345",
            "AASRA: 9820466627 (24/7)",
            "Snehi India: 044-24640050",
        ],
        "crisis": True,
        "urgency": "Urgent",
        "color": "#ef4444",
    },
    "Severe Depression": {
        "summary": (
            "Your responses indicate severe depression. "
            "Please reach out for help immediately — this is treatable and you are not alone."
        ),
        "immediate": [
            "PLEASE call a crisis helpline right now if you are having thoughts of harming yourself.",
            "Go to the nearest hospital emergency department if you feel unsafe.",
            "Tell someone you trust how you are feeling right now.",
        ],
        "lifestyle": [
            "Focus only on the next few hours — small steps matter.",
            "Do not be alone if possible.",
        ],
        "professional": (
            "Immediate professional intervention is required. "
            "Please contact your doctor, a psychiatrist, or go to a hospital emergency department. "
            "Inpatient care may be recommended. Treatment is effective — please reach out now."
        ),
        "resources": [
            "🚨 iCall India: 9152987821",
            "🚨 AASRA 24/7: 9820466627",
            "🚨 Vandrevala Foundation 24/7: 1860-2662-345",
            "🚨 iMumps: 9373111709",
            "🚨 Nearest hospital emergency department",
        ],
        "crisis": True,
        "urgency": "Emergency",
        "color": "#7c3aed",
    },
}


@dataclass
class DepressionReport:
    """Complete depression analysis report."""
    # Scores
    phq9_raw_score:     int
    phq9_weighted_score: float
    phq9_severity_index: float
    sentiment_score:    float
    final_score:        float

    # Labels
    phq9_label:         str
    sentiment_signal:   str
    final_label:        str
    urgency:            str

    # PHQ-9 details
    phq9_responses:     list
    phq9_memberships:   dict

    # Recommendations
    summary:            str
    immediate_actions:  list
    lifestyle_tips:     list
    professional_advice: str
    resources:          list
    is_crisis:          bool
    color:              str

    # Optional extras
    chat_trajectory:    Optional[str] = None
    severe_lang_detected: bool = False


class DepressionClassifier:
    """
    Fuses PHQ-9 fuzzy scoring + sentiment analysis into a final depression report.
    """

    # Fusion weights
    ALPHA = 0.65   # PHQ-9 weight
    BETA  = 0.35   # Sentiment weight

    def classify(
        self,
        phq9_result: dict,
        sentiment_result: dict,
        phq9_memberships: dict = None,
        batch_sentiment: dict = None,
    ) -> DepressionReport:
        """
        Produce a final DepressionReport.

        Args:
            phq9_result:      Output from FuzzyPHQ9Scorer.score()
            sentiment_result: Output from SentimentAnalyzer.analyze() (latest message)
            phq9_memberships: Output from FuzzyPHQ9Scorer.membership_degrees()
            batch_sentiment:  Output from SentimentAnalyzer.batch_analyze() (full chat)
        Returns:
            DepressionReport dataclass
        """
        # Extract scores
        phq9_si   = phq9_result.get("severity_index", 0.0)
        sent_score = sentiment_result.get("sentiment_score", 0.5)
        # Convert sentiment (0=bad, 1=good) to depression contribution (0=good, 1=bad)
        sent_dep  = (1.0 - sent_score) * 100.0

        # Weighted fusion
        final_score = self.ALPHA * phq9_si + self.BETA * sent_dep
        final_score = float(np.clip(final_score, 0.0, 100.0))

        # Override: severe language detected → floor at 61 (ModeratelySevere)
        severe_lang = sentiment_result.get("has_severe_keywords", False)
        if severe_lang and final_score < 61.0:
            final_score = max(final_score, 61.0)

        final_label = self._final_label(final_score)
        recs        = RECOMMENDATIONS[final_label]

        # Chat trajectory (from batch analysis)
        trajectory = None
        if batch_sentiment:
            trajectory = batch_sentiment.get("trajectory")

        return DepressionReport(
            phq9_raw_score      = phq9_result.get("raw_score", 0),
            phq9_weighted_score = phq9_result.get("weighted_score", 0.0),
            phq9_severity_index = phq9_si,
            sentiment_score     = sent_score,
            final_score         = round(final_score, 2),
            phq9_label          = phq9_result.get("severity_label", ""),
            sentiment_signal    = sentiment_result.get("depression_signal", ""),
            final_label         = final_label,
            urgency             = recs["urgency"],
            phq9_responses      = phq9_result.get("responses", []),
            phq9_memberships    = phq9_memberships or {},
            summary             = recs["summary"],
            immediate_actions   = recs["immediate"],
            lifestyle_tips      = recs["lifestyle"],
            professional_advice = recs["professional"],
            resources           = recs["resources"],
            is_crisis           = recs["crisis"],
            color               = recs["color"],
            chat_trajectory     = trajectory,
            severe_lang_detected = severe_lang,
        )

    @staticmethod
    def _final_label(score: float) -> str:
        if score <= 20:
            return "No/Minimal Depression"
        elif score <= 40:
            return "Mild Depression"
        elif score <= 60:
            return "Moderate Depression"
        elif score <= 80:
            return "Moderately Severe Depression"
        else:
            return "Severe Depression"

    def to_dict(self, report: DepressionReport) -> dict:
        """Serialise report to JSON-safe dict."""
        return asdict(report)

    def generate_chat_response(self, report: DepressionReport) -> str:
        """
        Generate a warm, empathetic chatbot response based on the report.
        """
        label   = report.final_label
        score   = report.final_score
        crisis  = report.is_crisis
        urgency = report.urgency

        base = f"Based on your PHQ-9 responses and our conversation, "

        if label == "No/Minimal Depression":
            return (
                f"{base}you appear to be doing well emotionally. "
                f"Your depression severity score is {score:.0f}/100, indicating minimal concern. "
                f"Keep up the healthy habits — even small daily practices matter. 🌿"
            )
        elif label == "Mild Depression":
            return (
                f"{base}I can see you've been going through some difficult moments. "
                f"Your score of {score:.0f}/100 suggests mild depression. "
                f"This is very manageable — speaking to someone you trust and building "
                f"small positive routines can make a real difference. You're not alone in this. 💛"
            )
        elif label == "Moderate Depression":
            return (
                f"{base}I want you to know that what you're feeling is real and valid. "
                f"Your score of {score:.0f}/100 indicates moderate depression. "
                f"I'd really encourage you to speak with a mental health professional — "
                f"therapy has helped many people in similar situations. "
                f"Would you like to know more about next steps? 🤝"
            )
        elif label == "Moderately Severe Depression":
            return (
                f"{base}I'm genuinely concerned about how you're feeling right now. "
                f"Your score of {score:.0f}/100 suggests moderately severe depression. "
                f"Please reach out to a doctor or therapist soon — "
                f"this level of depression responds well to treatment. "
                f"You deserve care and support. 💙\n\n"
                f"📞 iCall: 9152987821 | AASRA: 9820466627"
            )
        else:  # Severe
            return (
                f"I'm very concerned about you. Your score of {score:.0f}/100 "
                f"indicates severe depression. Please reach out for help right now — "
                f"you do not have to face this alone.\n\n"
                f"🚨 AASRA (24/7): 9820466627\n"
                f"🚨 Vandrevala Foundation: 1860-2662-345\n"
                f"🚨 iCall: 9152987821\n\n"
                f"Please call one of these numbers or go to your nearest hospital. "
                f"You matter, and help is available. ❤️"
            )


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate PHQ-9 + Sentiment inputs
    mock_phq9 = {
        "raw_score":      18,
        "weighted_score": 17.5,
        "severity_index": 72.3,
        "severity_label": "Moderately Severe Depression",
        "responses":      [3, 2, 2, 2, 2, 2, 2, 1, 2],
    }
    mock_sentiment = {
        "sentiment_score":   0.22,
        "depression_signal": "High Depression Signal",
        "has_severe_keywords": False,
    }
    mock_memberships = {
        "Minimal": 0.0, "Mild": 0.0, "Moderate": 0.12,
        "Moderately Severe": 0.88, "Severe": 0.0,
    }

    clf    = DepressionClassifier()
    report = clf.classify(mock_phq9, mock_sentiment, mock_memberships)

    print(f"Final Score:  {report.final_score}")
    print(f"Final Label:  {report.final_label}")
    print(f"Urgency:      {report.urgency}")
    print(f"Crisis:       {report.is_crisis}")
    print(f"\nSummary:\n{report.summary}")
    print(f"\nImmediate Actions:")
    for a in report.immediate_actions:
        print(f"  • {a}")
    print(f"\nBot response:\n{clf.generate_chat_response(report)}")