# 🧠 MindSense — AI-Powered Depression Analysis Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-Deployed-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)

**A clinical-grade depression assessment tool combining PHQ-9, Fuzzy Logic, Genetic Algorithms, and Multi-Modal Sentiment Analysis.**

[Live Demo](#) · [Report Bug](../../issues) · [Request Feature](../../issues)

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [How the AI Works](#-how-the-ai-works)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)
---

## 📖 About the Project

MindSense is a research-grade depression analysis chatbot that assesses mental health using two parallel pipelines:

1. **PHQ-9 Questionnaire** — The clinically validated 9-question Patient Health Questionnaire, scored using a **Fuzzy Logic Inference System** with question weights optimised by a **Genetic Algorithm (GA)**.

2. **Free-text Chat + Voice Input** — The user's typed or spoken messages are analysed using a **4-layer ensemble sentiment model** (TextBlob + VADER + DistilBERT + Clinical Keyword Detection).

Both signals are fused into a final **depression severity score (0–100)** mapped to one of five clinical levels, with tailored recommendations and helpline resources.

> ⚠️ **This tool is for research and educational purposes only. It is not a substitute for professional medical diagnosis or treatment.**

---

## 🛠 Tech Stack

### Backend
| Library | Purpose |
|---|---|
| `FastAPI` | REST API framework |
| `scikit-fuzzy` | Mamdani Fuzzy Inference System for PHQ-9 scoring |
| `DEAP` | Genetic Algorithm for question weight optimisation |
| `TextBlob` | Lexicon-based sentiment analysis |
| `NLTK (VADER)` | Rule-based sentiment, optimised for short/clinical text |
| `Transformers (DistilBERT)` | Deep learning sentiment classification |
| `SpeechRecognition` | Server-side speech-to-text fallback |
| `Uvicorn + Gunicorn` | ASGI production server |

### Frontend
| Technology | Purpose |
|---|---|
| Vanilla HTML/CSS/JS | Single-file frontend, zero build step |
| Web Speech API | Browser-native microphone / voice input |
| Fetch API | Async calls to FastAPI backend |

### Cloud
| Service | Purpose |
|---|---|
| Azure App Service | FastAPI backend hosting |
| Azure Static Web Apps | Frontend hosting (free tier) |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────┐
│                FRONTEND  (index.html)                │
│                                                      │
│   ┌──────────────┐       ┌───────────────────────┐  │
│   │  PHQ-9 Form  │       │  Chat + Mic Input     │  │
│   │  9 questions │       │  (Web Speech API)     │  │
│   │  scored 0–3  │       │                       │  │
│   └──────┬───────┘       └───────────┬───────────┘  │
└──────────┼───────────────────────────┼──────────────┘
           │      POST /analyze/full   │
           └─────────────┬─────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│              FASTAPI BACKEND  (api.py)               │
│                                                      │
│  ┌──────────────────┐    ┌─────────────────────────┐ │
│  │   PHQ-9 Pipeline │    │   Sentiment Pipeline    │ │
│  │                  │    │                         │ │
│  │  Genetic Algo    │    │  TextBlob + VADER        │ │
│  │  (DEAP) Weights  │    │  + DistilBERT           │ │
│  │       ↓          │    │  + Keyword Bank         │ │
│  │  Fuzzy Logic     │    │       ↓                 │ │
│  │  Inference Sys   │    │  Fused Sentiment Score  │ │
│  │  Severity Index  │    │  (0.0 – 1.0)           │ │
│  └────────┬─────────┘    └──────────┬──────────────┘ │
│           └──────────────┬──────────┘                │
│                          ▼                           │
│            Depression Classifier                     │
│         α × PHQ9 + β × Sentiment                    │
│            Final Score (0–100)                       │
│         Severity Label + Recommendations             │
└─────────────────────────────────────────────────────┘
```

---

## ✨ Features

- ✅ **PHQ-9 Clinical Assessment** — All 9 questions with animated progress tracking
- ✅ **Fuzzy Logic Scoring** — Mamdani inference system with 5 overlapping severity membership functions
- ✅ **Genetic Algorithm** — Evolves optimal per-question weights (Q9 — suicidal ideation — weighted highest)
- ✅ **4-Layer Sentiment Analysis** — TextBlob + VADER + DistilBERT Transformer + Clinical keyword bank
- ✅ **Voice Input** — Browser-native Web Speech API mic with live transcript preview
- ✅ **Crisis Detection** — Auto-escalates severity and shows emergency helplines when crisis language is detected
- ✅ **Fuzzy Membership Visualisation** — Animated bars showing membership degree per severity class
- ✅ **Severity Gauge** — Animated 0–100 score bar with dynamic colour coding
- ✅ **Tailored Recommendations** — Immediate actions, lifestyle tips, professional advice, and Indian helplines
- ✅ **REST API** — FastAPI with interactive Swagger UI at `/docs`
- ✅ **Azure Ready** — Dockerfile + startup script included

---

## 📁 Project Structure

```
depression_analysis/
│
├── README.md
├── requirements.txt
│
├── backend/
│   ├── __init__.py
│   ├── phq9_scorer.py            # Fuzzy Logic PHQ-9 Inference System
│   ├── genetic_algorithm.py      # DEAP Genetic Algorithm weight optimiser
│   ├── sentiment_analyzer.py     # 4-layer ensemble sentiment pipeline
│   ├── depression_classifier.py  # Score fusion + recommendation engine
│   ├── api.py                    # FastAPI server (all endpoints)
│   ├── Dockerfile                # Docker container definition
│   └── startup.sh                # Azure App Service startup script
│
├── frontend/
│   └── index.html                # Complete single-file frontend (PHQ-9 + Chat + Mic)
│
└── speech/
    └── speech_handler.py         # Server-side speech-to-text handler
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip
- Chrome or Edge (for microphone / voice input support)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/depression_analysis.git
cd depression_analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 The first run downloads the DistilBERT model (~250 MB). This happens once and is cached locally.

### 3. Start the Backend

```bash
cd backend
python -m uvicorn api:app --reload --port 8000
```

Expected output:
```
INFO: Loading GA-optimised weights...
INFO: Initialising FuzzyPHQ9Scorer...
INFO: Initialising SentimentAnalyzer...
INFO: Initialising DepressionClassifier...
INFO: All modules loaded ✓
INFO: Uvicorn running on http://127.0.0.1:8000
```

### 4. Open the Frontend

Open `frontend/index.html` directly in your browser — **no build step or extra server needed**.

> ✅ Confirm the backend is alive first: visit `http://127.0.0.1:8000/health`
> 📖 Explore all endpoints: `http://127.0.0.1:8000/docs`

### 5. (Optional) Re-train the Genetic Algorithm

To evolve new optimal weights on your own labelled data:

```bash
cd backend
python genetic_algorithm.py
```

Copy the printed `best_weights` list into `get_default_weights()` in `genetic_algorithm.py`.

---

## 📡 API Reference

Base URL: `http://127.0.0.1:8000` (local) · `https://depression-api.azurewebsites.net` (Azure)

Full interactive docs: `/docs`

### `GET /health`
```json
{ "status": "ok", "modules": ["phq9_scorer", "sentiment_analyzer", "classifier"] }
```

### `POST /analyze/phq9`
Score PHQ-9 responses using fuzzy logic + GA weights.

**Request body:**
```json
{ "responses": [1, 2, 1, 0, 2, 1, 1, 0, 0] }
```

**Response:**
```json
{
  "raw_score": 8,
  "weighted_score": 7.94,
  "severity_index": 34.2,
  "severity_label": "Mild Depression",
  "severity_pct": "34.2%",
  "memberships": {
    "Minimal": 0.0, "Mild": 0.88, "Moderate": 0.12,
    "Moderately Severe": 0.0, "Severe": 0.0
  },
  "weights_used": [1.0, 1.1, 0.9, 0.85, 0.95, 1.05, 1.0, 1.2, 1.5]
}
```

### `POST /analyze/chat`
Sentiment analysis on a single chat message.

**Request body:**
```json
{ "message": "I feel completely hopeless and exhausted all the time." }
```

**Response:**
```json
{
  "sentiment_score": 0.18,
  "depression_signal": "High Depression Signal",
  "has_severe_keywords": false,
  "textblob_polarity": -0.42,
  "vader_compound": -0.71
}
```

### `POST /analyze/full`
Full combined analysis — PHQ-9 + chat history + recommendations.

**Request body:**
```json
{
  "phq9_responses": [2, 3, 2, 2, 1, 2, 1, 1, 2],
  "chat_history": ["I've been feeling really low", "Nothing seems to help"],
  "latest_message": "I don't see any point anymore"
}
```

**Response includes:** `final_score`, `final_label`, `urgency`, `is_crisis`, `bot_response`, `immediate_actions`, `lifestyle_tips`, `professional_advice`, `resources`, `memberships`, `chat_trajectory`.

### `POST /speech/transcribe`
Upload audio file for server-side transcription.

```bash
curl -X POST "http://127.0.0.1:8000/speech/transcribe" \
  -F "audio=@recording.wav"
```

## 🤖 How the AI Works

### Genetic Algorithm — Weight Optimisation
Standard PHQ-9 weights all 9 questions equally. Our GA (via `DEAP`) evolves a population of 100 weight vectors over 50 generations to minimise classification error. Clinically critical items — especially Q9 (suicidal ideation) — receive higher evolved weights.

| GA Parameter | Value |
|---|---|
| Population | 100 individuals |
| Generations | 50 |
| Crossover | Uniform (p = 0.7) |
| Mutation | Gaussian (σ = 0.1, p = 0.2) |
| Selection | Tournament (k = 3) |

### Fuzzy Logic — PHQ-9 Severity Scoring
The GA-weighted PHQ-9 score (0–27) passes through a **Mamdani fuzzy inference system** with five overlapping membership functions. The crisp output (0–100 severity index) is produced by centroid defuzzification.

### Sentiment Analysis — 4-Layer Ensemble

| Layer | Weight | Purpose |
|---|---|---|
| DistilBERT Transformer | 35% | Deep learning, context-aware |
| VADER (NLTK) | 30% | Clinical-term lexicon boosted |
| Clinical Keyword Bank | 20% | Detects crisis / severe language |
| TextBlob | 15% | Baseline lexicon polarity |

Crisis keywords (e.g., "suicidal", "end my life") automatically cap the sentiment score and escalate the final label to at least *Moderately Severe*.

### Score Fusion

```
Final Score = 0.65 × PHQ9_Severity_Index + 0.35 × Sentiment_Depression_Score
```

PHQ-9 carries higher weight (0.65) as the clinical gold standard. The sentiment signal adds supplemental context from the user's own words.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open a Pull Request

### Ideas
- [ ] Persistent session storage (Azure Cosmos DB)
- [ ] Multilingual support — Hindi, Marathi, Tamil
- [ ] Expand GA training dataset with real clinical data
- [ ] Session history charts (score over time)
- [ ] Mobile app wrapper (React Native / Flutter)

---

## ⚠️ Disclaimer

This application is developed **strictly for research and educational purposes**.

- It is **not a medical device** and must not be used for clinical diagnosis.
- It does **not replace** a licensed mental health professional.
- If you or someone you know is in crisis, please contact a helpline immediately:

| Helpline | Number | Hours |
|---|---|---|
| iCall (India) | 9152987821 | Mon–Sat 8am–10pm |
| AASRA | 9820466627 | 24/7 |
| Vandrevala Foundation | 1860-2662-345 | 24/7 |
| iMumps | 9373111709 | 24/7 |

---

## 👤 Author

**Your Name**
- GitHub: [Priyanshu Jha](https://github.com/Priyanshu0423) , [Dhruv Marwal](https://github.com/DhruvMarwal) , [Shivang Jain](https://github.com/Xopse)
- LinkedIn: [Priyanshu Jha](https://linkedin.com/in/priyanshujha-) , [Dhruv Marwal](https://linkedin.com/in/dhruvmarwal) , [Shivang Jain](https://linkedin.com/in/shivang-jain-69602132a)

---

<div align="center">
Made with ❤️ for mental health awareness
<br/><br/>
⭐ Star this repo if you found it useful!
</div>
