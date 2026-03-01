# 🎯 SHL Assessment Recommender

An intelligent RAG-based recommendation engine that suggests the most relevant SHL assessments from a catalogue of 377+ Individual Test Solutions, given a natural language query or job description.

> Built as part of the SHL AI Research Engineer Take-Home Assessment.

---

## 🚀 Live Demo

| | Link |
|---|---|
| 🌐 **Web App** | [shl-recommender.streamlit.app](https://your-app.streamlit.app) |
| ⚡ **API** | [shl-api.onrender.com/docs](https://your-api.onrender.com/docs) |

---

## 🏗️ Architecture

```
User Query / Job Description
        │
        ▼
┌─────────────────────┐
│  Query Understanding │  ← Gemini 1.5 Flash extracts skills,
│  (LLM Enrichment)   │    job level, test types, duration limit
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Semantic Retrieval  │  ← Gemini text-embedding-004
│  (Top-30 Candidates) │    + Cosine Similarity
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Duration Filter     │  ← Hard filter on stated time constraints
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  LLM Reranker        │  ← Gemini reranks for relevance + balance
│  + Balance Enforcer  │    (ensures K + P types for mixed queries)
└─────────────────────┘
        │
        ▼
   Top 5–10 Results
```

---

## 📁 Project Structure

```
shl-assessment-recommender/
├── api.py                          # FastAPI backend (/health + /recommend)
├── app.py                          # Streamlit frontend
├── recommender.py                  # Core RAG pipeline
├── requirements.txt
├── render.yaml                     # One-click Render.com deployment
├── .env.example
├── data/
│   ├── shl_assessments.csv         # Scraped catalogue (377+ rows)
│   ├── Gen_AI_Dataset.xlsx         # Train + Test queries
│   └── evaluation_results.csv      # Recall@10 results per query
├── embeddings/
│   └── assessment_embeddings.pkl   # Cached Gemini embeddings
└── scripts/
    ├── scraper.py                  # SHL catalogue scraper
    ├── evaluate.py                 # Mean Recall@K evaluation
    └── generate_predictions.py     # Test set prediction CSV generator
```

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/shl-assessment-recommender.git
cd shl-assessment-recommender
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Add your Gemini API key — get it free at https://ai.google.dev/
```

### 3. Scrape the SHL catalogue

```bash
python scripts/scraper.py
# Creates data/shl_assessments.csv with 377+ assessments
# Takes ~10–15 minutes
```

### 4. Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Docs at: http://localhost:8000/docs
```

### 5. Run the frontend

```bash
streamlit run app.py
# Opens at: http://localhost:8501
```

---

## 🌐 API Reference

### Health Check
```
GET /health
→ {"status": "healthy"}
```

### Recommend Assessments
```
POST /recommend
Content-Type: application/json

{
  "query": "I am hiring Java developers who can collaborate with business teams"
}
```

**Response:**
```json
{
  "recommended_assessments": [
    {
      "name": "Core Java (Entry Level) (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
      "adaptive_support": "No",
      "description": "Multi-choice test measuring Java knowledge...",
      "duration": 13,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    },
    {
      "name": "Occupational Personality Questionnaire (OPQ32r)",
      "url": "https://www.shl.com/...",
      "test_type": ["Personality & Behavior"]
    }
  ]
}
```

---

## 🧪 Evaluation

```bash
# Measure Mean Recall@10 on the labelled train set
python scripts/evaluate.py

# Generate predictions CSV for test set submission
python scripts/generate_predictions.py
```

**Evaluation metric — Mean Recall@10:**

```
Recall@10 = (# relevant assessments in top 10) / (# total relevant)
Mean Recall@10 = average across all queries
```

---

## ☁️ Deployment

### API → Render.com (free)

1. Push to GitHub
2. [New Web Service](https://render.com) → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Add env var: `GEMINI_API_KEY`

### Frontend → Streamlit Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect repo → set main file: `app.py`
3. Add secret: `API_URL = https://your-api.onrender.com`

---

## 🔬 Tech Stack

| Component | Tool | Reason |
|---|---|---|
| Embeddings | Gemini `text-embedding-004` | Free, 768-dim, high quality |
| LLM | Gemini 1.5 Flash | Free tier, fast, long context for JDs |
| Vector Search | Cosine Similarity (NumPy) | No infra needed, fast for ~400 items |
| API | FastAPI | Async, auto docs, Pydantic validation |
| Frontend | Streamlit | Fast deployment, clean UI |
| Scraping | requests + BeautifulSoup | Lightweight, handles SHL's catalogue |

---

## 📊 Key Design Decisions

- **Balance enforcement:** Queries mentioning both technical skills (Java, Python) and soft skills (collaboration, communication) automatically trigger a balance requirement — results always include both `Knowledge & Skills` and `Personality & Behavior` assessments.
- **Two-stage LLM use:** Gemini is used once for query understanding (structured extraction) and once for reranking — keeping latency low while maximising relevance.
- **Embedding cache:** Gemini embeddings for all 377+ assessments are computed once and cached to disk, making subsequent startups instant.
