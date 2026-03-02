"""
SHL Assessment Recommender — RAG Pipeline
Uses Google Gemini embeddings + cosine similarity for semantic search.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATA_PATH = os.getenv("DATA_PATH", "data/shl_assessments.csv")
EMBEDDINGS_CACHE = "/tmp/assessment_embeddings.pkl"
EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=GEMINI_API_KEY)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_text(text):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return result.embeddings[0].values


def embed_query(text):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


def load_assessments():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Assessment data not found at '{DATA_PATH}'. Run: python scripts/scraper.py")
    df = pd.read_csv(DATA_PATH)
    df = df.fillna("")
    df["embed_text"] = (
        df["name"].astype(str) + ". " +
        df["description"].astype(str) + " " +
        "Test type: " + df["test_type"].astype(str) + ". " +
        "Duration: " + df["duration"].astype(str) + " minutes."
    )
    return df


def build_embeddings(df, force=False):
    os.makedirs("embeddings", exist_ok=True)
    if os.path.exists(EMBEDDINGS_CACHE) and not force:
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}")
        with open(EMBEDDINGS_CACHE, "rb") as f:
            cached = pickle.load(f)
        if len(cached) == len(df):
            return np.array(cached)
        print("Cache size mismatch — rebuilding.")

    print(f"Building embeddings for {len(df)} assessments...")
    embeddings = []
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Embedding {i+1}/{len(df)}...")
        import time; time.sleep(0.7); emb = embed_text(row["embed_text"])
        embeddings.append(emb)

    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {EMBEDDINGS_CACHE}")
    return np.array(embeddings)


SYSTEM_PROMPT = """You are an expert at understanding HR and recruitment queries.
Given a job description or natural language query, extract:
1. Required skills (technical and soft)
2. Job level (graduate, mid-level, senior, manager, executive)
3. Test types needed — choose from: Ability & Aptitude, Knowledge & Skills, Personality & Behavior, Competencies, Simulations, Biodata & Situational Judgement
4. Any duration constraints (max minutes)

Return ONLY valid JSON like:
{
  "skills": ["Python", "SQL"],
  "job_level": "mid-level",
  "test_types": ["Knowledge & Skills", "Personality & Behavior"],
  "max_duration": 60,
  "enriched_query": "full enriched search string"
}"""


def understand_query(query):
    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=f"{SYSTEM_PROMPT}\n\nQuery: {query}",
            config=types.GenerateContentConfig(temperature=0.1)
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Query understanding failed: {e}")
        return {"enriched_query": query, "test_types": [], "max_duration": None}


def llm_rerank(query, candidates, top_k=10):
    candidate_text = "\n".join([
        f"{i+1}. [{c['name']}] type={c.get('test_type','')} duration={c.get('duration','')}min"
        for i, c in enumerate(candidates)
    ])
    prompt = f"""You are an SHL assessment expert. Given this query:
"{query}"

Rank the following assessments from most to least relevant.
Important: If the query needs both technical AND behavioral skills, ensure BALANCE — include both Knowledge & Skills (K) AND Personality & Behavior (P) types.
Return ONLY a JSON array of the numbers (1-based index) in order of relevance, max {top_k} items.
Example: [3, 1, 7, 2, 5]

Assessments:
{candidate_text}

Return ONLY the JSON array, nothing else."""

    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        indices = json.loads(text.strip())
        reranked = []
        for idx in indices:
            if 1 <= idx <= len(candidates):
                reranked.append(candidates[idx - 1])
        return reranked[:top_k]
    except Exception as e:
        print(f"Reranking failed: {e}")
        return candidates[:top_k]


class SHLRecommender:
    def __init__(self):
        self.df = None
        self.embeddings = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print("Loading SHL assessments...")
        self.df = load_assessments()
        print(f"Loaded {len(self.df)} assessments.")
        self.embeddings = build_embeddings(self.df)
        self._loaded = True
        print("Recommender ready!")

    def recommend(self, query, top_k=10, use_llm_rerank=True):
        if not self._loaded:
            self.load()

        print(f"\nQuery: {query[:80]}...")
        parsed = understand_query(query)
        enriched = parsed.get("enriched_query", query)
        max_duration = parsed.get("max_duration")
        print(f"Enriched: {enriched[:100]}")
        print(f"Types: {parsed.get('test_types')}, max_duration: {max_duration}")

        q_emb = embed_query(enriched)
        scores = [cosine_similarity(q_emb, self.embeddings[i]) for i in range(len(self.df))]
        self.df["_score"] = scores

        filtered = self.df.copy()
        if max_duration:
            mask = (
                (filtered["duration"] == "") |
                (filtered["duration"].astype(str) == "nan") |
                (pd.to_numeric(filtered["duration"], errors="coerce").fillna(9999) <= max_duration)
            )
            filtered = filtered[mask]

        top_candidates = filtered.nlargest(30, "_score")
        candidates = top_candidates.to_dict("records")

        if use_llm_rerank and candidates:
            results = llm_rerank(query, candidates, top_k=top_k)
        else:
            results = candidates[:top_k]

        output = []
        for r in results:
            duration_val = r.get("duration", "")
            try:
                duration_int = int(float(duration_val)) if duration_val and str(duration_val) != "nan" else None
            except (ValueError, TypeError):
                duration_int = None

            test_types = r.get("test_type", "")
            if isinstance(test_types, str) and test_types:
                test_types_list = [t.strip() for t in test_types.split(",") if t.strip()]
            elif isinstance(test_types, list):
                test_types_list = test_types
            else:
                test_types_list = []

            output.append({
                "url": r.get("url", ""),
                "name": r.get("name", ""),
                "description": r.get("description", "")[:300],
                "duration": duration_int,
                "remote_support": r.get("remote_support", "No"),
                "adaptive_support": r.get("adaptive_support", "No"),
                "test_type": test_types_list,
            })
        return output


_recommender_instance: Optional[SHLRecommender] = None


def get_recommender() -> SHLRecommender:
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = SHLRecommender()
        _recommender_instance.load()
    return _recommender_instance


if __name__ == "__main__":
    rec = SHLRecommender()
    rec.load()
    results = rec.recommend("I am hiring for Java developers who can also collaborate effectively with my business teams.")
    for r in results:
        print(f"  - {r['name']} ({r['url']})")
