"""
Evaluation Script — Mean Recall@K
Uses the labelled Train Set to evaluate recommender performance.

Run: python scripts/evaluate.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommender import SHLRecommender

TRAIN_PATH = "data/Gen_AI_Dataset.xlsx"
K = 10


def load_train_data(path: str) -> dict:
    """
    Load train set from Excel.
    Returns dict: { query: [list of relevant URLs] }
    """
    df = pd.read_excel(path, sheet_name="Train-Set")
    df.columns = ["Query", "Assessment_url"]
    grouped = df.groupby("Query")["Assessment_url"].apply(list).to_dict()
    return grouped


def recall_at_k(relevant: list, predicted: list, k: int = 10) -> float:
    """Recall@K for a single query."""
    if not relevant:
        return 0.0
    predicted_k = predicted[:k]
    hits = sum(1 for url in relevant if url in predicted_k)
    return hits / len(relevant)


def mean_recall_at_k(train_data: dict, recommender: SHLRecommender, k: int = 10) -> dict:
    """
    Compute Mean Recall@K across all train queries.
    Returns dict with per-query scores and overall mean.
    """
    results = []

    for query, relevant_urls in train_data.items():
        print(f"\nEvaluating: {query[:70]}...")
        print(f"  Ground truth: {len(relevant_urls)} assessments")

        recs = recommender.recommend(query, top_k=k)
        predicted_urls = [r["url"] for r in recs]

        recall = recall_at_k(relevant_urls, predicted_urls, k=k)
        print(f"  Predicted {len(predicted_urls)} | Recall@{k}: {recall:.3f}")

        # Show hits and misses
        hits = [u for u in relevant_urls if u in predicted_urls]
        misses = [u for u in relevant_urls if u not in predicted_urls]
        for h in hits:
            print(f"    ✅ {h}")
        for m in misses:
            print(f"    ❌ {m}")

        results.append({
            "query": query,
            "ground_truth_count": len(relevant_urls),
            "predicted_count": len(predicted_urls),
            f"recall@{k}": recall,
            "hits": hits,
            "misses": misses,
            "predicted_urls": predicted_urls,
        })

    mean_recall = np.mean([r[f"recall@{k}"] for r in results])

    print("\n" + "=" * 60)
    print(f"MEAN RECALL@{k}: {mean_recall:.4f}")
    print("=" * 60)

    # Save results
    os.makedirs("data", exist_ok=True)
    results_df = pd.DataFrame([{
        "query": r["query"],
        f"recall@{k}": r[f"recall@{k}"],
        "hits": len(r["hits"]),
        "misses": len(r["misses"]),
    } for r in results])
    results_df.to_csv("data/evaluation_results.csv", index=False)
    print(f"\nSaved results to data/evaluation_results.csv")

    # Save full results as JSON
    with open("data/evaluation_results_full.json", "w") as f:
        json.dump({"mean_recall": mean_recall, "per_query": results}, f, indent=2)

    return {"mean_recall": mean_recall, "per_query": results}


if __name__ == "__main__":
    # Check data exists
    if not os.path.exists(TRAIN_PATH):
        print(f"ERROR: Train data not found at {TRAIN_PATH}")
        print("Please copy Gen_AI_Dataset.xlsx to data/")
        sys.exit(1)

    print("Loading recommender...")
    rec = SHLRecommender()
    rec.load()

    print("Loading train data...")
    train_data = load_train_data(TRAIN_PATH)
    print(f"Loaded {len(train_data)} unique queries.")

    print(f"\nRunning evaluation (Recall@{K})...")
    eval_results = mean_recall_at_k(train_data, rec, k=K)
    print(f"\nFinal Mean Recall@{K}: {eval_results['mean_recall']:.4f}")
