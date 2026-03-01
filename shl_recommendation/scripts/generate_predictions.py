"""
Generate Predictions on Test Set
Runs all 9 test queries through the recommender and saves CSV.

Run: python scripts/generate_predictions.py
Output: data/firstname_lastname.csv
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommender import SHLRecommender

TEST_PATH = "data/Gen_AI_Dataset.xlsx"
OUTPUT_NAME = "data/firstname_lastname.csv"   # ← CHANGE this to your actual name!
K = 10


def load_test_data(path: str) -> list:
    df = pd.read_excel(path, sheet_name="Test-Set")
    queries = df.iloc[:, 0].dropna().tolist()
    return queries


def generate_predictions(queries: list, recommender: SHLRecommender, k: int = K) -> pd.DataFrame:
    rows = []
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Query: {query[:80]}...")
        recs = recommender.recommend(query, top_k=k)
        for rec in recs:
            rows.append({
                "Query": query,
                "Assessment_url": rec["url"]
            })
    return pd.DataFrame(rows, columns=["Query", "Assessment_url"])


if __name__ == "__main__":
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: Test data not found at {TEST_PATH}")
        sys.exit(1)

    print("Loading recommender...")
    rec = SHLRecommender()
    rec.load()

    print("Loading test queries...")
    queries = load_test_data(TEST_PATH)
    print(f"Loaded {len(queries)} test queries.")

    print("\nGenerating predictions...")
    pred_df = generate_predictions(queries, rec)

    os.makedirs("data", exist_ok=True)
    pred_df.to_csv(OUTPUT_NAME, index=False)
    print(f"\n✅ Saved predictions to {OUTPUT_NAME}")
    print(pred_df.head(20).to_string(index=False))
