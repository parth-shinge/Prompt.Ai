"""
kfold_cv.py

Simple script to run k-fold cross validation comparisons between:
- TF-IDF + LogisticRegression
- Embedding (SentenceTransformer) + LogisticRegression (if available)

It reads choice dataset from database.get_choice_dataset(), runs CV and writes results to CSV.

Usage:
    python kfold_cv.py
"""

import csv
import os
from database import get_choice_dataset
from ranker import compare_models

OUT = "kfold_results.csv"

def main():
    rows = get_choice_dataset()
    if not rows:
        print("No choice dataset available. Generate both variants and choose in the app first.")
        return

    texts, labels = zip(*rows)
    print(f"Found {len(texts)} examples. Running comparison...")

    try:
        # Run for multiple embedding models; aggregate
        EMB_LIST = [
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "paraphrase-MiniLM-L6-v2",
            "paraphrase-MiniLM-L12-v2",
        ]
        all_results = {}
        for em in EMB_LIST:
            res = compare_models(list(texts), list(labels), embed_model_name=em, cv=5)
            all_results[em] = res
    except Exception as e:
        print("compare_models failed:", e)
        return

    # Write to CSV
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["embedding_model", "model", "mean_acc", "details"])
        for emb_name, results in all_results.items():
            for k, v in results.items():
                mean = v.get("mean")
                w.writerow([emb_name, k, mean, str(v)])
    print("Results written to", OUT)

if __name__ == "__main__":
    main()
