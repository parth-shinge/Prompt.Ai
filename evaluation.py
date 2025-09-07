"""
evaluation.py

Runs evaluation on the current ranker against the choices dataset.
Computes accuracy, confusion matrix, and simple significance test versus random.
Saves a plot of confusion matrix.

Usage:
  python evaluation.py --ranker ranker.pkl --out report.json
"""

import argparse
import json
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from database import get_choice_dataset
from ranker import predict_with_ranker


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return float(correct) / max(1, len(y_true))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranker", default="ranker.pkl")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    rows = get_choice_dataset()
    if not rows:
        print("No dataset available.")
        return
    texts, labels = zip(*rows)

    preds = []
    for t in texts:
        pred, _ = predict_with_ranker(t, args.ranker)
        if pred not in ("offline", "gemini"):
            pred = np.random.choice(["offline", "gemini"])  # fallback
        preds.append(pred)

    acc = compute_accuracy(list(labels), preds)

    # confusion matrix
    classes = ["offline", "gemini"]
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((2, 2), dtype=int)
    for y, p in zip(labels, preds):
        cm[idx[y], idx[p]] += 1

    # simple binomial test vs random 0.5 (approx using normal if n large)
    n = len(labels)
    successes = int(acc * n)
    # normal approx z-test
    p0 = 0.5
    var = n * p0 * (1 - p0)
    z = (successes - n * p0) / np.sqrt(var) if var > 0 else 0.0

    report = {
        "accuracy": acc,
        "num_samples": n,
        "z_vs_random": float(z),
        "confusion_matrix": cm.tolist(),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Saved report to", args.out)

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()


