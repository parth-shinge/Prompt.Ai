"""
ranker_retrain.py

Retrain the ranker from the choices dataset and save to ranker.pkl.

Usage:
  python ranker_retrain.py --model all-MiniLM-L6-v2 --out ranker.pkl
  python ranker_retrain.py --tfidf --out ranker.pkl
"""

import argparse
from database import get_choice_dataset
from ranker import train_with_embeddings, train_basic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ranker.pkl")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--tfidf", action="store_true")
    args = p.parse_args()

    rows = get_choice_dataset()
    if not rows:
        print("No choices available. Generate data in the app first.")
        return

    texts, labels = zip(*rows)
    if args.tfidf:
        acc, rep = train_basic(list(texts), list(labels), save_path=args.out)
        print(f"TF-IDF ranker trained. mean acc={acc:.3f}")
        print(rep)
    else:
        acc, rep = train_with_embeddings(list(texts), list(labels), embed_model_name=args.model, save_path=args.out)
        print(f"Embedding ranker trained with {args.model}. mean acc={acc:.3f}")
        print(rep)


if __name__ == "__main__":
    main()


