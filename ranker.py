"""
Simple ranker utilities.

Provides:
- train_basic(texts, labels) -> trains TF-IDF + LogisticRegression, returns (cv_acc, report)
- train_with_embeddings(texts, labels, embed_model_name, save_path) -> uses sentence-transformers to embed then LogisticRegression
- compare_models(texts, labels) -> cross-validated results for TF-IDF and embeddings (if available)
- load_ranker(path), predict_with_ranker(text, ranker_path)
"""

import os
import pickle
from typing import List, Tuple, Dict, Any, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_ST = False

# Basic TF-IDF + logistic training
def _compute_safe_cv(labels: List[str], requested_cv: int) -> int:
    n_samples = len(labels)
    class_counts = [labels.count(c) for c in set(labels)]
    min_per_class = min(class_counts) if class_counts else 0
    # Need at least 2 folds, each fold must contain at least 1 sample from every class
    max_cv = min(n_samples, min_per_class) if min_per_class > 0 else 0
    if max_cv < 2:
        return 0  # indicates CV not feasible
    return max(2, min(requested_cv, max_cv))


def train_basic(texts: List[str], labels: List[str], save_path: str = "ranker.pkl", cv: int = 5):
    # Adjust CV if dataset is small
    if len(set(labels)) < 2 or len(labels) < 2:
        raise ValueError("Need at least 2 samples and 2 classes to train.")
    safe_cv = _compute_safe_cv(labels, cv)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    if safe_cv >= 2:
        cv_scores = cross_val_score(clf, X, labels, cv=StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=42), scoring="accuracy")
        mean_acc = float(np.mean(cv_scores))
    else:
        cv_scores = []
        mean_acc = float("nan")
    clf.fit(X, labels)
    payload = {"type":"tfidf", "model":clf, "vectorizer":vec}
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    return mean_acc, {"cv_scores": [float(s) for s in cv_scores], "mean": mean_acc, "note": ("no_cv" if safe_cv < 2 else "ok")}


# Embedding-based training (logistic on embeddings). Keeps and returns saved model
def train_with_embeddings(texts: List[str], labels: List[str], embed_model_name: str = "all-MiniLM-L6-v2", save_path: str = "ranker.pkl", cv: int = 5):
    if not _HAS_ST:
        raise RuntimeError("sentence-transformers not installed.")
    if len(set(labels)) < 2 or len(labels) < 2:
        raise ValueError("Need at least 2 samples and 2 classes to train.")
    safe_cv = _compute_safe_cv(labels, cv)
    # Cache models under local folder if HF cache not desired; by default SentenceTransformer caches to ~/.cache
    # Users can set SENTENCE_TRANSFORMERS_HOME to a local directory to persist models between runs.
    # Try to load the SentenceTransformer explicitly on CPU first to avoid
    # 'meta' tensor initialization issues which can occur when transformers
    # uses lazy weight initialization (device_map / low_cpu_mem_usage).
    try:
        embedder = SentenceTransformer(embed_model_name, device="cpu")
        use_sentence_transformer = True
    except Exception as e:
        # Common failing symptom: "Cannot copy out of meta tensor; no data!"
        # Attempt a deterministic fallback that uses Hugging Face transformers
        # directly (AutoTokenizer + AutoModel) loaded onto CPU with
        # low_cpu_mem_usage=False and device_map=None to avoid meta device lazy init.
        msg = str(e)
        use_sentence_transformer = False
        if "meta tensor" in msg or "to_empty" in msg:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
                # Force non-lazy full weights load onto CPU
                model = AutoModel.from_pretrained(embed_model_name, low_cpu_mem_usage=False, device_map=None)
                model.to("cpu")

                def _embeddings_from_transformers(texts_list):
                    # Tokenize and compute mean-pooled embeddings like SentenceTransformers
                    enc = tokenizer(texts_list, padding=True, truncation=True, return_tensors="pt")
                    with torch.no_grad():
                        out = model(**{k: v.to("cpu") for k, v in enc.items()})
                    # Prefer `last_hidden_state`; for some models pooled output may exist
                    hidden = out.last_hidden_state
                    mask = enc.get("attention_mask")
                    if mask is None:
                        # simple mean pool over sequence dim
                        emb = hidden.mean(dim=1)
                    else:
                        mask = mask.unsqueeze(-1)
                        summed = (hidden * mask).sum(dim=1)
                        counts = mask.sum(dim=1).clamp(min=1)
                        emb = summed / counts
                    return emb.cpu().numpy()

                embedder = _embeddings_from_transformers
                use_sentence_transformer = False
            except Exception as e2:
                # If fallback also fails, raise an informative error describing both failures
                raise RuntimeError(
                    "Failed to load embedding model with SentenceTransformer and fallback to transformers failed as well. "
                    f"Original error: {msg}. Fallback error: {e2}. Try upgrading: `pip install --upgrade sentence-transformers transformers accelerate torch`"
                )
        else:
            # Not obviously a meta-device error; re-raise to surface the original traceback
            raise
    # embedder is either a SentenceTransformer instance (with `.encode`) or a
    # fallback callable that returns numpy arrays for a list of texts.
    if hasattr(embedder, "encode"):
        X = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    else:
        # our fallback callable
        X = embedder(texts)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    if safe_cv >= 2:
        cv_scores = cross_val_score(clf, Xs, labels, cv=StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=42), scoring="accuracy")
        mean_acc = float(np.mean(cv_scores))
    else:
        cv_scores = []
        mean_acc = float("nan")
    clf.fit(Xs, labels)
    payload = {
        "type":"embedding",
        "model":clf,
        "embed_model_name":embed_model_name,
        "scaler":scaler
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    return mean_acc, {"cv_scores": [float(s) for s in cv_scores], "mean": mean_acc, "embed_model": embed_model_name, "note": ("no_cv" if safe_cv < 2 else "ok")}


def compare_models(texts: List[str], labels: List[str], embed_model_name: str = "all-MiniLM-L6-v2", cv: int = 5):
    results = {}
    # store temporary models in a dedicated artifacts folder to keep repo clean
    tmp_dir = os.path.join("artifacts")
    os.makedirs(tmp_dir, exist_ok=True)
    tfidf_tmp_path = os.path.join(tmp_dir, "tmp_tfidf.pkl")
    emb_tmp_path = os.path.join(tmp_dir, "tmp_emb.pkl")

    tfidf_acc, tfidf_rep = train_basic(texts, labels, save_path=tfidf_tmp_path, cv=cv)
    results["tfidf"] = tfidf_rep
    if _HAS_ST:
        emb_acc, emb_rep = train_with_embeddings(texts, labels, embed_model_name, save_path=emb_tmp_path, cv=cv)
        results["embedding"] = emb_rep
    return results


def load_ranker(path: str = "ranker.pkl") -> Dict[str, Any]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload


def predict_with_ranker(text: Union[str, List[str]], ranker_path: str = "ranker.pkl"):
    payload = load_ranker(ranker_path)
    if not payload:
        return None, "no_ranker"
    typ = payload.get("type")
    texts: List[str] = [text] if isinstance(text, str) else list(text)
    if typ == "tfidf":
        vec = payload["vectorizer"]
        clf = payload["model"]
        X = vec.transform(texts)
        probs_list = clf.predict_proba(X)
        classes = clf.classes_
        # return first if single
        if len(texts) == 1:
            probs = probs_list[0]
            idx = int(np.argmax(probs))
            return classes[idx], float(probs[idx])
        # else return list of (label, prob)
        out = []
        for probs in probs_list:
            idx = int(np.argmax(probs))
            out.append((classes[idx], float(probs[idx])))
        return out, None
    elif typ == "embedding":
        if not _HAS_ST:
            return None, "embedding_unavailable"
        emname = payload["embed_model_name"]
        embedder = SentenceTransformer(emname)
        clf = payload["model"]
        scaler = payload["scaler"]
        emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embs = scaler.transform(emb)
        probs_list = clf.predict_proba(embs)
        classes = clf.classes_
        if len(texts) == 1:
            probs = probs_list[0]
            idx = int(np.argmax(probs))
            return classes[idx], float(probs[idx])
        out = []
        for probs in probs_list:
            idx = int(np.argmax(probs))
            out.append((classes[idx], float(probs[idx])))
        return out, None
    else:
        return None, "unknown_ranker_type"
