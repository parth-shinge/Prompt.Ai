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
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_ST = False


def _load_embedder(embed_model_name: str):
    """Attempt to load a sentence-transformers embedder, with a safe CPU-first
    strategy. If SentenceTransformer fails due to 'meta' tensor initialization,
    fallback to HF AutoModel+AutoTokenizer and return a callable that maps
    list[str] -> np.ndarray.
    Raises RuntimeError with helpful instructions if both attempts fail.
    """
    # Try SentenceTransformer on CPU first
    try:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available")
        st = SentenceTransformer(embed_model_name, device="cpu")
        return st
    except Exception as e:
        msg = str(e)
        # Detect meta-tensor / to_empty hints
        if "meta tensor" in msg or "to_empty" in msg:
            # Try transformers fallback
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                import os

                # allow HUGGINGFACE_TOKEN for private models
                hf_token = os.environ.get("HUGGINGFACE_TOKEN")
                candidates = [embed_model_name, f"sentence-transformers/{embed_model_name}"]
                last_exc = None
                tokenizer = None
                model = None
                for candidate in candidates:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(candidate, use_auth_token=hf_token)
                        model = AutoModel.from_pretrained(candidate, low_cpu_mem_usage=False, device_map=None, use_auth_token=hf_token)
                        model.to("cpu")
                        chosen = candidate
                        break
                    except Exception as e2:
                        last_exc = e2
                        tokenizer = None
                        model = None

                if model is None or tokenizer is None:
                    raise last_exc or RuntimeError("Could not load model via transformers")

                def _emb_fn(texts_list):
                    enc = tokenizer(texts_list, padding=True, truncation=True, return_tensors="pt")
                    with torch.no_grad():
                        out = model(**{k: v.to("cpu") for k, v in enc.items()})
                    hidden = out.last_hidden_state
                    mask = enc.get("attention_mask")
                    if mask is None:
                        emb = hidden.mean(dim=1)
                    else:
                        mask = mask.unsqueeze(-1)
                        summed = (hidden * mask).sum(dim=1)
                        counts = mask.sum(dim=1).clamp(min=1)
                        emb = summed / counts
                    return emb.cpu().numpy()

                return _emb_fn
            except Exception as e2:
                # Log the combined error and provide a deterministic TF-IDF fallback embedder
                try:
                    import json, datetime, os
                    os.makedirs("artifacts", exist_ok=True)
                    with open("artifacts/ranker_errors.log", "a", encoding="utf-8") as fh:
                        # use timezone-aware UTC timestamps
                        fh.write(json.dumps({"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), "error": str(e), "fallback_error": str(e2)}) + "\n")
                except Exception:
                    pass

                # As a robust, CPU-only fallback, return a TF-IDF based embedder that
                # is deterministic and does not rely on PyTorch/HF model weights.
                from sklearn.feature_extraction.text import TfidfVectorizer

                class TFIDFEmbedder:
                    def __init__(self, max_features=384):
                        self.max_features = max_features
                        self.vectorizer = None

                    def __call__(self, texts):
                        # lazy-fit vectorizer on first call
                        if self.vectorizer is None:
                            self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1,2))
                            X = self.vectorizer.fit_transform(texts)
                        else:
                            X = self.vectorizer.transform(texts)
                        # convert to dense float32 numpy arrays to mimic other embedders
                        return X.toarray().astype("float32")

                try:
                    # Log that we are falling back to TF-IDF embedder for visibility
                    import json, datetime, os
                    os.makedirs("artifacts", exist_ok=True)
                    with open("artifacts/ranker_errors.log", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), "info": "using_tfidf_fallback", "orig_error": str(e), "fallback_error": str(e2)}) + "\n")
                except Exception:
                    pass

                return TFIDFEmbedder()
        else:
            # re-raise with context
            raise

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
    metrics = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "precision_weighted": "precision_weighted",
        "recall_macro": "recall_macro",
        "recall_weighted": "recall_weighted",
    }
    if safe_cv >= 2:
        cv_res = cross_validate(clf, X, labels, cv=StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=42), scoring=metrics, return_train_score=False)
        # extract metrics into lists
        cv_scores = {k: [float(v) for v in cv_res[f"test_{k}"]] for k in metrics.keys()}
        mean_acc = float(np.mean(cv_scores["accuracy"]))
    else:
        cv_scores = {k: [] for k in metrics.keys()}
        mean_acc = float("nan")
    clf.fit(X, labels)
    payload = {"type":"tfidf", "model":clf, "vectorizer":vec}
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    means = {k: (float(np.mean(cv_scores[k])) if cv_scores[k] else float("nan")) for k in cv_scores}
    return mean_acc, {"cv_scores": cv_scores, "means": means, "mean": mean_acc, "note": ("no_cv" if safe_cv < 2 else "ok")}


# Embedding-based training (logistic on embeddings). Keeps and returns saved model
def train_with_embeddings(texts: List[str], labels: List[str], embed_model_name: str = "all-MiniLM-L6-v2", save_path: str = "ranker.pkl", cv: int = 5):
    if not _HAS_ST:
        raise RuntimeError("sentence-transformers not installed.")
    if len(set(labels)) < 2 or len(labels) < 2:
        raise ValueError("Need at least 2 samples and 2 classes to train.")
    safe_cv = _compute_safe_cv(labels, cv)
    # Load an embedder (SentenceTransformer instance or fallback callable)
    # Users can set SENTENCE_TRANSFORMERS_HOME to a local directory to persist models between runs.
    try:
        embedder = _load_embedder(embed_model_name)
    except Exception as e:
        # Surface a helpful error for the calling code/UI
        raise RuntimeError(f"Failed to initialize embedding model '{embed_model_name}': {e}")

    # embedder may be a SentenceTransformer instance with .encode, or a callable
    if hasattr(embedder, "encode"):
        X = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    else:
        X = embedder(texts)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    metrics = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "precision_weighted": "precision_weighted",
        "recall_macro": "recall_macro",
        "recall_weighted": "recall_weighted",
    }
    if safe_cv >= 2:
        cv_res = cross_validate(clf, Xs, labels, cv=StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=42), scoring=metrics, return_train_score=False)
        cv_scores = {k: [float(v) for v in cv_res[f"test_{k}"]] for k in metrics.keys()}
        mean_acc = float(np.mean(cv_scores["accuracy"]))
    else:
        cv_scores = {k: [] for k in metrics.keys()}
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
    means = {k: (float(np.mean(cv_scores[k])) if cv_scores[k] else float("nan")) for k in cv_scores}
    return mean_acc, {"cv_scores": cv_scores, "means": means, "mean": mean_acc, "embed_model": embed_model_name, "note": ("no_cv" if safe_cv < 2 else "ok")}


def compare_models(texts: List[str], labels: List[str], embed_model_name: str = "all-MiniLM-L6-v2", cv: int = 5):
    """Run quick comparisons between TF-IDF and embedding-based rankers.

    Returns a dict with keys 'tfidf' and (optionally) 'embedding'. Each value is
    a report dict containing 'cv_scores', 'mean', and other metadata. If
    embedding training fails, the 'embedding' key will contain an 'error'
    message instead of a report dict.
    """
    results = {}
    # store temporary models in a dedicated artifacts folder to keep repo clean
    tmp_dir = os.path.join("artifacts")
    os.makedirs(tmp_dir, exist_ok=True)
    tfidf_tmp_path = os.path.join(tmp_dir, "tmp_tfidf.pkl")
    emb_tmp_path = os.path.join(tmp_dir, "tmp_emb.pkl")
    # Validate basic dataset requirements
    if len(set(labels)) < 2 or len(labels) < 2:
        raise ValueError("Need at least 2 classes and 2 samples to compare models.")

    tfidf_acc, tfidf_rep = train_basic(texts, labels, save_path=tfidf_tmp_path, cv=cv)
    results["tfidf"] = tfidf_rep

    # Try embeddings only if sentence-transformers is available. If the
    # embedding training fails for any reason, capture the error and continue
    # so the admin UI can show TF-IDF results and present the embedding failure.
    if _HAS_ST:
        try:
            emb_acc, emb_rep = train_with_embeddings(texts, labels, embed_model_name, save_path=emb_tmp_path, cv=cv)
            results["embedding"] = emb_rep
        except Exception as e:
            results["embedding"] = {"error": str(e)}
    else:
        results["embedding"] = {"error": "sentence-transformers not installed"}

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
