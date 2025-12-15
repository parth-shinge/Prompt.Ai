import json
import os
import types
import builtins
import importlib

import ranker


def test_tfidf_fallback_on_meta_errors(tmp_path, monkeypatch):
    # Simulate SentenceTransformer existing but raising a meta-tensor error on init
    class BadST:
        def __init__(self, *a, **kw):
            raise RuntimeError("Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.")

    monkeypatch.setattr(ranker, "SentenceTransformer", BadST)

    # Simulate transformers AutoTokenizer/AutoModel raising on from_pretrained
    fake_transformers = types.SimpleNamespace()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*a, **kw):
            raise RuntimeError("Cannot copy out of meta tensor; no data! transformer tokenizer fail")

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*a, **kw):
            raise RuntimeError("Cannot copy out of meta tensor; no data! transformer model fail")

    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModel = FakeAutoModel

    monkeypatch.setitem(importlib.sys.modules, "transformers", fake_transformers)

    # Call the loader; it should return a callable TF-IDF embedder
    emb = ranker._load_embedder("all-MiniLM-L6-v2")
    assert callable(emb)

    # Use the embedder on some sample texts
    out = emb(["hello world", "another sample"])
    import numpy as np
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 2

    # Ensure a log entry recorded the fallback usage
    log_path = os.path.join("artifacts", "ranker_errors.log")
    assert os.path.exists(log_path)
    # last line should contain 'using_tfidf_fallback'
    with open(log_path, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh.read().splitlines() if l.strip()]
    assert any("using_tfidf_fallback" in l for l in lines)

def test_train_basic_returns_multiple_metrics(tmp_path):
    # use tokens with length >= 2 so TfidfVectorizer's default token_pattern matches
    texts = ["aa bb cc", "aa bb dd", "xx yy zz", "xx yy qq"]
    labels = ["p", "p", "n", "n"]
    save = str(tmp_path / "t.pkl")
    acc, rep = ranker.train_basic(texts, labels, save_path=save, cv=2)
    assert isinstance(rep.get("cv_scores"), dict)
    for m in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
        assert m in rep["cv_scores"]
        assert isinstance(rep["means"].get(m), float)
