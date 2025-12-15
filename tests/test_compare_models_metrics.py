import ranker


def test_compare_models_returns_custom_metrics(monkeypatch):
    fake_tfidf = (0.8, {"cv_scores": {"accuracy": [0.7, 0.9], "f1_macro": [0.6, 0.8]}, "means": {"accuracy": 0.8, "f1_macro": 0.7}, "mean": 0.8})
    fake_emb = (0.85, {"cv_scores": {"accuracy": [0.75, 0.95], "f1_micro": [0.7, 0.9]}, "means": {"accuracy": 0.85, "f1_micro": 0.8}, "mean": 0.85})

    monkeypatch.setattr(ranker, "train_basic", lambda texts, labels, save_path, cv=5: fake_tfidf)
    monkeypatch.setattr(ranker, "train_with_embeddings", lambda texts, labels, embed_model_name, save_path, cv=5: fake_emb)

    results = ranker.compare_models(["a", "b", "c", "d"], ["p", "p", "n", "n"], embed_model_name="all-MiniLM-L6-v2", cv=2)
    assert "tfidf" in results and "embedding" in results
    assert results["tfidf"]["means"]["accuracy"] == 0.8
    assert results["embedding"]["means"]["accuracy"] == 0.85
    assert "f1_micro" in results["embedding"]["means"]
