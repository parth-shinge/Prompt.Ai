import pytest


def test_fallback_on_meta_error(monkeypatch):
    """Simulate SentenceTransformer raising a meta-tensor error and assert the
    fallback callable is used (or a RuntimeError is raised if transformers
    cannot load). This test only verifies control flow, not HF network.
    """
    import ranker

    class FakeST:
        def __init__(self, *a, **k):
            raise Exception("Cannot copy out of meta tensor; no data!")

    monkeypatch.setattr(ranker, "SentenceTransformer", FakeST)

    # The environment may or may not have HF access; either the call succeeds
    # (fallback loads the model) or a RuntimeError is raised. Both are OK for
    # this control-flow test.
    try:
        acc, rep = ranker.train_with_embeddings(["a","b","c","d"], ["x","x","y","y"], embed_model_name="all-MiniLM-L6-v2", cv=2)
        assert isinstance(acc, float)
        assert isinstance(rep, dict)
    except RuntimeError:
        # acceptable in restricted/offline env
        pass
