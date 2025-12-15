import json
from prompt_generator import _mark_pair_labeled


def test_mark_pair_labeled(tmp_path):
    p = tmp_path / "hybrid_pairs.jsonl"
    entries = [
        {"offline_id": 1, "gemini_id": 2, "created": "t", "labeled": False},
        {"offline_id": 3, "gemini_id": 4, "created": "t", "labeled": False},
    ]
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    # mark the first pair labeled
    _mark_pair_labeled(str(p), 1, 2, 2, "gemini", 42)

    lines = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["labeled"] is True
    assert lines[0]["chosen_id"] == 2
    assert lines[0]["chosen_model"] == "gemini"
    assert int(lines[0]["labeled_by"]) == 42

    # ensure other entries unchanged
    assert lines[1]["labeled"] is False
