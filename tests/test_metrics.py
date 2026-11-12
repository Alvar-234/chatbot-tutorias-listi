import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from metrics import mrr, rate, recall_at_k  # noqa: E402


def test_metricas():
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5
    assert mrr(["x", "b"], ["b"]) == 0.5
    assert rate([True, False, True, True]) == 0.75
