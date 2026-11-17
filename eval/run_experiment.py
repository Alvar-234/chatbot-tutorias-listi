"""Ejecuta un conjunto de evaluación y guarda resultados + configuración (sección 9.5).

Uso:  python eval/run_experiment.py eval/dataset/dev.jsonl [etiqueta]
IMPORTANTE: el conjunto holdout solo se usa al final, sin ajustar nada después de verlo.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from chatbot.api.deps import build_pipeline  # noqa: E402
from chatbot.config import load_settings  # noqa: E402
from chatbot.domain.models import Session  # noqa: E402
from chatbot.tls import use_system_certificates  # noqa: E402
from metrics import mrr, rate, recall_at_k  # noqa: E402


def main(dataset: str, label: str = "run") -> None:
    use_system_certificates()
    settings = load_settings()
    pipeline = build_pipeline(settings)
    rows = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]

    results, decision_ok, fact_ok, recalls, mrrs, latencies = [], [], [], [], [], []
    for item in rows:
        ans = pipeline.answer(item["question"], Session(session_id=item["id"]))  # sesión nueva por pregunta
        t = ans.trace
        decision_ok.append(ans.decision.value == item["expected_decision"])
        if item.get("expected_facts"):
            fact_ok.append(all(f in ans.text for f in item["expected_facts"]))
        prefixes = item.get("relevant_chunk_prefixes", [])
        if prefixes:
            hits = [cid for cid in t.retrieved_ids if any(cid.startswith(p) for p in prefixes)]
            recalls.append(1.0 if hits else 0.0)
            mrrs.append(mrr(t.retrieved_ids, hits) if hits else 0.0)
        latencies.append(t.total_ms)
        results.append({"id": item["id"], "decision": ans.decision.value, "trace": t.__dict__})

    latencies.sort()
    summary = {
        "label": label, "n": len(rows),
        "decision_accuracy": rate(decision_ok),
        "structured_fact_accuracy": rate(fact_ok),
        "retrieval_hit_rate": rate([r == 1.0 for r in recalls]),
        "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        "latency_p95_ms": latencies[int(0.95 * (len(latencies) - 1))] if latencies else None,
        "config": settings.raw,
    }
    out = ROOT / "eval/results" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "run")
