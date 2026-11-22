"""Ejecuta un conjunto de evaluación y guarda resultados + configuración (sección 9.5).

Uso:  python eval/run_experiment.py eval/dataset/dev.jsonl [etiqueta]
IMPORTANTE: el conjunto holdout solo se usa al final, sin ajustar nada después de verlo.

Cada línea del conjunto (JSON):
  id, category, question
  expected_decision         answered | abstained | handoff | out_of_scope | unavailable
  expected_route            opcional: structured | documental | handoff | out_of_scope
  expected_facts            textos que deben aparecer en la respuesta (sin distinguir
                            mayúsculas ni acentos); una lista dentro = alternativas
                            (["6", "sexto"]); vacío = no se comprueba
  relevant_chunk_prefixes   prefijos de id de fragmento (p. ej. "programa_ee_networking");
                            vacío = no se mide la recuperación
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from chatbot.api.deps import build_pipeline  # noqa: E402
from chatbot.config import load_settings  # noqa: E402
from chatbot.domain.models import Session  # noqa: E402
from chatbot.domain.text import normalize  # noqa: E402
from chatbot.tls import use_system_certificates  # noqa: E402
from metrics import mrr, rate  # noqa: E402


def evaluate(item: dict, ans) -> dict:
    t = ans.trace
    checks = {"decision": ans.decision.value == item["expected_decision"]}
    if item.get("expected_route"):
        checks["route"] = ans.route.value == item["expected_route"]
    text = normalize(ans.text)
    # un hecho puede ser una lista de alternativas: ["6", "sexto"] (basta una)
    missing = [f for f in item.get("expected_facts", [])
               if not any(normalize(alt) in text for alt in (f if isinstance(f, list) else [f]))]
    if item.get("expected_facts"):
        checks["facts"] = not missing
    out = {"id": item["id"], "category": item.get("category", ""), "question": item["question"],
           "decision": ans.decision.value, "route": ans.route.value, "checks": checks,
           "missing_facts": missing, "answer": ans.text, "trace": t.__dict__}
    prefixes = item.get("relevant_chunk_prefixes", [])
    if prefixes:
        hits = [cid for cid in t.retrieved_ids if any(cid.startswith(p) for p in prefixes)]
        out["retrieval_hit"] = bool(hits)
        out["mrr"] = mrr(t.retrieved_ids, hits) if hits else 0.0
    return out


def main(dataset: str, label: str = "run") -> None:
    use_system_certificates()
    settings = load_settings()
    pipeline = build_pipeline(settings)
    rows = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines()
            if line.strip()]

    results = []
    for item in rows:
        ans = pipeline.answer(item["question"], Session(session_id=item["id"]))  # sesión nueva por pregunta
        r = evaluate(item, ans)
        results.append(r)
        ok = all(r["checks"].values())
        fallo = "" if ok else "  <- " + ", ".join(k for k, v in r["checks"].items() if not v)
        if r["missing_facts"]:
            fallo += f" (faltan: {r['missing_facts']})"
        print(f"{'OK ' if ok else 'MAL'} {r['id']} [{r['category']}] {r['question'][:60]}{fallo}",
              flush=True)

    def summarize(rs: list[dict]) -> dict:
        lat = sorted(r["trace"]["total_ms"] for r in rs)
        hits = [r["retrieval_hit"] for r in rs if "retrieval_hit" in r]
        mrrs = [r["mrr"] for r in rs if "mrr" in r]
        return {
            "n": len(rs),
            "all_checks_ok": rate([all(r["checks"].values()) for r in rs]),
            "decision_accuracy": rate([r["checks"]["decision"] for r in rs]),
            "route_accuracy": rate([r["checks"]["route"] for r in rs if "route" in r["checks"]]),
            "fact_accuracy": rate([r["checks"]["facts"] for r in rs if "facts" in r["checks"]]),
            "retrieval_hit_rate": rate(hits) if hits else None,
            "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
            "latency_p95_ms": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        }

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    summary = {"label": label, **summarize(results),
               "by_category": {c: summarize(rs) for c, rs in sorted(by_cat.items())},
               "config": settings.raw}
    out = ROOT / "eval/results" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False,
                              indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "by_category")},
                     indent=2))
    for cat, s in summary["by_category"].items():
        print(f"  {cat:22} n={s['n']:2}  todo bien={s['all_checks_ok']:.0%}")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "run")
