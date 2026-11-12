"""Manifiesto de documentos: documento, versión, vigencia, URL (sección 6.1-6.2).

data/raw/manifest.yaml:
  documents:
    - file: reglamento_alumnos.pdf
      doc_id: reglamento_alumnos
      title: "Estatuto de los alumnos"
      version: "2023"
      valid_from: "2023-08-01"
      vigente: true
      url: "https://..."
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_manifest(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    docs = data.get("documents", [])
    for d in docs:
        for key in ("file", "doc_id", "title"):
            if key not in d:
                raise ValueError(f"Falta '{key}' en una entrada del manifiesto: {d}")
    return docs
