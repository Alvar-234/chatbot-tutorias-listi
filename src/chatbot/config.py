"""Carga de configuración (YAML) y variables de entorno (.env opcional, sin dependencias)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Settings:
    raw: dict
    root: Path

    def __getitem__(self, key: str):
        return self.raw[key]

    def path(self, key: str) -> Path:
        return self.root / self.raw["paths"][key]


def load_settings(path: str | Path | None = None, root: str | Path | None = None) -> Settings:
    root = Path(root or os.environ.get("CHATBOT_ROOT", Path.cwd()))
    _load_dotenv(root / ".env")
    cfg = Path(path or os.environ.get("CHATBOT_CONFIG", root / "config/settings.yaml"))
    with cfg.open(encoding="utf-8") as f:
        return Settings(raw=yaml.safe_load(f), root=root)
