from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Small dict wrapper that supports dot-style access for nested config."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return value


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _wrap(raw)


def ensure_dirs(config: Config) -> None:
    for key in ("checkpoints_dir", "reports_dir"):
        Path(config.paths[key]).mkdir(parents=True, exist_ok=True)
