"""Configuration loading and saving utilities."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a SimpleNamespace tree to a plain dictionary."""
    if isinstance(cfg, SimpleNamespace):
        return {key: to_dict(value) for key, value in vars(cfg).items()}
    if isinstance(cfg, dict):
        return {key: to_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, list):
        return [to_dict(item) for item in cfg]
    return cfg


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a YAML config with optional recursive defaults inheritance."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    defaults = data.pop("defaults", None)
    if defaults is not None:
        parent_path = (config_path.parent / defaults).resolve()
        parent_cfg = to_dict(load_config(parent_path))
        data = _deep_merge(parent_cfg, data)

    return _to_namespace(data)


def save_config(cfg: SimpleNamespace | dict[str, Any], path: str | Path) -> None:
    """Save a resolved configuration to YAML."""
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(cfg), handle, sort_keys=False, allow_unicode=True)
