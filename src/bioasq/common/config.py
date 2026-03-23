"""Configuration loading for the BioASQ pipeline.

Provides typed config loading from YAML files into
:class:`transformers.TrainingArguments`, with flattening and normalisation
of YAML quirks.

Refactored from ``refactored-trainer/utils.py``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

# Recursive type for values coming out of YAML / flattened config dicts.
type ConfigValue = str | int | float | bool | None
type RawConfigValue = ConfigValue | list[RawConfigValue] | dict[str, RawConfigValue]


# ---------------------------------------------------------------------------
# YAML flattening
# ---------------------------------------------------------------------------


def _flatten(d: dict[str, RawConfigValue]) -> dict[str, ConfigValue]:
    """Flatten a nested config dict for :class:`TrainingArguments`.

    Handles nested dicts and lists of dicts.  Converts boolean-ish
    strings (``"true"``/``"false"``) to actual booleans.
    """
    items: list[tuple[str, ConfigValue]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(_flatten(v).items())
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        items.extend(_flatten(x).items())
            else:
                # Convert bool-ish strings
                result: ConfigValue = v
                if isinstance(v, str) and v.lower() in ("true", "false"):
                    result = v.lower() == "true"
                items.append((k, result))
    return dict(items)


def _load_flat_config(path: Path | str) -> dict[str, ConfigValue]:
    """Load a YAML config file and flatten it."""
    with open(path) as fp:
        config: dict[str, RawConfigValue] | None = yaml.safe_load(fp)
    if config is None:
        return {}
    return _flatten(config)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

_STRATEGY_KEYS: tuple[str, ...] = (
    "eval_strategy",
    "logging_strategy",
    "save_strategy",
)


def _normalize_training_config(
    config: dict[str, ConfigValue],
) -> dict[str, ConfigValue]:
    """Fix YAML quirks: unquoted ``no`` becomes ``False``, HF expects ``"no"``."""
    out: dict[str, ConfigValue] = dict(config)
    for key in _STRATEGY_KEYS:
        if key in out and out[key] is False:
            out[key] = "no"
    return out


def create_training_config(
    config_path: Path | str,
    **overrides: ConfigValue,
) -> object:
    """Load a YAML config, flatten, apply overrides, return TrainingArguments.

    Parameters
    ----------
    config_path:
        Path to the YAML training config file.
    **overrides:
        Key-value pairs that override values from the config file.

    Returns
    -------
    :class:`transformers.TrainingArguments` instance.
    """
    from transformers import TrainingArguments

    base: dict[str, ConfigValue] = _load_flat_config(config_path)
    joint: dict[str, ConfigValue] = base | overrides
    joint = _normalize_training_config(joint)
    return TrainingArguments(**joint)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------


def get_wandb_run_id(run_name: str) -> str:
    """Deterministic 8-char run ID from run name.

    Same experiment name → same run → updates instead of duplicating.
    """
    return hashlib.sha256(run_name.encode()).hexdigest()[:8]


def setup_wandb(
    name: str,
    *,
    project: str = "bioasq-14b-phaseA-reranker",
    entity: str = "bitua",
) -> None:
    """Configure Wandb environment variables.

    Parameters
    ----------
    name:
        Human-readable experiment name.
    project:
        Wandb project name.
    entity:
        Wandb team / user entity.
    """
    os.environ["WANDB_NAME"] = name
    os.environ["WANDB_RUN_ID"] = get_wandb_run_id(name)
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_WATCH"] = "false"


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility (delegates to HF)."""
    from transformers.trainer_utils import set_seed as _set_seed

    _set_seed(seed)
