import hashlib
import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import yaml
from transformers import TrainingArguments

from bioasq.phase_a.reranker.aliases import QrelsDict, SliceDataset


def get_wandb_run_id(run_name: str) -> str:
    """Deterministic 8-char run ID from run name. Same experiment => same run => updates instead of duplicating."""
    return hashlib.sha256(run_name.encode()).hexdigest()[:8]


def setup_wandb(name: str) -> None:
    # print(os.getcwd())
    os.environ["WANDB_NAME"] = name
    os.environ["WANDB_RUN_ID"] = get_wandb_run_id(name)
    os.environ["WANDB_RESUME"] = "allow"  # Update existing run if same id, else create new

    os.environ["WANDB_API_KEY"] = (
        "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"
    )
    os.environ["WANDB_PROJECT"] = "bioasq-14b-phaseA-reranker"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_ENTITY"] = "bitua"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    # print("If you want to use wandb please change the in setup_wandb function on utils file. And update the bert_config_yaml to report to wandb")


def get_negative_positive_index_from_dataset(dataset: SliceDataset) -> tuple[int, int]:
    _sample = dataset[next(iter(dataset.keys()))]
    relevance_order = [k for k in _sample.keys() if isinstance(k, int)]
    # print(relevance_order)
    return min(relevance_order), max(relevance_order)


def get_relevance_order_from_dataset(dataset: SliceDataset) -> list[int]:
    if len(dataset) == 0:
        return []
    _sample = dataset[next(iter(dataset.keys()))]
    relevance_order: list[int] = sorted(
        [k for k in _sample.keys() if isinstance(k, int)], reverse=True
    )
    # print(relevance_order)
    return relevance_order


class EmptyEncodeBatch:
    def __init__(self):
        self.input_ids: list[int] = []
        self.attention_mask: list[int] = []
        self.token_type_ids: list[int] = []


def split_chunks[T](a: list[T], n: int) -> Generator[list[T], None, None]:
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def load_rank_data(
    bm25_rank_path: str,
    at: int = 1000,
    qrels: QrelsDict | None = None,
) -> SliceDataset:
    dataset: SliceDataset = {}
    with open(bm25_rank_path) as f:
        for line in f:
            q_data: dict[str, str | list[dict[str, str]]] = json.loads(line)  # pyright: ignore[reportAny]
            if qrels:
                if q_data["id"] not in qrels:
                    continue
            dataset[str(q_data["id"])] = {
                "documents": q_data["documents"][:at],
                "question": q_data["question"],
            }
    return dataset


def _flatten(d: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested config dict for TrainingArguments. Handles eval of string booleans."""
    items: list[tuple[str, Any]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(_flatten(v).items())
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        items.extend(_flatten(x).items())
            else:
                # Convert bool-ish strings; keep "none" as-is for report_to etc.
                if isinstance(v, str) and v.lower() in ("true", "false"):
                    v = {"true": True, "false": False}[v.lower()]
                elif isinstance(v, str):
                    try:
                        v = eval(v)
                    except (NameError, TypeError):
                        pass
                items.append((k, v))
    return dict(items)


def _load_flat_config(path: Path | str) -> dict[str, Any]:
    with open(path) as fp:
        config = yaml.safe_load(fp)
    if config is None:
        return {}
    return _flatten(config)


_STRATEGY_KEYS = ("eval_strategy", "logging_strategy", "save_strategy")


def _normalize_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fix YAML quirks: unquoted 'no' becomes False, but HF expects string 'no'."""
    out = dict(config)
    for key in _STRATEGY_KEYS:
        if key in out and out[key] is False:
            out[key] = "no"
    return out


def create_training_config(
    config_path: Path | str,
    **overrides: Any,
) -> TrainingArguments:
    """Load YAML config, flatten, apply overrides, return TrainingArguments."""
    base = _load_flat_config(config_path)
    joint = base | overrides
    joint = _normalize_training_config(joint)
    return TrainingArguments(**joint)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    from transformers.trainer_utils import set_seed as _set_seed

    _set_seed(seed)


def _short_model_name(model_name: str) -> str:
    """Return the last path component for HF model names, else the name as-is."""
    if "://" in model_name or "/" not in model_name:
        return model_name
    return Path(model_name).name


def build_output_dir_name(
    *,
    model_name: str,
    seed: int,
    epoch: int,
    sampler_name: str,
    sample_preprocessing_name: str,
    val: str = "val",
    data: str = "data",
    callback: bool = False,
    num_neg_samples: int = 4,
    gradient_accumulation_steps: int = 2,
    use_expanded_pos: bool = False,
    warmup_ratio: bool = True,
    loss_mode: str,
) -> str:
    """
    Build output directory name following the hf_bert_trainer_v2 pattern.

    Pattern:
        {model}-{seed}-E{epoch}-S{sampler}-SP{preproc}-{val}-{data}_data-CB{callback}-KN{num_neg}
        -GA{grad_accum}-ExPOS{expanded}-warmup{ratio}-{loss_mode}
    """
    _model_identifier = _short_model_name(model_name).replace("/", "-")
    return (
        f"{_model_identifier}-{seed}-E{epoch}"
        f"-S{sampler_name}-SP{sample_preprocessing_name}"
        f"-{val}-{data}_data-CB{callback}-KN{num_neg_samples}"
        f"-GA{gradient_accumulation_steps}-ExPOS{use_expanded_pos}-warmup{warmup_ratio}"
        f"-{loss_mode}"
    )
