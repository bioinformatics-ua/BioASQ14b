"""Inference and evaluation utilities for reranker models.

Runs model over a BioASQ inference dataset, collects scores into a run,
and evaluates with ranx metrics (nDCG@k, MRR, recall@k, map@k).

Refactored from ``refactored-trainer/evaluation.py``.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from bioasq.common.io import save_json
from bioasq.common.metrics import DEFAULT_RETRIEVAL_METRICS, evaluate_retrieval_run

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.utils.data import DataLoader
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from transformers.modeling_outputs import SequenceClassifierOutput

    from bioasq.common.aliases import RunDict

# Type alias for batches from RankingCollator
type InferenceBatch = dict[str, dict[str, torch.Tensor] | list[str] | list[int]]


def extract_scores(logits: torch.Tensor) -> torch.Tensor:
    """Extract relevance scores from model logits.

    - Single output (num_labels=1): squeeze last dim
    - Binary (num_labels=2): use logits[:, 1] as relevance score
    """
    if logits.shape[-1] == 1:
        return logits.squeeze(-1)
    return logits[:, 1]


def run_inference(
    model: PreTrainedModel,
    dataloader: DataLoader[InferenceBatch],
    device: str | torch.device = "cuda",
    tokenizer: PreTrainedTokenizerBase | None = None,
    inspect_samples: int = 0,
    inspect_max_chars: int = 240,
    non_blocking: bool = True,
    amp_dtype: torch.dtype | None = None,
    show_progress: bool = True,
) -> RunDict:
    """Run model over dataloader and build run dict ``{qid: {doc_id: score}}``."""
    device_obj: torch.device = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        device_obj = torch.device("cpu")

    model = model.to(device_obj)
    model.eval()
    run_dict: dict[str, dict[str, float]] = defaultdict(dict)
    inspected: int = 0
    use_autocast: bool = device_obj.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference", disable=not show_progress):
            inputs: dict[str, torch.Tensor] = batch["inputs"]  # type: ignore[assignment]
            inputs = {
                k: v.to(device_obj, non_blocking=non_blocking)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_autocast
                else nullcontext()
            )
            with autocast_ctx:
                output: SequenceClassifierOutput = model(**inputs)
                logits: torch.Tensor = output.logits
            logits_cpu: torch.Tensor = logits.detach().float().cpu()
            scores: torch.Tensor = extract_scores(logits).detach().float().cpu()

            batch_ids: list[str] = batch["id"]  # type: ignore[assignment]
            batch_doc_ids: list[str] = batch["doc_id"]  # type: ignore[assignment]

            for i in range(scores.shape[0]):
                qid: str = batch_ids[i] if isinstance(batch_ids[i], str) else str(batch_ids[i])
                doc_id: str = (
                    batch_doc_ids[i] if isinstance(batch_doc_ids[i], str) else str(batch_doc_ids[i])
                )
                if not doc_id:  # TODO CHECK THIS
                    continue
                run_dict[qid][doc_id] = float(scores[i])

                if inspect_samples > 0 and inspected < inspect_samples:
                    decoded: str = ""
                    if tokenizer is not None and "input_ids" in inputs:
                        decoded = tokenizer.decode(
                            inputs["input_ids"][i].detach().cpu(),
                            skip_special_tokens=True,
                        )
                        if inspect_max_chars > 0 and len(decoded) > inspect_max_chars:
                            decoded = decoded[:inspect_max_chars] + "..."

                    tqdm.write(
                        "[inspect] "
                        f"idx={inspected} qid={qid} doc_id={doc_id} "
                        f"logits={logits_cpu[i].tolist()} score={float(scores[i]):.6f}"
                    )
                    if decoded:
                        tqdm.write(f"[inspect] decoded_input={decoded}")
                    inspected += 1

    return dict(run_dict)


def save_predictions(
    run_dict: RunDict,
    output_dir: str | Path,
) -> Path:
    """Save predictions (run dict) to ``output_dir/predictions/predictions.json``."""
    pred_dir: Path = Path(output_dir) / "predictions"
    pred_path: Path = pred_dir / "predictions.json"
    save_json(run_dict, pred_path, indent=True)
    return pred_path


def evaluate_run(
    run_dict: RunDict,
    qrels_dict: Mapping[str, Mapping[str, int]],
    metrics: list[str] | None = None,
    per_file_results: Mapping[str, list[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute ranx metrics for a run against qrels.

    Delegates to :func:`bioasq.common.metrics.evaluate_retrieval_run`.
    """
    return evaluate_retrieval_run(
        run_dict,
        qrels_dict,
        metrics=metrics or DEFAULT_RETRIEVAL_METRICS,
        per_file_results=per_file_results,
    )
