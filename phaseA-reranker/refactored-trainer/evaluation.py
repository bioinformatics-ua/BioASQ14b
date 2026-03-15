"""
Inference and evaluation utilities for reranker models.

Runs model over a BioASQ inference dataset, collects scores into a run,
and evaluates with ranx metrics (nDCG@k, MRR, recall@k, map@k, map-bioasq@10).
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch
from ranx import Qrels, Run, evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_METRICS = [
    "ndcg@5",
    "mrr",
    "recall@10",
    "recall@100",
    "recall@1000",
    "map@10",
    "map-bioasq@10",
]


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
    dataloader: DataLoader[dict[str, Any]],
    device: str | torch.device = "cuda",
    tokenizer: PreTrainedTokenizerBase | None = None,
    inspect_samples: int = 0,
    inspect_max_chars: int = 240,
    non_blocking: bool = True,
    amp_dtype: torch.dtype | None = None,
    show_progress: bool = True,
) -> dict[str, dict[str, float]]:
    """Run model over dataloader and build run dict {qid: {doc_id: score}}."""
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        device_obj = torch.device("cpu")

    model = model.to(device_obj)
    model.eval()
    run_dict: dict[str, dict[str, float]] = defaultdict(dict)
    inspected = 0
    use_autocast = (
        device_obj.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
    )

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference", disable=not show_progress):
            inputs = batch["inputs"]
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
                logits = model(**inputs).logits
            logits_cpu = logits.detach().float().cpu()
            scores = extract_scores(logits).detach().float().cpu()

            for i in range(scores.shape[0]):
                qid = (
                    batch["id"][i]
                    if isinstance(batch["id"], list)
                    else str(batch["id"][i])
                )
                doc_id = (
                    batch["doc_id"][i]
                    if isinstance(batch["doc_id"], list)
                    else str(batch["doc_id"][i])
                )
                run_dict[qid][doc_id] = float(scores[i])

                if inspect_samples > 0 and inspected < inspect_samples:
                    decoded = ""
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


def evaluate_run(
    run_dict: dict[str, dict[str, float]],
    qrels_dict: dict[str, dict[str, int]],
    metrics: list[str] | None = None,
    per_file_results: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute ranx metrics for a run against qrels.

    Args:
        run_dict: {qid: {doc_id: score}}
        qrels_dict: {qid: {doc_id: relevance}}
        metrics: List of ranx metric names. Defaults to DEFAULT_METRICS.
        per_file_results: Optional {filename: [qid, ...]} to also report per-file metrics.

    Returns:
        {"total": {...}, "file.json": {...}, ...} or {"total": {...}} if per_file_results is None.
    """
    metrics = metrics or DEFAULT_METRICS
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    results: dict[str, dict[str, float]] = {}
    results["total"] = evaluate(qrels, run, metrics)

    if per_file_results:
        for filename, qids in per_file_results.items():
            subset_qrels = {k: v for k, v in qrels_dict.items() if k in qids}
            subset_run = {k: v for k, v in run_dict.items() if k in qids}
            if subset_qrels and subset_run:
                results[filename] = evaluate(
                    Qrels(subset_qrels), Run(subset_run), metrics
                )

    return results
