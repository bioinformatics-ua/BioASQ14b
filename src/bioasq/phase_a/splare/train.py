"""Training script for SPLARE (KL distillation + FLOPS sparsity).

Follows the paper's training procedure:
  1. Bidirectional pretraining (Masked Next Token Prediction) — optional
  2. KL distillation from a cross-encoder teacher with FLOPS regularization
  3. LoRA adapters on backbone, SAE encoder frozen

Usage::

    python -m bioasq.phase_a.splare.train \
        --training-data data/training/negatives.jsonl \
        --teacher-model nvidia/Llama-3.1-Nemotron-70B-Instruct \
        --output-dir data/splare/checkpoints
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bioasq.phase_a.splare.model import SplareConfig, SplareModel

# ---------------------------------------------------------------------------
# FLOPS regularization (from SPLADE)
# ---------------------------------------------------------------------------


def flops_loss(reps: torch.Tensor) -> torch.Tensor:
    """FLOPS regularization: penalises the mean activation per feature.

    Encourages sparsity by minimising the squared sum of average activations
    across the batch, following Paria et al. (2020) / SPLADE.

    Args:
        reps: Sparse representations ``(B, W)`` — already pooled.
    """
    # Mean activation per feature across the batch
    mean_act = reps.mean(dim=0)  # (W,)
    return (mean_act**2).sum()


# ---------------------------------------------------------------------------
# KL distillation loss
# ---------------------------------------------------------------------------


def kl_distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence between teacher and student relevance distributions.

    Args:
        student_scores: ``(B, M)`` — student dot-product scores for M documents.
        teacher_scores: ``(B, M)`` — teacher scores for the same documents.
        temperature: Softmax temperature (default 1.0).
    """
    log_student = F.log_softmax(student_scores / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_scores / temperature, dim=-1)
    return F.kl_div(log_student, teacher_probs, reduction="batchmean") * (temperature**2)


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------


@dataclass
class SplareTrainingSample:
    query: str
    positives: list[str]
    negatives: list[str]
    teacher_scores: list[float] | None = None


class SplareTrainingDataset(Dataset):
    """Loads training data from BioASQ negatives JSONL format.

    Expected JSONL format per line::

        {
            "body": "question text",
            "pos_docs": [{"text": "..."}],
            "neg_docs": [{"full_text": "..."}],
            "teacher_scores": [0.9, 0.1, ...]  // optional
        }
    """

    def __init__(self, path: Path) -> None:
        import msgspec

        decoder = msgspec.json.Decoder()
        self.samples: list[SplareTrainingSample] = []
        with open(path, "rb") as f:
            for line in f:
                obj = decoder.decode(line)
                self.samples.append(
                    SplareTrainingSample(
                        query=obj["body"],
                        positives=[d.get("text", d.get("full_text", "")) for d in obj.get("pos_docs", [])],
                        negatives=[d.get("full_text", d.get("text", "")) for d in obj.get("neg_docs", [])],
                        teacher_scores=obj.get("teacher_scores"),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SplareTrainingSample:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train_splare(
    training_data_path: Path,
    output_dir: Path,
    *,
    backbone: str = "meta-llama/Llama-3.1-8B",
    sae_path: str = "",
    sae_layer: int = 26,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 2,
    grad_accum_steps: int = 64,
    lambda_q: float = 1e-4,
    lambda_d: float = 1e-4,
    temperature: float = 80.0,
    query_topk: int = 40,
    doc_topk: int = 400,
    max_negatives: int = 7,
    device: str = "cuda",
    wandb_project: str | None = None,
) -> None:
    """Train a SPLARE model with KL distillation and FLOPS regularization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SplareConfig(
        backbone_name_or_path=backbone,
        sae_name_or_path=sae_path,
        sae_layer=sae_layer,
        query_topk=query_topk,
        doc_topk=doc_topk,
    )
    model = SplareModel(config).load(device).apply_lora()

    dataset = SplareTrainingDataset(training_data_path)
    print(f"Loaded {len(dataset)} training samples")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = (len(dataset) * epochs) // (batch_size * grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=vars(config))

    model.backbone.train()
    global_step = 0

    for epoch in range(epochs):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(pbar):
            # Build text lists: query + positive + negatives
            queries = [s.query for s in batch]
            all_docs: list[str] = []
            n_docs_per_query: list[int] = []

            for sample in batch:
                docs = sample.positives[:1] + sample.negatives[:max_negatives]
                all_docs.extend(docs)
                n_docs_per_query.append(len(docs))

            # Tokenize
            q_inputs = model.tokenizer(
                queries, return_tensors="pt", padding=True, truncation=True,
                max_length=config.max_length,
            ).to(device)
            d_inputs = model.tokenizer(
                all_docs, return_tensors="pt", padding=True, truncation=True,
                max_length=config.max_length,
            ).to(device)

            # Forward pass (with gradients through backbone via LoRA)
            q_hidden = model._extract_hidden_states(q_inputs["input_ids"], q_inputs["attention_mask"])
            q_sae = model.sae(q_hidden)
            q_pooled = model._splade_pool(q_sae, q_inputs["attention_mask"])

            d_hidden = model._extract_hidden_states(d_inputs["input_ids"], d_inputs["attention_mask"])
            d_sae = model.sae(d_hidden)
            d_pooled = model._splade_pool(d_sae, d_inputs["attention_mask"])

            # Compute student scores via sparse dot product
            # Split d_pooled by query
            offset = 0
            student_scores_list = []
            teacher_scores_list = []
            max_docs = max(n_docs_per_query)

            for i, n_docs in enumerate(n_docs_per_query):
                q_vec = q_pooled[i]  # (W,)
                d_vecs = d_pooled[offset : offset + n_docs]  # (n_docs, W)

                scores = (q_vec.unsqueeze(0) * d_vecs).sum(dim=-1)  # (n_docs,)

                # Pad to max_docs for batched KL loss
                if n_docs < max_docs:
                    pad = torch.full((max_docs - n_docs,), float("-inf"), device=device)
                    scores = torch.cat([scores, pad])

                student_scores_list.append(scores)

                # Teacher scores (if available, otherwise use 1-hot)
                sample = batch[i]
                if sample.teacher_scores and len(sample.teacher_scores) >= n_docs:
                    t_scores = torch.tensor(
                        sample.teacher_scores[:n_docs], device=device, dtype=torch.float32,
                    )
                else:
                    t_scores = torch.zeros(n_docs, device=device)
                    t_scores[0] = 1.0  # positive is first
                if n_docs < max_docs:
                    pad = torch.full((max_docs - n_docs,), float("-inf"), device=device)
                    t_scores = torch.cat([t_scores, pad])
                teacher_scores_list.append(t_scores)

                offset += n_docs

            student_scores = torch.stack(student_scores_list)  # (B, max_docs)
            teacher_scores = torch.stack(teacher_scores_list)  # (B, max_docs)

            # Losses
            loss_kl = kl_distillation_loss(student_scores, teacher_scores, temperature)
            loss_flops_q = flops_loss(q_pooled) * lambda_q
            loss_flops_d = flops_loss(d_pooled) * lambda_d
            loss = loss_kl + loss_flops_q + loss_flops_d

            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps
            pbar.set_postfix(
                loss=f"{loss.item() * grad_accum_steps:.4f}",
                kl=f"{loss_kl.item():.4f}",
                flops_q=f"{loss_flops_q.item():.6f}",
                flops_d=f"{loss_flops_d.item():.6f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if wandb_project:
                import wandb
                wandb.log({
                    "loss": loss.item() * grad_accum_steps,
                    "loss_kl": loss_kl.item(),
                    "loss_flops_q": loss_flops_q.item(),
                    "loss_flops_d": loss_flops_d.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_dir = output_dir / f"checkpoint-{epoch + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter weights
        model.backbone.save_pretrained(ckpt_dir / "lora_adapter")

        # Save config
        import msgspec
        (ckpt_dir / "splare_config.json").write_bytes(
            msgspec.json.encode(vars(config))
        )
        print(f"Checkpoint saved to {ckpt_dir}")

    if wandb_project:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(name="splare-train", help="Train SPLARE model.")


@app.command()
def train_command(
    training_data: Annotated[str, typer.Option(help="Path to training JSONL.")],
    output_dir: Annotated[str, typer.Option("-o", "--output", help="Output directory.")] = "data/splare/checkpoints",
    backbone: Annotated[str, typer.Option(help="Backbone model.")] = "meta-llama/Llama-3.1-8B",
    sae: Annotated[str, typer.Option(help="SAE checkpoint path.")] = "",
    sae_layer: Annotated[int, typer.Option(help="SAE layer.")] = 26,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 3,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 5e-5,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 2,
    grad_accum: Annotated[int, typer.Option(help="Gradient accumulation steps.")] = 64,
    lambda_q: Annotated[float, typer.Option(help="FLOPS query regularization.")] = 1e-4,
    lambda_d: Annotated[float, typer.Option(help="FLOPS document regularization.")] = 1e-4,
    device: Annotated[str, typer.Option(help="Torch device.")] = "cuda",
    wandb_project: Annotated[str | None, typer.Option(help="W&B project name.")] = None,
) -> None:
    """Train a SPLARE model with KL distillation."""
    train_splare(
        Path(training_data),
        Path(output_dir),
        backbone=backbone,
        sae_path=sae,
        sae_layer=sae_layer,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum_steps=grad_accum,
        lambda_q=lambda_q,
        lambda_d=lambda_d,
        device=device,
        wandb_project=wandb_project,
    )
