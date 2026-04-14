"""Batch document/query encoding with SPLARE for offline indexing.

Follows the same sharding pattern as
:mod:`bioasq.data.embeddings.generate_embeddings` — processes PubMed
baseline JSONL in chunks and produces sparse vector shards on disk.

Shard format:
  - ``splare_{start}_{end}.indices.json`` — list of ``list[int]``
  - ``splare_{start}_{end}.values.json``  — list of ``list[float]``
  - ``splare_{start}_{end}.ids.json``     — list of PMID strings
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import msgspec
import torch
import typer

from bioasq.common import PROJECT_DATA_DIR
from bioasq.common.io import load_collection

_SPLARE_EXPORT_DIR = PROJECT_DATA_DIR / "splare" / "export"


@dataclass
class SplareShardMeta:
    """Metadata for one shard of SPLARE-encoded documents."""

    ids: list[str]
    indices: list[list[int]]
    values: list[list[float]]


def _save_shard(
    output_dir: Path,
    ids: list[str],
    indices: list[list[int]],
    values: list[list[float]],
    start: int,
    end: int,
) -> None:
    """Persist one shard of sparse vectors to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = msgspec.json.Encoder()

    (output_dir / f"splare_{start}_{end}.ids.json").write_bytes(encoder.encode(ids))
    (output_dir / f"splare_{start}_{end}.indices.json").write_bytes(encoder.encode(indices))
    (output_dir / f"splare_{start}_{end}.values.json").write_bytes(encoder.encode(values))


def _load_shard(
    output_dir: Path, prefix: str
) -> tuple[list[str], list[list[int]], list[list[float]]]:
    """Load one shard from disk."""
    decoder = msgspec.json.Decoder
    ids = decoder(list[str]).decode((output_dir / f"{prefix}.ids.json").read_bytes())
    indices = decoder(list[list[int]]).decode((output_dir / f"{prefix}.indices.json").read_bytes())
    values = decoder(list[list[float]]).decode((output_dir / f"{prefix}.values.json").read_bytes())
    return ids, indices, values


def list_shards(export_dir: Path) -> list[str]:
    """List shard prefixes in the export directory."""
    ids_files = sorted(export_dir.glob("splare_*.ids.json"))
    return [f.stem.rsplit(".", 1)[0] for f in ids_files]


def encode_corpus(
    baseline_path: Path,
    output_dir: Path,
    *,
    model_path: str = "meta-llama/Llama-3.1-8B",
    sae_path: str = "",
    sae_layer: int = 26,
    chunk_size: int = 10_000,
    batch_size: int = 4,
    device: str = "cuda",
    doc_topk: int = 400,
    resume: bool = True,
) -> None:
    """Encode an entire PubMed baseline JSONL into SPLARE sparse vectors.

    Documents are processed in chunks of ``chunk_size``; each chunk is saved
    as a shard to disk so the process can be resumed.
    """
    from bioasq.phase_a.splare.model import SplareConfig, SplareModel

    config = SplareConfig(
        backbone_name_or_path=model_path,
        sae_name_or_path=sae_path,
        sae_layer=sae_layer,
        doc_topk=doc_topk,
    )
    model = SplareModel(config).load(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine already-processed shards for resume
    done_shards: set[str] = set()
    if resume:
        for prefix in list_shards(output_dir):
            done_shards.add(prefix)

    global_start = 0
    for docs in load_collection(baseline_path, chunk_size=chunk_size):
        end = global_start + len(docs)
        shard_prefix = f"splare_{global_start}_{end}"
        if shard_prefix in done_shards:
            print(f"Skipping shard {shard_prefix} (already exists)")
            global_start = end
            continue

        texts = [d.full_text for d in docs]
        ids = [d.pmid for d in docs]

        sparse_vecs = model.encode_documents(texts, batch_size=batch_size, show_progress=True)

        all_indices = [sv[0] for sv in sparse_vecs]
        all_values = [sv[1] for sv in sparse_vecs]

        _save_shard(output_dir, ids, all_indices, all_values, global_start, end)
        print(f"Saved shard {shard_prefix}: {len(ids)} documents")

        global_start = end
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Encoding complete. {global_start} documents processed.")


# ---------------------------------------------------------------------------
# Typer CLI
# ---------------------------------------------------------------------------

app = typer.Typer(name="splare-encode", help="SPLARE corpus encoding.")


@app.command()
def encode_command(
    baseline: Annotated[str, typer.Option(help="Path to PubMed baseline JSONL.")],
    output: Annotated[
        str, typer.Option("-o", "--output", help="Output directory for shards.")
    ] = str(_SPLARE_EXPORT_DIR),
    model: Annotated[
        str, typer.Option(help="Backbone model name or path.")
    ] = "meta-llama/Llama-3.1-8B",
    sae: Annotated[str, typer.Option(help="SAE checkpoint path (local or HF).")] = "",
    sae_layer: Annotated[int, typer.Option(help="Transformer layer for SAE.")] = 26,
    chunk_size: Annotated[int, typer.Option(help="Documents per shard.")] = 10_000,
    batch_size: Annotated[int, typer.Option(help="Encoding batch size.")] = 4,
    device: Annotated[str, typer.Option(help="Torch device.")] = "cuda",
    doc_topk: Annotated[int, typer.Option(help="Top-K pooling for documents.")] = 400,
    no_resume: Annotated[bool, typer.Option(help="Disable resume from existing shards.")] = False,
) -> None:
    """Encode a PubMed baseline into SPLARE sparse vectors."""
    encode_corpus(
        Path(baseline),
        Path(output),
        model_path=model,
        sae_path=sae,
        sae_layer=sae_layer,
        chunk_size=chunk_size,
        batch_size=batch_size,
        device=device,
        doc_topk=doc_topk,
        resume=not no_resume,
    )
