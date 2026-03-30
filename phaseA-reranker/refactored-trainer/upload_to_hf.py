"""
Upload fine-tuned reranker models to IEETA/BioASQ-14B on Hugging Face Hub.

For each model variant:
  - Picks the last checkpoint (highest step number)
  - Uploads only inference-relevant files (strips optimizer/scheduler/rng states)
  - Uses the variant directory name as the HF subfolder

Usage:
    uv run python upload_to_hf.py [--dry-run] [--filter SUBSTRING]

Requirements:
    HF_TOKEN env var set, or `huggingface-cli login` done.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
HF_REPO_ID = "IEETA/BioASQ-14B"
COMMIT_MESSAGE = (
    "Upload fine-tuned rerankers for BioASQ 14B\n\n"
    "Co-authored-by: André Ribeiro <andrepedro2004@hotmail.com>\n"
    "Co-authored-by: Rúben Garrido <rubengarrido@ua.pt>\n"
)

# Each entry: (outputs_root, subfolder_suffix)
# subfolder_suffix is appended to the model dir name for E5-Pairwise to disambiguate
OUTPUT_ROOTS: list[tuple[Path, str]] = [
    (SCRIPT_DIR / "outputs", ""),
    (SCRIPT_DIR / "outputs-E5-Pairwise", "-E5-Pairwise"),
]

# Files to keep from each checkpoint (everything else is dropped)
KEEP_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
    # custom model code (nvidia llama variants)
    "llama_bidirectional_model.py",
}

# Extra files at the variant level (not inside a checkpoint dir)
KEEP_VARIANT_LEVEL = {
    "ranx_results.json",
}


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

README = """\
---
language: en
license: apache-2.0
tags:
  - bioasq
  - biomedical
  - reranking
  - information-retrieval
---

# BioASQ Phase A Reranker Models

Fine-tuned rerankers for biomedical document retrieval, trained on BioASQ data.
All models are cross-encoders fine-tuned from publicly available base models.

## Loading a model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

subfolder = "nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData"
tokenizer = AutoTokenizer.from_pretrained("IEETA/BioASQ-14B", subfolder=subfolder)
model = AutoModelForSequenceClassification.from_pretrained("IEETA/BioASQ-14B", subfolder=subfolder)
```

For nvidia/llama-nemotron variants, also copy `llama_bidirectional_model.py` from the subfolder
and pass `trust_remote_code=True`.

---

## outputs-E5-Pairwise — Shifter sampler, 5 epochs, pairwise

| Model | Path | map-bioasq@10 |
|---|---|---|
| nvidia/llama-nemotron-rerank-1b-v2 | `nvidia-llama-nemotron-rerank-1b-v2-E5-Pairwise` | 0.9970 |
| BAAI/bge-reranker-v2-m3 | `BAAI-bge-reranker-v2-m3-E5-Pairwise` | 0.6824 |
| BAAI/bge-reranker-base | `BAAI-bge-reranker-base-E5-Pairwise` | 0.6686 |
| nboost/pt-biobert-base-msmarco | `nboost-pt-biobert-base-msmarco-E5-Pairwise` | 0.6608 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | `cross-encoder-ms-marco-MiniLM-L-6-v2-E5-Pairwise` | 0.6373 |
| ncbi/MedCPT-Cross-Encoder | `ncbi-MedCPT-Cross-Encoder-E5-Pairwise` | 0.6404 |
| michiyasunaga/BioLinkBERT-base | `michiyasunaga-BioLinkBERT-base-E5-Pairwise` | 0.6403 |
| monologg/biobert_v1.1_pubmed | `monologg-biobert_v1.1_pubmed-E5-Pairwise` | 0.6346 |
| microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext | `microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-E5-Pairwise` | 0.6291 |
| pritamdeka/S-PubMedBert-MS-MARCO | `pritamdeka-S-PubMedBert-MS-MARCO-E5-Pairwise` | 0.5985 |
| allenai/specter2_base | `allenai-specter2_base-E5-Pairwise` | 0.5912 |
| dmis-lab/biobert-base-cased-v1.2 | `dmis-lab-biobert-base-cased-v1.2-E5-Pairwise` | 0.5848 |
| cross-encoder/ms-marco-electra-base | `cross-encoder-ms-marco-electra-base-E5-Pairwise` | 0.5654 |
| emilyalsentzer/Bio_ClinicalBERT | `emilyalsentzer-Bio_ClinicalBERT-E5-Pairwise` | 0.4587 |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | `cambridgeltl-SapBERT-from-PubMedBERT-fulltext-E5-Pairwise` | 0.2594 |

---

## outputs — Experiments

| Model | Characteristics | Path | map-bioasq@10 |
|---|---|---|---|
| nvidia/llama-nemotron-rerank-1b-v2 | E2-S4, multi_neg_pairwise, InfoNCE, FullData | `nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData` | 0.9995 |
| nvidia/llama-nemotron-rerank-1b-v2 | E2, pairwise (13B1+13B2) | `nvidia-llama-nemotron-rerank-1b-v2_llama-E2-Pairwise` | 0.9970 |
| BAAI/bge-reranker-v2-m3 | E2-S1, pairwise, FullData, shifter | `BAAI-bge-reranker-v2-m3-E2-S1-Mpairwise-FullDataTrue` | 0.6705 |
| BAAI/bge-reranker-base | E2-S1, pairwise, FullData, shifter | `BAAI-bge-reranker-base-E2-S1-Mpairwise-FullDataTrue` | 0.6489 |
| nboost/pt-biobert-base-msmarco | E2-S1, pairwise, FullData, shifter | `nboost-pt-biobert-base-msmarco-E2-S1-Mpairwise-FullDataTrue` | 0.6274 |
| ncbi/MedCPT-Cross-Encoder | E2-S1, pairwise, FullData, shifter | `ncbi-MedCPT-Cross-Encoder-E2-S1-Mpairwise-FullDataTrue` | 0.6251 |
| michiyasunaga/BioLinkBERT-base | E2-S1, pairwise, FullData, shifter | `michiyasunaga-BioLinkBERT-base-E2-S1-Mpairwise-FullDataTrue` | 0.6178 |
| microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext | E2-S1, pairwise, FullData, shifter | `microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-E2-S1-Mpairwise-FullDataTrue` | 0.6153 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | E3-S8, multi_neg_pairwise | `cross-encoder-ms-marco-MiniLM-L-6-v2-E3-S8-Mmulti_neg_pairwise` | 0.6098 |
| monologg/biobert_v1.1_pubmed | E2-S1, pairwise, FullData, shifter | `monologg-biobert_v1.1_pubmed-E2-S1-Mpairwise-FullDataTrue` | 0.6053 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | E2-S1, pairwise, FullData, shifter | `cross-encoder-ms-marco-MiniLM-L-6-v2-E2-S1-Mpairwise-FullDataTrue` | 0.5944 |
| pritamdeka/S-PubMedBert-MS-MARCO | E2-S1, pairwise, FullData, shifter | `pritamdeka-S-PubMedBert-MS-MARCO-E2-S1-Mpairwise-FullDataTrue` | 0.5839 |
| michiyasunaga/BioLinkBERT-large | E2-S1, pairwise, FullData, shifter | `michiyasunaga-BioLinkBERT-large-E2-S1-Mpairwise-FullDataTrue` | 0.5781 |
| ncbi/MedCPT-Cross-Encoder | E3-S1, pairwise, FullData, shifter | `ncbi-MedCPT-Cross-Encoder-E3-S1-Mpairwise-FullDataTrue` | 0.5766 |
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def last_checkpoint(d: Path) -> Path | None:
    """Return the checkpoint subdir with the highest step number."""
    ckpts = sorted(
        [c for c in d.iterdir() if c.is_dir() and c.name.startswith("checkpoint-")],
        key=lambda c: int(c.name.split("-")[1]),
    )
    return ckpts[-1] if ckpts else None


def sanitize(name: str) -> str:
    """Convert dir-style underscores to hyphens for a clean HF subfolder name.

    e.g. 'BAAI_bge-reranker-base' → 'BAAI-bge-reranker-base'
    Only replaces the first underscore (the org/model separator).
    """
    return name.replace("_", "-", 1)


def collect_variants() -> list[tuple[str, Path, Path | None]]:
    """
    Return (subfolder_name, variant_dir, best_checkpoint_dir) for every variant.

    Three layout types exist across the two output roots:

      A) outputs/<base_model>/<variant_name>/checkpoint-XXXX   (most models in outputs/)
      B) outputs/<base_model_with_variant>/checkpoint-XXXX     (nvidia E2-Pairwise in outputs/)
      C) outputs-E5-Pairwise/<base_model>/checkpoint-XXXX      (all E5-Pairwise models)
    """
    results = []

    for root, suffix in OUTPUT_ROOTS:
        if not root.exists():
            print(f"  [warn] {root} does not exist, skipping.")
            continue

        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue

            # Check if this dir directly contains checkpoints (layouts B and C)
            direct_ckpts = [
                d for d in entry.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ]
            if direct_ckpts:
                ckpt = sorted(direct_ckpts, key=lambda d: int(d.name.split("-")[1]))[-1]
                subfolder = sanitize(entry.name) + suffix
                results.append((subfolder, entry, ckpt))
                continue

            # Layout A: subdirectories are variants
            for variant_dir in sorted(entry.iterdir()):
                if not variant_dir.is_dir():
                    continue
                ckpt = last_checkpoint(variant_dir)
                results.append((variant_dir.name, variant_dir, ckpt))

    return results


def stage_variant(variant_dir: Path, ckpt_dir: Path | None, staging: Path) -> None:
    """Copy inference-relevant files into staging/."""
    staging.mkdir(parents=True, exist_ok=True)

    if ckpt_dir and ckpt_dir.exists():
        for f in ckpt_dir.iterdir():
            if f.name in KEEP_FILES:
                shutil.copy2(f, staging / f.name)

    for fname in KEEP_VARIANT_LEVEL:
        src = variant_dir / fname
        if src.exists():
            shutil.copy2(src, staging / fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print plan without uploading")
    parser.add_argument("--filter", default="", help="Only process variants containing this substring")
    args = parser.parse_args()

    api = HfApi()
    token = os.environ.get("HF_TOKEN")

    if not args.dry_run:
        create_repo(
            repo_id=HF_REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False,
            token=token,
        )
        print(f"Repo {HF_REPO_ID} ready.")
        print("  Uploading README.md ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=README.encode(),
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"Add model card\n\n{COMMIT_MESSAGE}",
            token=token,
        )
        print("done")

    variants = collect_variants()
    if args.filter:
        variants = [(n, v, c) for n, v, c in variants if args.filter in n]

    print(f"\nFound {len(variants)} variants to upload:\n")
    for subfolder, variant_dir, ckpt in variants:
        ckpt_name = ckpt.name if ckpt else "NO CHECKPOINT FOUND"
        print(f"  {subfolder}/  <-  {ckpt_name}")

    if args.dry_run:
        print("\n[dry-run] No files uploaded.")
        return

    print()
    with tempfile.TemporaryDirectory() as tmpdir:
        for subfolder, variant_dir, ckpt in variants:
            if ckpt is None:
                print(f"  SKIP {subfolder} — no checkpoint found")
                continue

            staging = Path(tmpdir) / subfolder
            stage_variant(variant_dir, ckpt, staging)

            staged_files = list(staging.iterdir())
            if not staged_files:
                print(f"  SKIP {subfolder} — no files to upload")
                continue

            print(f"  Uploading {subfolder}/ ({len(staged_files)} files) ...", end=" ", flush=True)
            api.upload_folder(
                folder_path=str(staging),
                repo_id=HF_REPO_ID,
                repo_type="model",
                path_in_repo=subfolder,
                commit_message=f"Add {subfolder}\n\n{COMMIT_MESSAGE}",
                token=token,
            )
            print("done")

            # Clean up staging to avoid tmp disk pressure (models can be large)
            shutil.rmtree(staging)

    print(f"\nAll done. View at: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
