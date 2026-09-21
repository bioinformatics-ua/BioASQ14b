# BioASQ14b

Codebase for the BIT.UA team's (University of Aveiro) participation in the **14th edition of the BioASQ Task B** challenge on biomedical question answering.

This repository is the official implementation accompanying the paper:

> André Ribeiro, Rúben Garrido, Alexander Christiansen, Richard A. A. Jonker, Sérgio Matos. **"BIT.UA at BioASQ 14B: Modular Retrieval with pg_textsearch and Qdrant, and Agent-Based Answer Generation."** CLEF 2026 Working Notes. [arXiv:2609.04999](https://arxiv.org/abs/2609.04999)

## System Overview & Approaches

The pipeline tackles the primary BioASQ Task B phases through a modular architecture:

### 1. Phase A — Document Retrieval
A hybrid multi-stage retrieval pipeline designed for scalable search and precise ranking over biomedical literature (PubMed):
- **Lexical Search (BM25):** Replaces traditional inverted-index tooling with PostgreSQL-based [`pg_textsearch`](https://github.com/timescale/pg_textsearch), enabling fast BM25 scoring and dynamic index updates directly inside the database.
- **Dense Retrieval:** Vector embeddings served via Hugging Face **Text Embeddings Inference (TEI)** and indexed in [Qdrant](https://qdrant.tech/) for GPU-accelerated similarity search.
- **Query Expansion:** Incorporates **Hypothetical Document Embeddings (HyDE)** to bridge vocabulary gaps between colloquial questions and medical literature, alongside an agentic **Context-1** retrieval strategy.
- **Neural Reranking:** Fine-tuned cross-encoder rerankers trained with hard-negative mining sourced directly from the dense retrieval pool to maximize top-$k$ Precision and MAP.

### 2. Snippet Generation
- **Fine-Tuned Gemma4 31b Model:** Uses a Gemma model fine-tuned via LoRA in two stages:
  1. *Biomedical domain adaptation* for domain knowledge injection.
  2. *Task-specific fine-tuning* for passage extraction and retrieval-augmented generation.
- **Span Extraction:** Predicts and extracts relevant text spans from top-ranked documents, mapped back to precise abstract character offsets.
.

### 3. Phase A+ / Phase B — Answer Generation
Addresses exact (Yes/No, Factoid, List) and ideal (Summary) answers using system-generated snippets (Phase A+) or gold-standard annotations (Phase B):
- **Agent Quorum Mechanism:** Deploys multiple LLM agents configured with diverse prompt strategies that independently propose, critique, and iteratively converge on a consensus answer.
- **Adaptive Document Retention:** Iteratively trims redundant or noisy context between debate rounds to keep generation focused on salient facts.
- **LLM-as-a-Judge:** An automated evaluation harness used during development to score candidates on factual consistency, completeness, and alignment with BioASQ task formats.


## Repository Structure

- `src/bioasq/phase_a/` - retrieval pipeline (BM25 + dense retrieval, reranking, query expansion).
- `src/bioasq/phase_b/` - answer generation, including the agent quorum logic.
- `src/bioasq/snippets/` - snippet extraction/generation.
- `src/bioasq/common/` and `src/bioasq/data/` - shared utilities and data handling.
- `src/bioasq/cli.py` - command-line entry point.
- `compose.yaml`, `init.sql`, `qdrant_config.yaml` - Docker Compose setup for the PostgreSQL and Qdrant services used for indexing/search.


## Requirements

The project uses [`uv`](https://github.com/astral-sh/uv) for Python dependency management (see `pyproject.toml` / `uv.lock`) and Docker Compose to run PostgreSQL (with `pg_textsearch`) and Qdrant.

## Getting Started

1. Install dependencies with `uv sync`.
2. Start the supporting services: `docker compose up -d`.
3. Use the `bioasq` cli

## Citation

If you use this code, please cite the associated paper:

```bibtex
@misc{ribeiro2026bituabioasq14bmodular,
      title={BIT.UA at BioASQ 14B: Modular Retrieval with pg_textsearch and Qdrant, and Agent-Based Answer Generation}, 
      author={André Ribeiro and Rúben Garrido and Alexander Christiansen and Richard A. A. Jonker and Sérgio Matos},
      year={2026},
      eprint={2609.04999},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.04999}, 
}
```
