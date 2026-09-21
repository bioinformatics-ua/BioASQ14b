# BioASQ14b

Codebase for the BIT.UA team's (University of Aveiro) participation in the **14th edition of the BioASQ Task B** challenge on biomedical question answering.

This repository is the official implementation accompanying the paper:

> André Ribeiro, Rúben Garrido, Alexander Christiansen, Richard A. A. Jonker, Sérgio Matos. **"BIT.UA at BioASQ 14B: Modular Retrieval with pg_textsearch and Qdrant, and Agent-Based Answer Generation."** CLEF 2026 Working Notes. [arXiv:2609.04999](https://arxiv.org/abs/2609.04999)

## Overview

The system tackles the two main BioASQ Task B phases:

- **Phase A (document retrieval):** combines BM25 retrieval via PostgreSQL's `pg_textsearch` with dense embedding retrieval indexed in [Qdrant](https://qdrant.tech/), plus a trained reranker and optional HyDE-based query expansion.
- **Phase A+/B (answer & snippet generation):** LLM-based generation with an agent quorum mechanism, where multiple agents debate and converge on a consensus answer, evaluated with an LLM-as-a-judge setup. The pipeline also supports the snippet generation subtask.

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
