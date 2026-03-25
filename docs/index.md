# BioASQ Pipeline Documentation

Welcome to the internal documentation for the BioASQ pipeline. This project is structured into modular Python packages to strictly isolate domain concepts, baseline retrieval workflows, and deep-learning rankers.

## Modules Overview

### [Common Module (`src/bioasq/common`)](./common.md)

The foundational utilities that ensure type-safety, fast input/output, and shared mathematical metrics between all phases of the pipeline. Read this to understand the core `Document` and `Question` objects, protocol interfaces (`protocols.py`), or configuration loading.

### [Data Module (`src/bioasq/data`)](./data.md)

Contains robust strategies for acquiring, generating, and traversing datasets. It's built primarily to interact with the raw PubMed XML baseline drops (fetching and parsing to `JSONL`), evaluating and caching fast semantic arrays mapping textual distances recursively spanning whole gigabyte corpus data, and loading conventional challenges structures into iterables.

### [Phase A Module (`src/bioasq/phase_a`)](./phase_a.md)

The pipeline's Information Retrieval (IR) layer. Implements highly scalable indexing and abstractive negative augmentation using **BM25**, followed by the deep neural ranking architectures encompassing contrastive infoNCE and pointwise objective evaluations using cross-encoders iteratively scaled over heavily parallel iterations on DDP environments.

---

> 💡 **Tip:** Each module's documentation dives deeper into the specific CLI tools and module hierarchies. Ensure you have installed the correct dependencies (e.g. `pyterrier_pisa`, `msgspec`) before executing complex commands or imports.
