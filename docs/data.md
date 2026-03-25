# BioASQ Data Module (`src/bioasq/data`)

The `data` module encapsulates all logic to download, load, parse, and process BioASQ datasets, including the massive PubMed baselines.

## Main Components

### 1. Baseline Downloader (`baseline_downloader.py`)

A fully-featured multiprocessing Typer CLI app meant to fetch gigabytes of data from the official PubMed FTP / web-archives and subsequently parse it into localized JSONL datastores utilized by the semantic indexers.

It relies dynamically on the `pubmed_parser` library to handle the complex underlying XML structure and metadata nodes.

**Commands:**

- `download`: Fetches the extensive `.xml.gz` baselines concurrently and handles robustly internal connection retry/backoff policies.
- `parse`: Extracts the exact data needed (PMID, title, abstract) into streamlined `JSONL` outputs to be fed later into dense embedding generators or BM25 indexers.

**How to use:**

```bash
# Download 2026 data using 8 workers
python -m bioasq.data.baseline_downloader download 2026 -w 8

# Parse the downloaded files into a JSONL structure
python -m bioasq.data.baseline_downloader parse 2026 -o baseline_2026.jsonl
```

### 2. General Data Loader (`dataloader.py`)

The `BioASQDataLoader` takes the idiosyncratic BioASQ task JSON schema dumps (typically encapsulated under `{"questions": [...]}`) or native `JSONL` and hydrates strongly-typed `Question` objects containing `Snippet` definitions and hierarchical exact/ideal answers structures safely. It provides a standard iterable interface usable widely in Python pipelines.

**How to use:**

```python
from bioasq.data.dataloader import BioASQDataLoader

loader = BioASQDataLoader("test_batch_1.json")
for question in loader:
    print(question.id, question.body)
```

### 3. Embeddings Tools (`embeddings/`)

Under heavy experimental scenarios, standard semantic document retrieval relies on computing Dense Vector Embeddings across massive arrays of data representations natively.

#### `generate_embeddings.py`

Connects asynchronously against an operational local HuggingFace Text Embeddings Inference (TEI) server implementation via fast parallel asynchronous endpoints (`httpx`). It chunks the data and flushes gigabytes of embedding `numpy` representations (`.npy` binary files) chunk by chunk.
**Usage via CLI:**

```bash
python -m bioasq.data.embeddings.generate_embeddings baseline_2026.jsonl -o ../dense_vectors
```

#### `find_similarity.py`

Spawns highly efficient multi-GPU routines scaling semantic cosine distance matrix multiplications (`sim = file0 @ file1.T`) directly on available `CUDA` hardware. Discards values under a parameterized threshold (`T=0.80`) and saves the exact index blocks for similarities across pairwise batches iteratively.

#### `merge_shards.py`

Collects the disconnected binary shards provided by `find_similarity.py` and produces an optimized lookup hash table mapping PMIDs to their nearest relevant neighbors across the whole corpus space mapping, essential for large scale negative augmentations.
