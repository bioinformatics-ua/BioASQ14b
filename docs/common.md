# BioASQ Common Module (`src/bioasq/common`)

The `common` module provides shared utilities, type definitions, protocol interfaces, and configuration loaders that are used throughout both Phase A (document retrieval and reranking) and Phase B (answer generation) of the BioASQ pipeline.

## Main Components

### 1. Types (`types.py`)

This file contains the core domain models of the BioASQ pipeline. All domain objects are strictly typed using `msgspec.Struct` to ensure high-performance serialization and clear schemas.

- **`Document`**: Represents a PubMed article with `id` (PMID), `title`, `abstract`, and `text`.
- **`Question`**: Represents a BioASQ question, containing its `id`, `body`, `type` (yesno, factoid, list, summary), `documents`, `snippets`, and answer structures (`ideal_answer`, `exact_answer`).
- **`Snippet`**: Represents an extracted text span from a document.
- **`NegDoc`**: Represents a negative document mined for training a reranker.
- Phase A structural types like `QuestionWithNegatives` and `RerankerPrediction`.
- Phase B generation types like `GeneratedAnswer` and `SynthesisResult`.
- Challenge submission schemas like `BioASQSubmission` and `BioASQSubmissionQuestion`.

**How to use:**
Import the required structure and use it as a standard Python object.

```python
from bioasq.common.types import Document, Question, QuestionType

doc = Document(id="12345", title="Example", abstract="...")
q = Question(id="q1", body="Is X related to Y?", type=QuestionType.YESNO)
```

### 2. Type Aliases (`aliases.py`)

Provides Python 3.12+ `type` aliases for common identifier types and complex data structures used heavily in data loading and evaluation.

- `DocumentId` (str), `QuestionId` (str), `Score` (float)
- Phase A datastructures: `SliceDataset`, `Collection`, `QrelsDict`, `Sample`, `ProcessedSample`.

**How to use:**
Used purely for static type checking across the codebase.

### 3. I/O Utilities (`io.py`)

A centralized, high-performance module for reading and writing JSON/JSONL formats using `msgspec`.

- **`load_json(path, type_)` / `save_json(data, path)`**: Loads JSON, optionally decoding it into a specific type.
- **`load_jsonl(path, type_)` / `save_jsonl(data, path)`**: Loads/saves JSONL data efficiently.
- **`load_collection(path, constraints, id_present)`**: Iterator for streaming massive BioASQ JSONL datastores such as PubMed baselines, applying dynamic constraints on the fly.

**How to use:**

```python
from pathlib import Path
from bioasq.common.io import load_jsonl
from bioasq.common.types import Question

questions = load_jsonl(Path("data.jsonl"), type_=Question)
```

### 4. Protocols (`protocols.py`)

Python `Protocol` definitions for structural subtyping and `ABC` classes for shared implementations in the pipeline.

- **`Scorable`**: Protocol for entities evaluating a `<query, document>` pair.
- **`Loadable`**: Interfaces for lazily loading and unloading resources/models into memory.
- **`Generatable`**: Protocol for Text Generation models (LLMs).
- **`SamplePreprocessor`**: Interface mapping unstructured training samples into tokenized model inputs.
- **`BaseModelBackend`**: Provides a standard ABC with `load()`, `generate()`, `generate_batch()`, and `unload()` for abstraction over local LLMs (vLLM) and external providers (OpenRouter).

### 5. Config (`config.py`)

Configuration parsers to smoothly load parameters into heavily typed structures like HuggingFace `TrainingArguments`, fixing usual YAML quirks natively (e.g. converting string "true" to booleans in heavily nested trees). Also provides utilities for Weights & Biases (W&B) experiment tracking.

- **`_flatten(d)`**: Unnests dictionaries.
- **`setup_wandb(name, project, entity)`**: Deterministically links runs based on ID strings to easily resume Wandb runs.

### 6. Metrics (`metrics.py`)

Pipeline evaluation metrics pooling together standard retrieval evaluations (phase A) and answer generation (phase B).

- **Phase A**: `evaluate_retrieval_run` wraps `ranx` to calculate nDCG, MRR, Recall-at-k, and Map-at-k for predicted search runs against qrels.
- **Phase B**:
  - `accuracy_yesno`: Accuracy and Macra-F1 for yes/no questions.
  - `mrr_factoid`: Exact string matching ranking for short factoid answers.
  - `mean_f1_list`: Sets intersection precision/recall metrics.
  - `rouge2_summary`: ROUGE-2 metric evaluation for summary abstractive generations.

### 7. Decoders (`decoders.py`)

Pre-instantiated global json decoders from `msgspec` for fast parsing of frequent complex objects like `Document`.
