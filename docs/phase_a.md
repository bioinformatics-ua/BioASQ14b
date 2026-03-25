# BioASQ Phase A Module (`src/bioasq/phase_a`)

Phase A is strictly dedicated to Information Retrieval (IR). It has two main sub-systems: initial statistical retrieval via **BM25**, and high-precision deep learning reranking models in the **Reranker** module.

## 1. BM25 Submodule (`src/bioasq/phase_a/bm25`)

Used fundamentally both for yielding initial baseline retrieval metrics and for generating hard negative candidates to teach the semantic transformer rerankers effectively.

### Indexing (`index.py`)

Provides functions integrating natively with `pyterrier_pisa`, leveraging highly optimized PISA indexing for generating extremely fast search trees.

- `create_index(baseline_path, output_dir)`: Consumes the raw text JSONL stream into PISA artifacts natively.
- `load_index(index_dir)`: Instantiates a PISA index usable instantly for vector queries.

### Negative Mining (`negatives.py`)

Generates pairwise training datasets containing hard negatives.

- **`mine_negatives(...)`**: Retrieves the `num_results` topmost similar documents to a question abstract using BM25, aggressively subtracting out the known correct "positives" labels. The output is a highly competitive, robust training `JSONL` dataset containing `pos_docs` alongside difficult but incorrect `neg_docs` candidates.

## 2. Neural Reranker Submodule (`src/bioasq/phase_a/reranker`)

The deep-learning core architecture of Phase A. An interconnected ecosystem of Dataset abstractions, Preprocessors, dynamic Samplers, multi-paradigm Losses, and overridden HuggingFace Trainers.

### Preprocessing (`preprocessing.py`)

Provides instances conforming to the `SamplePreprocessor` protocol.

- **`BasicSamplePreprocessing`**: The conventional CrossEncoder protocol joining queries and documents (e.g. BERT with `[SEP]`).
- **`NemotronSamplePreprocessing`**: Specific template alignments conforming to LlaMA Nemotron specifications natively using direct prompts (`question:{query} \n \n passage:{passage}`).

### Collators (`collator.py`)

Extends `transformers.DataCollator` concepts handling 2D and 3D batch stacking logically for the models inputs.

- `RankingCollator`, `PairwiseCollator`, `MultiNegativePairwiseCollator`: Handles `pos_inputs` against multiple stacked lists of `neg_inputs` efficiently.
- Supports variations like `SentenceCollator` extending hierarchical chunk aggregation dynamically.

### Loss Functions (`losses.py`)

Mathematical objective evaluations natively optimized into PyTorch operations.

- **`bce_loss`**: Standard point-wise binary classifier metrics.
- **`margin_ranking_loss`**: Classical pairwise hinge margins (`pos_score > neg_score + margin`).
- **`multi_negative_margin_loss`**: Reduced margins across N-candidates aggregations natively.
- **`multi_negative_infonce_loss`**: Employs scalable InfoNCE contrastive temperatures to handle 1 Pos + N NeG combinations effectively with sharp gradients.

### Samplers (`sampler.py`)

Decides iteratively how the datasets represent data variations continuously to models across epochs over complex SliceDatasets natively.

- **`BasicSampler`**: Selects items randomly uniformly.
- **`ExponentialWeightSampler`**: Heavily prioritizes items with higher BioASQ relevance ranking classes.
- **`HigherConfidenceNegativesSampler`**: Truncates the topmost 10 entries of BM25 index hits explicitly allowing only harder negatives further down the distribution.
- **`ShifterSampler`**: Actively enforces complex Curriculum Learning protocols shrinking candidate pools continuously down over defined `epoch` limits forcing the model to distinguish more and more difficult pairs natively as epochs pass.

### Datasets (`data.py`)

Encapsulates heavily optimized multi-GPU dataset streams integrating PyTorch `IterableDataset` objects internally. Generates structured `BioASQPointwiseIterator`, `BioASQPairwiseIterator`, and `BioASQMultiNegativePairwiseIterator`.

### Trainers (`trainer.py`)

Provides overridden standard HuggingFace `Trainer` instances executing custom batch prediction steps inherently. Supports seamlessly pointwise `PointwiseRerankerTrainer`, basic pairwise structures `PairwiseRerankerTrainer`, and highly advanced variants such as `MultiNegativePairwiseRerankerTrainer` handling InfoNCE loss gradients inside conventional HF training pipelines.

### Evaluation (`evaluate.py`)

Conducts model inferences strictly over mapped testing datasets directly on tensors leveraging hardware `autocast` protocols seamlessly outputting generated validation metric results mapped structurally against canonical Qrels mapping sets with **Ranx**. Contains the actual inference engine mapping function architectures to produce raw PhaseA metrics (`nDCG`, `MAP`, `Recall`).
