# BioASQ 13B — Phase B

Answer generation pipeline for BioASQ Phase B. Given a set of biomedical questions with associated PubMed documents/snippets, generates and synthesizes answers using local (vLLM) or cloud (OpenRouter) LLMs.

## Structure

```
phaseB/
├── loaders/
│   ├── base.py              # BaseModelBackend interface
│   ├── dataloader.py        # BioASQDataLoader — reads BioASQ JSONL
│   ├── local.py             # VLLMBackend — local GPU inference
│   └── cloud.py             # OpenRouterBackend — cloud API inference
├── inference/
│   ├── run.py               # Main inference runner
│   ├── prompts_generic.json # Type-agnostic prompts (ids 1-5), use {d_type}
│   └── prompts_typed.json   # Per question-type prompts (yesno/factoid/list/summary)
├── synthesis/
│   ├── synthesize.py        # Synthesizes answers from multiple run outputs
│   └── prompts.json         # Synthesis prompt variants (ids 1-6)
├── evaluation/
│   ├── evaluate.py          # Evaluation CLI
│   └── metrics.py           # Official BioASQ metrics (ROUGE-2, MRR, F1, maF1)
└── utils/
    ├── lookup_abstracts.py  # Resolves PubMed URLs → full abstract text
    ├── format_converter.py  # Converts run output → BioASQ submission format
    └── convert_for_rerank.py # Prepares run outputs for BM25 reranking
```

## Data format

`BioASQDataLoader` reads JSONL files where each line is a BioASQ question:

```json
{
  "id": "67d6c10618b1e36f2e000027",
  "body": "Is the use of surfactant restricted to neonatology?",
  "type": "yesno",
  "documents": [{"id": "38444695", "text": "...abstract text..."}],
  "snippets": ["...snippet text..."],
  "ideal_answer": ["..."],
  "exact_answer": "no"
}
```

`ideal_answer` and `exact_answer` are present in training data, absent in competition test batches.

If your test batch has documents as URLs only (no text), run `utils/lookup_abstracts.py` first to resolve them against the PubMed baseline.

## Inference

Runs all combinations of `--num-support` × `--prompt-ids` in a single model load, writing one output file per combination.

```bash
python inference/run.py \
    --data-path   data/batch1.jsonl \
    --output-dir  outputs/ \
    --model       /path/to/model \
    --input-type  abstracts \         # or: snippets
    --num-support 3,5,10 \
    --prompt-ids  1,2,3 \
    --backend     local               # or: openrouter
```

Output files: `outputs/{model_name}_{input_type}_{num_support}_{pid}.json`

Output format per file:
```json
{
  "67d6c10618b1e36f2e000027": {"text": "No, surfactant is not...", "valid": true}
}
```

### Input types

| `--input-type` | Source field | Context format |
|---|---|---|
| `abstracts` | `documents[].text` | `abstracts: <text>` |
| `snippets` | `snippets[]` | `snippets: <text>` |

### Prompt files

- `prompts_generic.json` — type-agnostic prompts with `{d_type}` placeholder. Works for all question types. Use these for most runs.
- `prompts_typed.json` — per-type prompts (separate templates for yesno/factoid/list/summary). Use these when question-type-specific formatting matters (e.g. exact answer format for factoid/list).

### Cloud backend

Set `OPENROUTER_API_KEY` in `.env` or the environment, then:

```bash
python inference/run.py \
    --backend    openrouter \
    --model      google/gemini-2.5-flash \
    --data-path  data/batch1.jsonl \
    --output-dir outputs/
```

## Synthesis

Takes multiple `run.py` output files and synthesizes a final answer via LLM. Also runs as a grid over prompt IDs in one model load.

```bash
python synthesis/synthesize.py \
    outputs/run_A.json outputs/run_B.json outputs/run_C.json \
    --data-path  data/batch1.jsonl \
    --output-dir outputs/synthesis/ \
    --model      /path/to/model \
    --prompt-ids 1,2,3 \
    --out-id     exp1
```

Output files: `outputs/synthesis/{out_id}_{model_name}_{num_runs}_{pid}.json`

## Evaluation

Requires run output files that contain `ideal_answer` and `exact_answer` fields (i.e. from synthesis, or from runs using `prompts_typed.json`).

```bash
python evaluation/evaluate.py \
    --predictions outputs/run_A.json \
    --gold        data/training.jsonl
```

Metrics reported:
- **All types**: ROUGE-2 F1 on ideal answers
- **yesno**: macro F1
- **factoid**: MRR, strict accuracy, lenient accuracy (top-5)
- **list**: mean F1

## Utilities

### lookup_abstracts.py

Resolves document URLs in a BioASQ test batch to full abstract text from the PubMed baseline. Run this before inference when your test set has documents as URLs only.

```bash
python utils/lookup_abstracts.py test_batch.json enriched_batch.json
```

Reads from `../../../data/pubmed_baseline_2025.jsonl` by default.

### format_converter.py

Converts a run output file to the official BioASQ submission format.

```bash
python utils/format_converter.py test_set.json run_output.json submission.json baseline.json
```

### convert_for_rerank.py

Merges run outputs and prepares them as JSONL for BM25 reranking (≤200 word answers only).

```bash
python utils/convert_for_rerank.py \
    outputs/run_A.json outputs/run_B.json \
    --out rerank_input.jsonl \
    --testset data/batch1.json
```
