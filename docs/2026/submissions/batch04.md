# Batch 03

## Phase A
- Context1 with all rerankers
- Context1 with best rerankers
- Hybrid rrf w/ HyDE + best rerankers
- Hybrid rrf + best rerankers
- BM25 + best rerankers

## Phase A+
Context1 with best rerankers as base.

- Agents v0
- Agents v1
- Agents v3
- Agents v4

## Phase B
- Agents v0
- Agents v1 
- Agents v2
- Agents v3
- Agents v5

## Agents description

Version 0:
- google/gemma-4-E4B-it
- google/gemma-4-E2B-it
- nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
- Synthesizer model: google/gemma-4-E4B-it

Version 1:
- google/gemma-4-26b-a4b-it
- google/gemma-4-31b-it
- Synthesizer model: google/gemma-4-26b-a4b-it

Version 2:
- nvidia/nemotron-3-super-120b-a12b
- google/gemini-3-flash-preview
- google/gemini-2.5-flash
- openai/gpt-5-mini
- Synthesizer model: google/gemini-2.5-flash

Version 3:
- qwen/qwen3.6-35b-a3b
- qwen/qwen3.6-27b
- qwen/qwen3.6-flash
- Synthesizer model: qwen/qwen3.6-35b-a3b

Version 4:
- mistralai/mistral-medium-3-5
- mistralai/mistral-small-2603
- mistralai/mistral-large-2512
- Synthesizer model: mistralai/mistral-medium-3-5

Version 5:
- mistralai/mistral-medium-3-5
- mistralai/mistral-small-2603
- Synthesizer model: mistralai/mistral-medium-3-5
