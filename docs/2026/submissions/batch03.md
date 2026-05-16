# Batch 03

## Phase A
- Context1 with new rerankers
- Context1 with old rerankers
- Hybrid rrf w/ HyDE + best rerankers
- Hybrid wsum w/ HyDE + best rerankers
- BM25 + best rerankers

## Phase A+
Context1 with new rerankers as base.

- Agents v0
- Agents v1
- Agents v2
- Agents v3
- Agents ensemble

## Phase B
- Agents v0
- Agents v1 
- Agents v2
- Agents v3
- Agents v4
- LLM-as-a-judge + ensemble summarization with mixed models

## Agents description

Version 0 (small models):
- local|google/gemma-4-E4B-it
- local|google/gemma-4-E2B-it
- local|nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
- Synthesizer model: local|google/gemma-4-E4B-it

Version 1 (medium models):
- openrouter|google/gemma-4-26b-a4b-it
- openrouter|google/gemma-4-31b-it
- openrouter|nvidia/nemotron-3-nano-30b-a3b
- openrouter|qwen/qwen3.5-35b-a3b
- Synthesizer model: openrouter|google/gemma-4-26b-a4b-it

Version 2:
- openrouter|nvidia/nemotron-3-super-120b-a12b
- openrouter|x-ai/grok-4.1-fast
- openrouter|google/gemini-3-flash-preview
- openrouter|google/gemini-2.5-flash
- openrouter|openai/gpt-5-mini
- Synthesizer model: openrouter|google/gemini-2.5-flash

Version 3:
- openrouter|google/gemma-4-31b-it
- openrouter|qwen/qwen3.5-35b-a3b
- openrouter|mistralai/mistral-small-2603
- openrouter|mistralai/mistral-large-2512
- Synthesizer model: openrouter|mistralai/mistral-small-2603


## Models to run

AAlborg:
- Qwen/Qwen3.5-9B (8004)
- nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (8005)

AMD:
- Gemma LoRA (snippets) (80GB) (8001)
- google/gemma-4-E4B-it (8002)
- google/gemma-4-E2B-it (8003)

```bash
vllm serve /app/gguf --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.50 --trust-remote-code --tokenizer unsloth/gemma-4-31B --chat-template chat-template.jinja
```
