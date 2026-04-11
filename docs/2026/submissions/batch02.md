# Batch 02

## Retrieval
- Hybrid rrf + best rerankers
- Hybrid rrf + all rerankers
- Hybrid wsum + best rerankers
- BM25 + best rerankers
- BM25 + all rerankers

## Phase A+
- Agents v1
- Agents v2
- Agents v3
- LLM-as-a-judge + top k1 (answers)
- LLM-as-a-judge + top k3 (answers)

## Generation (all use snippets)
- Agents v1
- Agents v2 
- Agents v4 (local)
- LLM-as-a-judge + ensemble summarization with mixed models
- Ensemble (gemini + nemotron + pixa) for exact + Sonnet 4.6 for ideal

## Agents

Version 1:
- Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
- nvidia/nemotron-3-super-120b-a12b
- google/medgemma-27b-text-it
- Synthesizer model: nvidia/nemotron-3-super-120b-a12b

Version 2:
- nvidia/nemotron-3-super-120b-a12b
- x-ai/grok-4.1-fast
- google/gemini-3-flash-preview
- google/gemini-2.5-flash
- openai/gpt-5-mini
- Synthesizer model: anthropic/claude-sonnet-4.6

Version 3:
- Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
- xiaomi/mimo-v2-flash
- qwen/qwen3.5-35b-a3b
- nvidia/nemotron-3-super-120b-a12b
- Synthesizer model: openrouter|openai/gpt-5.4-nano

Version 4:
- google/gemma-4-31b-it
- google/gemma-4-26B-A4B-it
- google/medgemma-27b-text-it
- Synthesizer model: google/gemma-4-31b-it
