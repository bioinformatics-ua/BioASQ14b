

./scripts/run_snippets.sh extract 2>&1 | tee lora.log

./scripts/run_snippets.sh evaluate 2>&1 | tee evaluate.log


docker compose --profile tei up -d

cd ~/vllm-shit
uv run vllm serve chromadb/context-1 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.65 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss 2>&1 > vllm.log &


docker run -it --name gemma3-e4b --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri -v ~/.cache/huggingface:/root/.cache/huggingface -p 8002:8000 --ipc=host --entrypoint /bin/bash vllm/vllm-openai-rocm:latest

uv run vllm serve google/gemma-4-E4B-it \
  --port 8000 \
  --gpu-memory-utilization 0.28 \
  --trust-remote-code

docker run -it --name gemma3-e2b --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri -v ~/.cache/huggingface:/root/.cache/huggingface -p 8003:8000 --ipc=host --entrypoint /bin/bash vllm/vllm-openai-rocm:latest

uv run vllm serve google/gemma-4-E2B-it \
  --port 8000 \
  --gpu-memory-utilization 0.18 \
  --trust-remote-code


cd ~/BioASQ14B

sleep 1800

uv run -m bioasq.phase_a.context1.cli negatives \
  data/quality/training14b_inflated_clean_wContents.jsonl \
  --reranker-model /home/ucloud/lixo-rollback/phaseA-reranker/outputs_old/BAAI_bge-reranker-base/BAAI-bge-reranker-base-E1-S1-Mpairwise-FullDataTrue--BasicV2/checkpoint-1113 \
  --reranker-model /home/ucloud/lixo-rollback/phaseA-reranker/outputs_old/BAAI_bge-reranker-v2-m3/BAAI-bge-reranker-v2-m3-E1-S1-Mpairwise-FullDataTrue--BasicV2/checkpoint-1113 \
  --reranker-model /home/ucloud/lixo-rollback/phaseA-reranker/outputs_old/michiyasunaga_BioLinkBERT-large/michiyasunaga-BioLinkBERT-large-E1-S1-Mpairwise-FullDataTrue--BasicV2/checkpoint-1113 \
  --reranker-model /home/ucloud/lixo-rollback/phaseA-reranker/outputs_old/michiyasunaga_BioLinkBERT-base/michiyasunaga-BioLinkBERT-base-E1-S1-Mpairwise-FullDataTrue--BasicV2/checkpoint-1113 \
  --tei-embed-url http://localhost:8080/embed
CUDA_VISIBLE_DEVICES=0,1
uv run -m bioasq.phase_a.context1.cli retrieve \
  data/batch03/phasea/BioASQ-task14bPhaseA-testset3.json \
  -o data/batch03/phasea/retrieval/context1_new_rerankers.json \
  --base-url http://localhost:8000 \
  --reranker-model /home/ucloud/BioASQ13B/src/bioasq/phase_a/reranker/outputs/nvidia_llama-nemotron-rerank-1b-v2/nvidia-llama-nemotron-rerank-1b-v2-E1-S1-Mpairwise-FullDataTrue--Shifter/checkpoint-500 \
  --reranker-model /home/ucloud/lixo-rollback/phaseA-reranker/outputs_old/google_medgemma-4b-pt/google-medgemma-4b-pt-E1-S1-Mpairwise-FullDataTrue--BasicV2/checkpoint-500 \
  --reranker-model /home/ucloud/BioASQ13B/src/bioasq/phase_a/reranker/outputs/ncbi_MedCPT-Cross-Encoder/ncbi-MedCPT-Cross-Encoder-E1-S1-Mpairwise-FullDataTrue--Shifter/checkpoint-1178 \
  --reranker-model /home/ucloud/BioASQ13B/src/bioasq/phase_a/reranker/outputs/nboost_pt-biobert-base-msmarco/nboost-pt-biobert-base-msmarco-E1-S1-Mpairwise-FullDataTrue--Shifter/checkpoint-1178 \
  --reranker-model /home/ucloud/BioASQ13B/src/bioasq/phase_a/reranker/outputs/michiyasunaga_BioLinkBERT-large/michiyasunaga-BioLinkBERT-large-E1-S1-Mpairwise-FullDataTrue--Shifter/checkpoint-1178 \
  --reranker-model /home/ucloud/BioASQ13B/src/bioasq/phase_a/reranker/outputs/BAAI_bge-reranker-v2-m3/BAAI-bge-reranker-v2-m3-E1-S1-Mpairwise-FullDataTrue--Shifter/checkpoint-1178 \
  --tei-embed-url http://localhost:8080/embed


uv run python -m src.bioasq.phase_a.retrieval.cli retrieve \
  data/batch03/phasea/BioASQ-task14bPhaseA-testset3.json \
  -o data/batch03/phasea/retrieval/testset3_retrieval.json \
  --hyde-model "openrouter|nvidia/nemotron-3-super-120b-a12b"

CUDA_VISIBLE_DEVICES=1 uv run vllm serve Qwen/Qwen3.5-9B \
  --port 8004 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code

CUDA_VISIBLE_DEVICES=0 uv run vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --port 8005 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code



./scripts/run_snippets.sh extract data/batch03/phasea/retrieval/testset3_retrieval.bm25.jsonl data/batch03/phasea/retrieval/testset3_retrieval.bm25_snippets.jsonl
./scripts/run_snippets.sh extract data/batch03/phasea/retrieval/testset3_retrieval.rrf.jsonl data/batch03/phasea/retrieval/testset3_retrieval.rrf_snippets.jsonl
./scripts/run_snippets.sh extract data/batch03/phasea/retrieval/testset3_retrieval.wsum.jsonl data/batch03/phasea/retrieval/testset3_retrieval.wsum_snippets.jsonl
./scripts/run_snippets.sh extract data/batch03/phasea/retrieval/context1_new_rerankers.jsonl data/batch03/phasea/retrieval/context1_new_rerankers_snippets.jsonl