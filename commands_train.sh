cd data
# rsync --progress -azvh -e "ssh -i $HOME/.ssh/id_ed25519_ruben" deucalion:/projects/F202500004HPCVLABUAVEIRO/bioasq/similarity_results/lookup.json ./lookup.json

uv run expand_training_data.py quality/training14b_inflated_clean_wContents.jsonl -l lookup.json -c baselines/pubmed_baseline_2026.jsonl -i ids_per_baseline.json -o quality/training14b_expanded.jsonl
cd ..

cd phaseA-reranker/refactored-trainer
uv run run_experiments.py 2>&1 > run_experiments.log &
uv run run_llama_experiments.py | tee run_llama_experiments.log
cd ..

uv run pseudo_relevance_feedback.py --testset data/BioASQ-task14bPhaseA-testset1.json \
  --ranx-runs refactored-trainer/outputs/nvidia_llama-nemotron-rerank-1b-v2_llama/nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData/predictions/predictions.json,refactored-trainer/outputs/BAAI_bge-reranker-v2-m3/BAAI-bge-reranker-v2-m3-E2-S1-Mpairwise-FullDataTrue/predictions/predictions.json,refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-base/predictions/predictions.json,refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-v2-m3/predictions/predictions.json,refactored-trainer/outputs-E5-Pairwise/michiyasunaga_BioLinkBERT-base/predictions/predictions.json,refactored-trainer/outputs/nboost_pt-biobert-base-msmarco/nboost-pt-biobert-base-msmarco-E2-S1-Mpairwise-FullDataTrue/predictions/predictions.json \
  --models refactored-trainer/outputs/nvidia_llama-nemotron-rerank-1b-v2_llama/nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData/checkpoint-600,refactored-trainer/outputs/BAAI_bge-reranker-v2-m3/BAAI-bge-reranker-v2-m3-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452,refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-base/checkpoint-6375,refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-v2-m3/checkpoint-6375,refactored-trainer/outputs-E5-Pairwise/michiyasunaga_BioLinkBERT-base/checkpoint-6375,refactored-trainer/outputs/nboost_pt-biobert-base-msmarco/nboost-pt-biobert-base-msmarco-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  --baseline ../data/baselines/pubmed_baseline_2026.jsonl \
  --lookup ../data/lookup.json \
  --output ./dprf
