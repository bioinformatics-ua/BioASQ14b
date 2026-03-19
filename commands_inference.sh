cd phaseA-reranker/refactored-trainer

# declare -A MODELS=(
#   ["ncbi_MedCPT-Cross-Encoder"]="outputs/ncbi_MedCPT-Cross-Encoder/ncbi-MedCPT-Cross-Encoder-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452"
#   ["microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"]="outputs/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452"
#   ["michiyasunaga_BioLinkBERT-base"]="outputs/michiyasunaga_BioLinkBERT-base/michiyasunaga-BioLinkBERT-base-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452"
#   ["pritamdeka_S-PubMedBert-MS-MARCO"]="outputs/pritamdeka_S-PubMedBert-MS-MARCO/pritamdeka-S-PubMedBert-MS-MARCO-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452"
#   ["nboost_pt-biobert-base-msmarco"]="outputs/nboost_pt-biobert-base-msmarco/nboost-pt-biobert-base-msmarco-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452"
#   ["BAAI_bge-reranker-base"]="outputs/BAAI_bge-reranker-base/BAAI-bge-reranker-base-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452"
#   ["BAAI_bge-reranker-v2-m3"]="outputs/BAAI_bge-reranker-v2-m3/BAAI-bge-reranker-v2-m3-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452"
#   ["nvidia_llama-nemotron-rerank-1b-v2_llama"]="outputs/nvidia_llama-nemotron-rerank-1b-v2_llama/nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData/checkpoint-600"
# )

declare -a revisions_array=(
"michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-4452"
"michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922"
"michiyasunaga-BioLinkBERT-base-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-2226"
"michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-4452"
"michiyasunaga-BioLinkBERT-base-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue-checkpoint-4452"
"michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922"
"michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-8904"
"michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-17844"
"michiyasunaga-BioLinkBERT-large-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922"
"michiyasunaga-BioLinkBERT-large-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-8904"
"michiyasunaga-BioLinkBERT-large-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-4461"
"michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-6678"
"microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-4452"
"microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922"
"microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-2226"
"microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-4452"
"microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue-checkpoint-4452"
"microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922"
"microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-8904"
"microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-17844"
"microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922"
"microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-8904"
"microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-4461"
"microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-6678"
)


for revision in "${revisions_array[@]}"; do
  echo "\nRunning inference for ${revision}..."
  uv run main.py inference \
    --model-name "IEETA/BioASQ-13B" \
    --revision "${revision}" \
    --questions-path ../../phaseA-BM25/Batch01/b01.jsonl \
    --output-path inference_batch1_2025/${revision}/predictions.json \
    --batch-size 16 \
    --max-length 512 \
    --max-docs 100 \
    --inference-dtype bfloat16 \
    --top-k 10
done

# TODO: Ensemble, etc.

# uv run pseudo_relevance_feedback.py data/BioASQ-task14bPhaseA-testset1.json \
#   --ranx-runs refactored-trainer/outputs/nvidia_llama-nemotron-rerank-1b-v2_llama/nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData/predictions/predictions.json,refactored-trainer/outputs/BAAI_bge-reranker-v2-m3/BAAI-bge-reranker-v2-m3-E2-S1-Mpairwise-FullDataTrue/predictions/predictions.json,refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-base/predictions/predictions.json,refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-v2-m3/predictions/predictions.json,refactored-trainer/outputs-E5-Pairwise/michiyasunaga_BioLinkBERT-base/predictions/predictions.json,refactored-trainer/outputs/nboost_pt-biobert-base-msmarco/nboost-pt-biobert-base-msmarco-E2-S1-Mpairwise-FullDataTrue/predictions/predictions.json \
#   --models refactored-trainer/outputs/nvidia_llama-nemotron-rerank-1b-v2_llama/nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData/checkpoint-600,refactored-trainer/outputs/BAAI_bge-reranker-v2-m3/BAAI-bge-reranker-v2-m3-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452,refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-base/checkpoint-6375,refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-v2-m3/checkpoint-6375,refactored-trainer/outputs-E5-Pairwise/michiyasunaga_BioLinkBERT-base/checkpoint-6375,refactored-trainer/outputs/nboost_pt-biobert-base-msmarco/nboost-pt-biobert-base-msmarco-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
#   --baseline ../data/baselines/pubmed_baseline_2026.jsonl \
#   --lookup ../data/lookup_0.9.json \
#   --output ./dprf
