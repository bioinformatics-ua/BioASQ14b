cd phaseA-reranker
uv run pseudo_relevance_feedback_duck.py ../data/BioASQ-task14bPhaseA-testset1.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-4452 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-4452/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-base-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-2226 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-base-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-2226/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-4452 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-4452/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-base-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue-checkpoint-4452 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-base-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue-checkpoint-4452/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-8904 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-8904/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-17844 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-17844/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-large-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-large-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-large-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-8904 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-large-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-8904/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-large-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-4461 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-large-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-4461/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-6678 \
  --ranx-run refactored-trainer/inference_batch1_2025/michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-6678/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-4452 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-4452/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-2226 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-2226/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-4452 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-4452/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue-checkpoint-4452 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue-checkpoint-4452/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-8904 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-8904/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-17844 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-17844/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-8922/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-8904 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-8904/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-4461 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-4461/predictions.json \
  --model IEETA/BioASQ-13B \
  --revision microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-6678 \
  --ranx-run refactored-trainer/inference_batch1_2025/microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-6678/predictions.json \
  --baseline ../data/baselines/pubmed_baseline_2026.jsonl \
  --lookup ../data/lookup_0.9.json \
  --output ./dprf

# --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/BAAI_bge-reranker-base/BAAI-bge-reranker-base-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/BAAI-bge-reranker-base-E2-S1-Mpairwise-FullDataTrue-checkpoint-4452.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/BAAI_bge-reranker-v2-m3/BAAI-bge-reranker-v2-m3-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/BAAI-bge-reranker-v2-m3-E2-S1-Mpairwise-FullDataTrue-checkpoint-4452.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/cross-encoder_ms-marco-MiniLM-L-6-v2/cross-encoder-ms-marco-MiniLM-L-6-v2-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/cross-encoder-ms-marco-MiniLM-L-6-v2-E2-S1-Mpairwise-FullDataTrue-checkpoint-4452.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/cross-encoder_ms-marco-MiniLM-L-6-v2/cross-encoder-ms-marco-MiniLM-L-6-v2-E3-S8-Mmulti_neg_pairwise/checkpoint-3636 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/cross-encoder-ms-marco-MiniLM-L-6-v2-E3-S8-Mmulti_neg_pairwise-checkpoint-3636.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/michiyasunaga_BioLinkBERT-base/michiyasunaga-BioLinkBERT-base-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/michiyasunaga-BioLinkBERT-base-E2-S1-Mpairwise-FullDataTrue-checkpoint-4452.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/michiyasunaga_BioLinkBERT-large/michiyasunaga-BioLinkBERT-large-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/michiyasunaga-BioLinkBERT-large-E2-S1-Mpairwise-FullDataTrue-checkpoint-4452.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-E2-S1-Mpairwise-FullDataTrue-checkpoint-4452.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/monologg_biobert_v1.1_pubmed/monologg-biobert_v1.1_pubmed-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/monologg-biobert_v1.1_pubmed-E2-S1-Mpairwise-FullDataTrue-checkpoint-4452.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/ncbi_MedCPT-Cross-Encoder/ncbi-MedCPT-Cross-Encoder-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/ncbi-MedCPT-Cross-Encoder-E2-S1-Mpairwise-FullDataTrue-checkpoint-4452.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/ncbi_MedCPT-Cross-Encoder/ncbi-MedCPT-Cross-Encoder-E3-S1-Mpairwise-FullDataTrue/checkpoint-6678 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/ncbi-MedCPT-Cross-Encoder-E3-S1-Mpairwise-FullDataTrue-checkpoint-6678.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/nvidia_llama-nemotron-rerank-1b-v2_llama-E2-Pairwise/checkpoint-5098 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/nvidia_llama-nemotron-rerank-1b-v2_llama-E2-Pairwise-checkpoint-5098.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/nvidia_llama-nemotron-rerank-1b-v2_llama/nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData/checkpoint-600 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData-checkpoint-600.json \
  # --model /home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer/outputs/pritamdeka_S-PubMedBert-MS-MARCO/pritamdeka-S-PubMedBert-MS-MARCO-E2-S1-Mpairwise-FullDataTrue/checkpoint-4452 \
  # --revision main \
  # --ranx-run refactored-trainer/batch01/runs/pritamdeka-S-PubMedBert-MS-MARCO-E2-S1-Mpairwise-FullDataTrue-checkpoint-4452.json \
  