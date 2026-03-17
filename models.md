# BioASQ Phase A Reranker Models

Models trained at `phaseA-reranker/refactored-trainer/`. All paths relative to project root.

## outputs-E5-Pairwise — Base models

Shifter sampler, 5 epochs, pairwise.


| Model                                                         | Characteristics             | Path                                                                                                                    | map-bioasq@10 |
| ------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------- |
| nvidia/llama-nemotron-rerank-1b-v2                            | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/nemotron_fixed/`                                                | 0.9970        |
| BAAI/bge-reranker-v2-m3                                       | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-v2-m3/`                                       | 0.6824        |
| BAAI/bge-reranker-base                                        | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/BAAI_bge-reranker-base/`                                        | 0.6686        |
| nboost/pt-biobert-base-msmarco                                | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/nboost_pt-biobert-base-msmarco/`                                | 0.6608        |
| cross-encoder/ms-marco-MiniLM-L-6-v2                          | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/cross-encoder_ms-marco-MiniLM-L-6-v2/`                          | 0.6373        |
| ncbi/MedCPT-Cross-Encoder                                     | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/ncbi_MedCPT-Cross-Encoder/`                                     | 0.6404        |
| michiyasunaga/BioLinkBERT-base                                | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/michiyasunaga_BioLinkBERT-base/`                                | 0.6403        |
| monologg/biobert_v1.1_pubmed                                  | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/monologg_biobert_v1.1_pubmed/`                                  | 0.6346        |
| microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/` | 0.6291        |
| pritamdeka/S-PubMedBert-MS-MARCO                              | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/pritamdeka_S-PubMedBert-MS-MARCO/`                              | 0.5985        |
| allenai/specter2_base                                         | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/allenai_specter2_base/`                                         | 0.5912        |
| dmis-lab/biobert-base-cased-v1.2                              | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/dmis-lab_biobert-base-cased-v1.2/`                              | 0.5848        |
| cross-encoder/ms-marco-electra-base                           | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/cross-encoder_ms-marco-electra-base/`                           | 0.5654        |
| emilyalsentzer/Bio_ClinicalBERT                               | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/emilyalsentzer_Bio_ClinicalBERT/`                               | 0.4587        |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext                 | shifter, 5 epochs, pairwise | `phaseA-reranker/refactored-trainer/outputs-E5-Pairwise/cambridgeltl_SapBERT-from-PubMedBERT-fulltext/`                 | 0.2594        |


## outputs — Experiments


| Model                                                         | Characteristics                              | Path                                                                                                                                                                                                   | map-bioasq@10 |
| ------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| nvidia/llama-nemotron-rerank-1b-v2                            | E2-S4, multi_neg_pairwise, InfoNCE, FullData | `phaseA-reranker/refactored-trainer/outputs/nvidia_llama-nemotron-rerank-1b-v2_llama/nvidia-llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData/`                                  | 0.9995        |
| nvidia/llama-nemotron-rerank-1b-v2                            | E2, pairwise (13B1+13B2)                     | `phaseA-reranker/refactored-trainer/outputs/nvidia_llama-nemotron-rerank-1b-v2_llama-E2-Pairwise/`                                                                                                     | 0.9970        |
| BAAI/bge-reranker-v2-m3                                       | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/BAAI_bge-reranker-v2-m3/BAAI-bge-reranker-v2-m3-E2-S1-Mpairwise-FullDataTrue/`                                                                             | 0.6705        |
| BAAI/bge-reranker-base                                        | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/BAAI_bge-reranker-base/BAAI-bge-reranker-base-E2-S1-Mpairwise-FullDataTrue/`                                                                               | 0.6489        |
| nboost/pt-biobert-base-msmarco                                | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/nboost_pt-biobert-base-msmarco/nboost-pt-biobert-base-msmarco-E2-S1-Mpairwise-FullDataTrue/`                                                               | 0.6274        |
| ncbi/MedCPT-Cross-Encoder                                     | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/ncbi_MedCPT-Cross-Encoder/ncbi-MedCPT-Cross-Encoder-E2-S1-Mpairwise-FullDataTrue/`                                                                         | 0.6251        |
| michiyasunaga/BioLinkBERT-base                                | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/michiyasunaga_BioLinkBERT-base/michiyasunaga-BioLinkBERT-base-E2-S1-Mpairwise-FullDataTrue/`                                                               | 0.6178        |
| microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-E2-S1-Mpairwise-FullDataTrue/` | 0.6153        |
| cross-encoder/ms-marco-MiniLM-L-6-v2                          | E3-S8, multi_neg_pairwise                    | `phaseA-reranker/refactored-trainer/outputs/cross-encoder_ms-marco-MiniLM-L-6-v2/cross-encoder-ms-marco-MiniLM-L-6-v2-E3-S8-Mmulti_neg_pairwise/`                                                      | 0.6098        |
| monologg/biobert_v1.1_pubmed                                  | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/monologg_biobert_v1.1_pubmed/monologg-biobert_v1.1_pubmed-E2-S1-Mpairwise-FullDataTrue/`                                                                   | 0.6053        |
| cross-encoder/ms-marco-MiniLM-L-6-v2                          | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/cross-encoder_ms-marco-MiniLM-L-6-v2/cross-encoder-ms-marco-MiniLM-L-6-v2-E2-S1-Mpairwise-FullDataTrue/`                                                   | 0.5944        |
| pritamdeka/S-PubMedBert-MS-MARCO                              | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/pritamdeka_S-PubMedBert-MS-MARCO/pritamdeka-S-PubMedBert-MS-MARCO-E2-S1-Mpairwise-FullDataTrue/`                                                           | 0.5839        |
| michiyasunaga/BioLinkBERT-large                               | E2-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/michiyasunaga_BioLinkBERT-large/michiyasunaga-BioLinkBERT-large-E2-S1-Mpairwise-FullDataTrue/`                                                             | 0.5781        |
| ncbi/MedCPT-Cross-Encoder                                     | E3-S1, pairwise, FullData, shifter           | `phaseA-reranker/refactored-trainer/outputs/ncbi_MedCPT-Cross-Encoder/ncbi-MedCPT-Cross-Encoder-E3-S1-Mpairwise-FullDataTrue/`                                                                         | 0.5766        |


