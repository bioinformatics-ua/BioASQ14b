# Trainer Architecture Support

This document maps the architectures in [RERANKER_ARCHITECTURES_REFERENCE.md](RERANKER_ARCHITECTURES_REFERENCE.md) to the refactored-trainer modules.

| Architecture                     | Supported | Modules                                                                 |
| -------------------------------- | --------- | ----------------------------------------------------------------------- |
| **CrossEncoder (BERT/RoBERTa)**  | Yes       | `BasicSamplePreprocessing`, `RankingCollator`, `PairwiseCollator`, `MultiNegativePairwiseCollator` |
| **CrossEncoder (long docs)**     | Yes       | `SentenceCollator`, `RankingSentenceCollator`, `PairwiseSentenceCollator` |
| **Causal LM (Llama, Qwen, etc.)**| Yes       | `RankingCollatorForCasualLM`                                            |
| **T5 / Seq2Seq (MonoT5)**        | Yes       | `RankingCollatorForSeq2Seq`                                            |
| **ColBERT**                      | No        | Requires token-level embeddings and MaxSim; not implemented             |
| **LLM listwise (RankGPT-style)** | No        | Would require prompt-based generation pipeline                          |
| **Multimodal (VL rerankers)**    | No        | Would require vision encoder and image inputs                           |

## Module Locations

- **sample_preprocessing.py:** `BasicSamplePreprocessing` (query+doc concatenation)
- **collator.py:** `RankingCollator`, `PairwiseCollator`, `MultiNegativePairwiseCollator`, `RankingCollatorForCasualLM`, `RankingCollatorForSeq2Seq`, `SentenceCollator`, `RankingSentenceCollator`, `PairwiseSentenceCollator`
- **data.py:** `BioASQPointwiseIterator`, `BioASQPairwiseIterator`, `BioASQMultiNegativePairwiseIterator`, `BioASQDataset`, `BioASQInferenceDataset`
