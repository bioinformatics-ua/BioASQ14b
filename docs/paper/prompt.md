# Prompt

This project is made for the BioASQ 14B challenge. Now, it's time to write the paper. We've been doing submissions for the past years, so each time the paper reflect on the work done, when comparing to the previous year. We also need to explain how it has improved our results.

The 13B paper is available at @paper_13b.pdf for reference on the structure and content.

## What was done this year

General:

- Huge code refactor

Phase A (retrieval):

- New reranker pipeline
- RRF
- HyDE
- Context1
- Snippets
- Include dense retrieval in negative generation
- BM25 using pg_textsearch
- pgvectorscale
- Qdrant
- Future work: SPLARe, ColBERT

Phase A+/B (generation):

- LLM as a judge
- Agent quorum
- Use snippets for better generation

## Paper structure6a

- Introduction
- Previous work (summary of 13B modifications)
- Methodology (what was done this year)
- Results (ours and official ones)
  - For our internal results, go to @https://huggingface.co/IEETA/BioASQ-14B/tree/main and, for each folder, check the ranx_results.json file, if exists.
  - For official results, check the @results.md file.
- Discussion (what we learned, what we want to do next)
- Conclusion
