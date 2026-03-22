# TODO

## Common
- create proper scripts & full pipeline
- decursify code (autistic mode)
- lookup for rust-based things
- check if numrs2 can be fitted into this
- strongly type code with common types
- use msgspec instead of orjson
- use typer for all CLIs
- fix outputs
- document submissions

## Data
- shard partitioning
- reorganize codebase, allow for python modules
- other ways/frameworks to index, fuck pisa (duckdb)
- db for embeddings (pgvectorscale)
- try to check if there's a local non-rest api for TEI
- schedule documents by year?
- embeddings **URGENT**

## Phase A
- snippets?
- exact answer?
- use embeds, merge with bm25
- bm25 visualization, and verify how many docs are optimal
- parallelize expanded positives
- hard negatives, training on documents the reranker gets good
- expand positives? how cna we filter out easy negatives from this. maybe we run reranker on the dense matrix
- testset inference (bm25)

## Phase B
- run local and cloud at the same time
- review llm-as-a-judge
- metric hallucination
- ensembling thingssss
- quantization
- temperature shit (recommended but try to check if there are optimal values)
- review agents
- subadgents that decide together ideal answers
