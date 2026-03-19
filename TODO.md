# TODO

## phase a
- shard partitioning
- lookup for rust-based things
- try to check if there's a local non-rest api for TEI
- check if numrs2 can be fitted into this
- strongly type code with common types
- use msgspec instead of orjson
- use typer for all CLIs
- reorganize codebase, allow for python modules
- other ways/frameworks to index, fuck pisa
- snippets?
- exact answer?
- retireval for phase A (reranker)
    - use embeds, merge with bm25
    - bm25 visualization, and verify how many docs are optimal
- rerank
    - hard negatives, training on documents the reranker gets good
    - schedul documents by year?
    - expand positives? how cna we filter out easy negatives from this. maybe we run reranker on the dense matrix
- fix outputs
- parallelize expanded positives
- create proper scripts
- decursify code (autistic mode)
