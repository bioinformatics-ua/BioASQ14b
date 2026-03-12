import pickle
import json

print("load lookup", flush=True)
with open("similarity_results/lookup_T0.8.p", "rb") as f:
    lookup = pickle.load(f)

with open("ids_per_baseline_2025.p", "rb") as f:
    ids_per_baseline = {k: set(v) for k, v in pickle.load(f).items()}

with open("pubmed_baseline_2025.jsonl") as f:
    collection = {
        doc["pmid"]: doc["title"] + " " + doc["abstract"] for doc in map(json.loads, f)
    }


def semantic_search_based_on_list_ids(list_ids, th=0.8, topk=1):
    expanding_results = set()
    for doc_id in list_ids:
        for d_id, score in lookup[doc_id]:
            if score > th:
                expanding_results.add(d_id)
    return expanding_results


with (
    open("quality/training13b_inflated_clean_wContents.jsonl.jsonl") as f,
    open(
        "quality/training13b_inflated_clean_wContents_dense_expanded.jsonl", "w"
    ) as fout,
):
    for qdata in map(json.loads, f):
        all_pos_docs = {doc["id"] for doc in qdata["documents"]}
        expanded_docs_095 = (
            semantic_search_based_on_list_ids(all_pos_docs, th=0.95)
            - all_pos_docs
            - ids_per_baseline[qdata["baseline"]]
        )
        expanded_docs_09 = (
            semantic_search_based_on_list_ids(all_pos_docs, th=0.9)
            - all_pos_docs
            - expanded_docs_095
            - ids_per_baseline[qdata["baseline"]]
        )
        expanded_docs_085 = (
            semantic_search_based_on_list_ids(all_pos_docs, th=0.85)
            - all_pos_docs
            - expanded_docs_09
            - ids_per_baseline[qdata["baseline"]]
        )
        expanded_docs_08 = (
            semantic_search_based_on_list_ids(all_pos_docs)
            - all_pos_docs
            - expanded_docs_085
            - ids_per_baseline[qdata["baseline"]]
        )

        qdata["expanded_docs_095"] = list(
            map(lambda x: {"id": x, "text": collection[x]}, expanded_docs_095)
        )
        qdata["expanded_docs_09"] = list(
            map(lambda x: {"id": x, "text": collection[x]}, expanded_docs_09)
        )
        qdata["expanded_docs_085"] = list(
            map(lambda x: {"id": x, "text": collection[x]}, expanded_docs_085)
        )
        qdata["expanded_docs_08"] = list(
            map(lambda x: {"id": x, "text": collection[x]}, expanded_docs_08)
        )

        fout.write(f"{json.dumps(qdata)}\n")
