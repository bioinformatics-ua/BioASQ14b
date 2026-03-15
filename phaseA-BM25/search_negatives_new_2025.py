from typing import Literal
import pyterrier as pt

if not pt.started():
    pt.init()
import pandas as pd
import json

from grid_search import load_index, get_queries

type Year = Literal["2024", "2025"]


def add_content(baseline: Year, docs):

    def get_file_path(baseline):
        return "../../data/baselines/pubmed_baseline_" + str(baseline) + ".jsonl"

    def load_content(baseline, lst_docs):
        docs_set = set(lst_docs)
        total = len(docs_set)
        doc_content = {}

        with open(get_file_path(baseline), "r") as f:
            for line in f:
                doc = json.loads(line)
                if doc["pmid"] in docs_set:
                    doc_content[doc["pmid"]] = " ".join([doc["title"], doc["abstract"]])
                    docs_set.remove(doc["pmid"])
                    print(f"{total - len(docs_set)}/{total}", end="\r")

                    if len(doc_content) == total:
                        break

        print(f"{baseline} done")
        return doc_content

    lst_docs = [doc["id"] for qid, neg_docs in docs.items() for doc in neg_docs]

    doc_content = load_content(baseline, lst_docs)

    for qid, neg_docs in docs.items():
        neg_docs_wContent = []
        for doc in neg_docs:
            pid = doc["id"]
            if pid not in doc_content:
                print(pid)
                continue
            text = {"text": doc_content[pid]}

            neg_docs_wContent.append({**doc, **text})

        docs[qid] = neg_docs_wContent

    return docs


def main():
    """main"""
    # best bm25 params based on the grid search results (see more in analyze_grid_search.py)
    k1 = 0.4
    b = 0.3

    queries_path = "../data/training/training14b_inflated_clean_wContents_IA.jsonl"
    # print(get_queries(queries_path))
    queries, qrels_dict, _ = get_queries(
        queries_path
    )  # get list of queries for each baseline

    with open("../results/13b/hard_negatives_IA_clean_fixed.jsonl", "w") as f:
        for baseline, query_list in queries.items():
            if baseline != "2024":
                print("skipping ", baseline)
                continue
            index = load_index("2024")
            print(f">> baseline {baseline} (k1: {k1}, b: {b})\n")

            bm25 = index.bm25(k1=k1, b=b, num_results=1000, threads=32)

            questions_dataframe = pd.DataFrame([query for query in query_list])

            results = {
                qid: [
                    {"id": row["docno"], "score": row["score"]}
                    for _, row in results.iterrows()
                ]
                for qid, results in bm25.transform(questions_dataframe).groupby("qid")
            }

            # write_results(baseline, results, results_path="../results/search/")

            labeled_docs = {
                qid: set(doc_ids.keys())
                for qid, doc_ids in qrels_dict[baseline].items()
            }

            negatives = {
                qid: [doc for doc in res if doc["id"] not in labeled_docs[qid]]
                for qid, res in results.items()
            }

            neg_docs_wContent = add_content("2025", negatives)

            for qid, neg_docs in neg_docs_wContent.items():
                out = {"id": qid, "neg_docs": neg_docs}
                f.write(f"{json.dumps(out)}\n")


if __name__ == "__main__":
    main()
