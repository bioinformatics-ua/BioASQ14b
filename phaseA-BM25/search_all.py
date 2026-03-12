import pyterrier as pt

if not pt.started():
    pt.init()
import pandas as pd
import json

from grid_search import load_index, get_queries, write_results
import click


def add_content(baseline, docs):

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
            text = {"text": doc_content[pid]}

            neg_docs_wContent.append({**doc, **text})

        docs[qid] = neg_docs_wContent

    return docs


@click.command()
@click.argument("queries_path")
@click.argument("output_file")
def main(queries_path, output_file):
    """main"""
    # best bm25 params based on the grid search results (see more in analyze_grid_search.py)
    k1 = 0.4
    b = 0.3

    # queries_path = "../../data/training/training12b_inflated_wContents_IA_complete.jsonl"
    queries, qrels_dict, queryid2text = get_queries(
        queries_path
    )  # get list of queries for each baseline

    # f"../results/hard_negatives_IA_complete.jsonl"
    with open(output_file, "w") as f:
        for baseline, query_list in queries.items():  # ["what ...". "how ..."]
            index = load_index(baseline)
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

            bm25_rank = {
                qid: [doc for doc in res if doc["id"]] for qid, res in results.items()
            }

            bm25_wContent = add_content(baseline, bm25_rank)

            for qid, bm25_r in bm25_wContent.items():
                out = {"id": qid, "query_text": queryid2text[qid], "bm25": bm25_r}
                f.write(f"{json.dumps(out)}\n")


if __name__ == "__main__":
    main()
