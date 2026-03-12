from grid_search import load_index
import pandas as pd
import json
import click


def get_queries(filepath):

    # queryid2text = {}

    with open(filepath, "r") as file:
        content = json.load(file)

        queries = [
            {"qid": question["id"], "query": question["body"]}
            for question in content["questions"]
        ]
        queryid2text = {
            question["id"]: question["body"] for question in content["questions"]
        }
    return queries, queryid2text


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

        print(f"done")
        return doc_content

    lst_docs = [doc["id"] for qid, neg_docs in docs.items() for doc in neg_docs]

    doc_content = load_content(baseline, lst_docs)

    for qid, neg_docs in docs.items():
        neg_docs_wContent = []
        for doc in neg_docs:
            pid = doc["id"]
            if pid not in doc_content:
                text = {"text": ""}
                print("error " + pid)
            else:
                text = {"text": doc_content[pid]}

            neg_docs_wContent.append({**doc, **text})

        docs[qid] = neg_docs_wContent

    return docs


@click.command()
@click.argument("testset_file")
@click.argument("output_file")
@click.option("--baseline", type=str, default="2025")
@click.option("--topk", default=1000)
@click.option("--k1", default=0.4)
@click.option("--b", default=0.3)
@click.option("--add_contents", is_flag=True)
def main(testset_file, output_file, baseline, topk, k1, b, add_contents):
    # best bm25 params based on the grid search results (see more in analyze_grid_search.py)
    k1 = 0.4
    b = 0.3

    queries, queryid2text = get_queries(
        testset_file
    )  # get list of queries for each baseline

    index = load_index(baseline)

    bm25 = index.bm25(k1=k1, b=b, num_results=topk, threads=32)

    questions_dataframe = pd.DataFrame(queries)
    # results = { qid : [ {"id": row["docno"], "score": row["score"] } for _, row in results.iterrows() ]

    results = {
        qid: [{"id": row["docno"]} for _, row in results.iterrows()]
        for qid, results in bm25.transform(questions_dataframe).groupby("qid")
    }

    if add_contents:
        results = add_content("2025", results)
        # results = add_content(baseline, results)

    with open(output_file, "w") as f:
        for qid, docs in results.items():
            # out = { "id": qid, "documents": docs }
            out = {"id": qid, "query_text": queryid2text[qid], "bm25": docs}

            f.write(f"{json.dumps(out)}\n")


if __name__ == "__main__":
    main()
