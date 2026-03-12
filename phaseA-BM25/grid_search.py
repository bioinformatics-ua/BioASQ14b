import pyterrier as pt

if not pt.started():
    pt.init()
import pandas as pd
from pyterrier_pisa import PisaIndex
from itertools import product
from collections import defaultdict
from ranx import Qrels, Run
from ranx import evaluate
import json, os, sys


def load_index(baseline):
    index_path = "../../data/indexes/baseline_" + str(baseline) + "/"

    if not os.path.exists(index_path):
        print(f"[ERROR]: Index '{index_path}' doesn't exist.")
        sys.exit(-1)

    index = PisaIndex(index_path, text_field="text", threads=32)
    print(f"Pisa index '{index_path}' loaded.")
    return index


def get_queries(filename):
    qrels_dict = {}
    queries = {}
    queryid2text = {}

    with open(filename, "r") as f:
        for question in map(json.loads, f):
            qid = question["id"]
            doc_ids = question["documents"]
            query = question["body"]
            baseline = question["baseline"]

            queryid2text[qid] = query

            if baseline not in queries.keys():
                queries[baseline] = [{"qid": qid, "query": query.lower()}]
            else:
                queries[baseline].append({"qid": qid, "query": query.lower()})

            do_the_doc_have_content = (
                isinstance(doc_ids, list) and doc_ids
            ) and isinstance(doc_ids[0], dict)
            if do_the_doc_have_content:
                doc_ids = [doc["id"] for doc in doc_ids]

            if baseline not in qrels_dict.keys():
                qrels_dict[baseline] = {qid: {did: 1 for did in doc_ids}}
            else:
                qrels_dict[baseline].update({qid: {did: 1 for did in doc_ids}})

    return queries, qrels_dict, queryid2text


def get_metrics(qrels_dict, run_dict):
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    # metrics = ["hits", "hit_rate", "precision", "recall", "f1", "r-precision", "mrr", "map", "map@5", "ndcg", "ndcg_burges"]
    metrics = [
        "recall@1000",
        "recall@100",
        "recall@10",
        "map@1000",
        "map@100",
        "map@10",
        "ndcg@1000",
        "ndcg@100",
        "ndcg@10",
    ]

    metrics = [
        "recall@1000",
        "recall@200",
        "recall@100",
        "recall@10",
        "map@1000",
        "map@200",
        "map@100",
        "map@10",
        "ndcg@1000",
        "ndcg@200",
        "ndcg@100",
        "ndcg@10",
    ]
    return evaluate(qrels, run, metrics)


def write_results(baseline, results, results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"Folder '{results_path}' has been created.")

    with open(f"{results_path}baseline_{baseline}.json", "w") as f:
        json.dump(results, f)


def calculate_average_evaluation(results_path):
    sum_scores = defaultdict(lambda: defaultdict(float))
    num_baselines = 0

    for filename in os.listdir(results_path):
        baseline_results = os.path.join(results_path, filename)

        if os.path.isfile(baseline_results):
            num_baselines += 1

            with open(baseline_results, "r") as file:
                results = json.load(file)

                for params, metrics in results.items():
                    for metric, score in metrics.items():
                        sum_scores[params][metric] += score
        else:
            print(
                f"[WARNING]: The file '{baseline_results}' was not found in {results_path}"
            )

    avg_results = {}
    for params, metrics in sum_scores.items():
        avg_results[params] = {
            metric: value / num_baselines for metric, value in metrics.items()
        }

    with open(f"{results_path}bm25_avg.json", "w") as f:
        json.dump(avg_results, f)


def main():
    """main"""
    # bm25 params
    k1_lst = [i / 10 for i in range(1, 13)]  # interval: [0.1, 1.2]
    b_lst = [i / 10 for i in range(1, 11)]  # interval: [0.1, 1.0]

    combinations = list(product(k1_lst, b_lst))

    queries_path = "../../data/training/training12b_inflated_clean.jsonl"
    queries, qrels_dict = get_queries(
        queries_path
    )  # get list of queries for each baseline

    for baseline, query_list in queries.items():
        index = load_index(baseline)  # load index based on the baseline

        evaluation = {}
        for k1, b in combinations:
            print(f"\n>> baseline {baseline} (k1: {k1}, b: {b})")

            bm25 = index.bm25(
                k1=k1, b=b, num_results=1000, threads=32
            )  # create bm25 index with k1 and b parameters

            questions_dataframe = pd.DataFrame([query for query in query_list])

            run_dict = {
                qid: {row["docno"]: row["score"] for _, row in results.iterrows()}
                for qid, results in bm25.transform(questions_dataframe).groupby("qid")
            }  # index search for each query

            evaluation[str((k1, b))] = get_metrics(
                qrels_dict[baseline], run_dict
            )  # evaluate results

        write_results(
            baseline, evaluation, results_path="../results/grid_search/"
        )  # write the evaluations for each baseline

        # os.system(f'rm ../../data/indexes/baseline_{baseline}/bm25.*')  # delete unnecessary files

    calculate_average_evaluation(results_path="../results/grid_search/")


if __name__ == "__main__":
    main()
