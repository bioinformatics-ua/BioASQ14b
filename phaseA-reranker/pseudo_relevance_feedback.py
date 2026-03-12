import click
import pickle
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collator import RankingCollator
from datasets import Dataset
import torch
from collections import defaultdict
from tqdm import tqdm


@click.command()
@click.argument("testset")
@click.option("--ranx_runs", nargs=24)
@click.option("--model_checkpoints", nargs=24)
def main(testset, ranx_runs, model_checkpoints):
    # ranx_runs = ranx_runs]
    # model_checkpoints = [model_checkpoints]
    print(len(ranx_runs))
    print(len(model_checkpoints))
    print("load collection", flush=True)

    collection = {}
    with open("../data/pubmed_baseline_2025.jsonl") as f:
        for article in map(json.loads, f):
            collection[article["pmid"]] = article["title"] + " " + article["abstract"]

    print("load lookup", flush=True)
    with open("../data/similarity_results/lookup_T0.8.p", "rb") as f:
        lookup = pickle.load(f)

    print("load testset", flush=True)
    with open(testset) as f:
        testset_data = {
            q_data["id"]: q_data["body"] for q_data in json.load(f)["questions"]
        }

    for i in range(len(ranx_runs)):
        ranx_run = ranx_runs[i]

        model_checkpoint = model_checkpoints[i]
        print(model_checkpoints)
        print(model_checkpoint)
        is_pairwise = "pairwise" in model_checkpoint
        print(f"Running in {'pairwise' if is_pairwise else 'pointwise'} mode")

        print("load run", flush=True)
        with open(ranx_run) as f:
            run = json.load(f)

        print("load model", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(
            "cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        MAX_LENGTH = 512
        tokenizer.model_max_length = MAX_LENGTH

        def semantic_search_based_on_list_ids(list_ids, th=0.8, topk=1):
            expanding_results = set()
            for doc_id in list_ids:
                for d_id, score in lookup[doc_id][:topk]:
                    if score > th:
                        expanding_results.add(d_id)
            return expanding_results

        print("semantic find and score", flush=True)

        for q_data_id, documents in tqdm(run.items()):
            doc_ids = list(documents.keys())

            new_ids = semantic_search_based_on_list_ids(doc_ids[:50], 0.8, 75)
            new_ids = new_ids - set(doc_ids)

            def gen_docs_pairs():
                for doc_id in new_ids:
                    q_text = testset_data[q_data_id]
                    doc_text = collection[doc_id]
                    inputs = tokenizer(
                        q_text, doc_text, truncation=True, max_length=MAX_LENGTH
                    )
                    yield inputs | {"id": q_data_id, "doc_id": doc_id}

            # prepare new docs for inference
            class IterDataset(torch.utils.data.IterableDataset):
                def __init__(self, generator):
                    self.generator = generator

                def __iter__(self):
                    return self.generator()

            dl = torch.utils.data.DataLoader(
                IterDataset(gen_docs_pairs),
                batch_size=128,
                collate_fn=RankingCollator(tokenizer=tokenizer),
            )

            with torch.no_grad():
                for sample in dl:
                    if is_pairwise:
                        scores = (
                            model(**sample["inputs"].to("cuda"))
                            .logits.squeeze()
                            .cpu()
                            .tolist()
                        )
                    else:
                        logits = model(**sample["inputs"].to("cuda")).logits
                        scores = (
                            torch.nn.functional.softmax(logits, dim=-1)[:, 1]
                            .cpu()
                            .tolist()
                        )

                    for i, doc_id in enumerate(sample["doc_id"]):
                        if type(scores) == list:
                            run[q_data_id][doc_id] = scores[i]
                        else:
                            run[q_data_id][doc_id] = scores
            # sort by value
            run[q_data_id] = dict(
                sorted(run[q_data_id].items(), key=lambda x: x[1], reverse=True)
            )

        out_run = ranx_run[:-5]
        print("write")

        outfile = f"{out_run}_dprf.json".replace("2025", "2025_dprf")

        with open(outfile, "w") as f:
            json.dump(run, f)


if __name__ == "__main__":
    main()
