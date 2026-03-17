from collections import defaultdict
import orjson
from pathlib import Path
import pickle
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TokenizersBackend,
)
from collator import RankingCollator
import torch
from tqdm import tqdm
import typer

type PMID = str
app = typer.Typer()


@app.command()
def main(
    testset: Path = typer.Argument(..., help="Path to testset."),
    ranx_runs: Path = typer.Option(..., help="Path to ranx runs."),
    model_checkpoints: Path = typer.Option(..., help="Path to model checkpoints."),
    baseline: Path = typer.Option(
        Path("../data/pubmed_baseline_2026.jsonl"),
        "-b",
        "--baseline",
        help="Path to baseline.",
    ),
    lookup_path: Path = typer.Option(
        Path("../data/similarity_results/lookup.json"),
        "-l",
        "--lookup",
        help="Path to lookup.",
    ),
    output_dir: Path = typer.Option(
        Path("../dprf"),
        "-o",
        "--output",
        help="Path to output directory.",
    ),
    max_length: int = typer.Option(
        512, "-m", "--max-length", help="Maximum length of the input text."
    ),
):
    with baseline.open("rb") as f:
        collection: dict[PMID, str] = {
            article["pmid"]: article["title"] + " " + article["abstract"]
            for article in map(orjson.loads, f)
        }

    print("load lookup", flush=True)
    with lookup_path.open("rb") as f:
        lookup: dict[PMID, list[tuple[PMID, float]]] = orjson.loads(f.read())

    print("load testset", flush=True)
    with open(testset) as f:
        testset_data = {
            q_data["id"]: q_data["body"]
            for q_data in orjson.loads(f.read())["questions"]
        }

    for ranx_run, model_checkpoint in tqdm(
        zip(ranx_runs.iterdir(), model_checkpoints.iterdir()),
        desc="Processing runs",
        unit="run",
    ):
        is_pairwise = "pairwise" in model_checkpoint.name
        print(f"Running in {'pairwise' if is_pairwise else 'pointwise'} mode")

        print("load run", flush=True)
        with ranx_run.open("rb") as f:
            run = orjson.loads(f.read())

        print("load model", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(
            "cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        if not tokenizer:  # Just to shut up pyright and ty
            raise ValueError(f"Tokenizer not found for {model_checkpoint}")
        tokenizer.model_max_length = max_length

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
                        q_text, doc_text, truncation=True, max_length=max_length
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
                        if type(scores) is list:
                            run[q_data_id][doc_id] = scores[i]
                        else:
                            run[q_data_id][doc_id] = scores
            # sort by value
            run[q_data_id] = dict(
                sorted(run[q_data_id].items(), key=lambda x: x[1], reverse=True)
            )

        out_run = ranx_run[:-5]

        outfile = output_dir / f"{out_run}_dprf.json"
        outfile.parent.mkdir(parents=True, exist_ok=True)

        with outfile.open("wb") as f:
            f.write(orjson.dumps(run))


if __name__ == "__main__":
    main()
