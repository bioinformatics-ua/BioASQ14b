from duckdb import DuckDBPyConnection
import msgspec
import os
import sys
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Ensure project root is on path for data.baseline_duckdb
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import orjson
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collator import RankingCollator
import torch
from tqdm import tqdm
import typer
from data.baseline_duckdb import (
    get_article,
    get_articles_batch,
    get_lookup_entries_batch,
    init_lookup_table,
    init_pubmed_db,
)
from model import TestCollection, Lookup, LookupEntry, PMID

app = typer.Typer()


def _sanitize_position_ids_buffers(model: torch.nn.Module) -> None:
    """Ensure any `position_ids` buffers are monotonic 0..N-1.

    Some remote-code checkpoints may load a corrupted `position_ids` buffer,
    which later causes out-of-bounds indexing in positional embeddings/RoPE.
    """
    for module in model.modules():
        position_ids = getattr(module, "position_ids", None)
        if not isinstance(position_ids, torch.Tensor):
            continue
        if position_ids.dtype not in (torch.int32, torch.int64):
            continue
        if position_ids.ndim == 1:
            expected = torch.arange(
                position_ids.shape[0],
                device=position_ids.device,
                dtype=position_ids.dtype,
            )
        elif position_ids.ndim == 2 and position_ids.shape[0] == 1:
            expected = torch.arange(
                position_ids.shape[1],
                device=position_ids.device,
                dtype=position_ids.dtype,
            ).unsqueeze(0)
        else:
            continue

        if not torch.equal(position_ids, expected):
            position_ids.copy_(expected)


@app.command()
def main(
    testset: Path = typer.Argument(..., help="Path to testset."),
    ranx_runs: list[Path] = typer.Option(
        ..., "--ranx-run", "-r", help="Paths to ranx runs."
    ),
    model_checkpoints: list[Path] = typer.Option(
        ...,
        "--model",
        "-M",
        help="Paths to model checkpoints.",
    ),
    revisions: list[str] = typer.Option(
        ...,
        "--revision",
        "-R",
        help="Revisions of the models.",
    ),
    baseline: Path = typer.Option(
        Path("../data/pubmed_baseline_2026.jsonl"),
        "-b",
        "--baseline",
        help="Path to baseline JSONL.",
    ),
    baseline_db: Path | None = typer.Option(
        Path("../data/db_baselines/pubmed_baseline_2026.db"),
        "--baseline-db",
        help="Path to DuckDB file (default: derived from baseline path).",
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
    print("collect model checkpoints", flush=True)
    if not model_checkpoints:
        raise ValueError("At least one model checkpoint path is required")

    if not revisions:
        raise ValueError("At least one revision is required")

    ranx_runs = sorted(ranx_runs)

    if len(ranx_runs) != len(model_checkpoints) != len(revisions):
        raise ValueError(
            f"Number of ranx runs, model checkpoints, and revisions must match. Got {len(ranx_runs)}, {len(model_checkpoints)}, {len(revisions)}"
        )

    print("init baseline db", flush=True)
    db_path = baseline_db or (baseline.parent / "db_baselines" / f"{baseline.stem}.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con: DuckDBPyConnection = init_pubmed_db(baseline, db_path)

    # print("init lookup table (DuckDB)", flush=True)
    # init_lookup_table(con, lookup_path)

    print("load lookup", flush=True)
    with lookup_path.open("rb") as f:
        lookup: dict[PMID, list[tuple[PMID, float]]] = orjson.loads(f.read())

    print("load testset", flush=True)
    with open(testset) as f:
        decoder = msgspec.json.Decoder(TestCollection)
        testset_data = {
            q_data.id: q_data.body for q_data in decoder.decode(f.read()).questions
        }

    for ranx_run, model_checkpoint, revision in tqdm(
        zip(ranx_runs, model_checkpoints, revisions),
        desc="Processing runs",
        unit="run",
    ):
        is_pairwise = "pairwise" in model_checkpoint.name
        print(f"Running in {'pairwise' if is_pairwise else 'pointwise'} mode")

        print("load run", flush=True)
        with ranx_run.open("rb") as f:
            run = orjson.loads(f.read())

        print("load model", flush=True)
        is_local = model_checkpoint.exists()
        if is_local:
            model_id = str(model_checkpoint.resolve())
            load_kwargs = dict(
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        else:
            model_id = str(model_checkpoint)  # Hub ID e.g. "IEETA/BioASQ-13B"
            load_kwargs = dict(
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                revision=revision,
            )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            **load_kwargs,
        ).to("cuda")
        # fix the nemotron model ids
        _sanitize_position_ids_buffers(model)
        if getattr(model.config, "num_labels", 1) != 1:
            model.config.num_labels = 1
            model.config.id2label = {0: "SCORE"}
            model.config.label2id = {"SCORE": 0}
        tokenizer_kwargs = dict(trust_remote_code=True)
        if is_local:
            tokenizer_kwargs["local_files_only"] = True
            tokenizer_path = str(model_checkpoint.resolve())
        else:
            tokenizer_kwargs["revision"] = revision
            tokenizer_path = str(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs,
        )
        if not tokenizer:  # Just to shut up pyright and ty
            raise ValueError(f"Tokenizer not found for {model_checkpoint}")
        tokenizer.model_max_length = max_length

        def semantic_search_based_on_list_ids(list_ids, th=0.9, topk=1):
            expanding_results = set()
            for doc_id in list_ids:
                if doc_id not in lookup:
                    continue
                for pmid, score in lookup[doc_id][:topk]:
                    if score > th:
                        expanding_results.add(pmid)
            return expanding_results

        print("semantic find and score", flush=True)

        for q_data_id, documents in tqdm(run.items()):
            doc_ids = list(documents.keys())

            new_ids = semantic_search_based_on_list_ids(doc_ids[:50], 0.8, 75)
            new_ids = new_ids - set(doc_ids)

            # Batch fetch baseline texts for all new_ids (single fallback for any missing)
            batch_texts = get_articles_batch(con, list(new_ids)) if new_ids else {}

            def _get_doc_text(doc_id: str) -> str | None:
                text = batch_texts.get(doc_id)
                if text is not None:
                    return text
                # Single fallback if pmid was missing from batch (e.g. edge case)
                article = get_article(con, doc_id)
                if article:
                    return f"{article['title']} {article['abstract']}"
                return None

            def gen_docs_pairs():
                for doc_id in new_ids:
                    doc_text = _get_doc_text(doc_id)
                    if doc_text is None:
                        continue
                    q_text = testset_data[q_data_id]
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
                batch_size=4,
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
                        if logits.shape[-1] == 1:
                            scores = logits.squeeze(-1).cpu().tolist()
                        else:
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

        out_run = ranx_run.parent.stem + "_" + ranx_run.stem

        outfile = output_dir / f"{out_run}_dprf.json"
        outfile.parent.mkdir(parents=True, exist_ok=True)

        with outfile.open("wb") as f:
            f.write(orjson.dumps(run))


if __name__ == "__main__":
    app()
