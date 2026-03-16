from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from ranker_trainer import RankerTrainer, RankingEvalPrediction
from metrics import RanxMetrics
from collator import RankingCollator
from utils import create_config, load_rank_data
from data2 import create_test_dataset
from sample_preprocessing import BasicSamplePreprocessing
from optimum.bettertransformer import BetterTransformer
from tqdm import tqdm


from collections import defaultdict
import torch
import os
from ranx import Qrels, Run
import click
import re

## add the other here
MODEL_NAME = {
    "michiyasunaga-BioLinkBERT-large": "BioL-L",
}


@click.command()
# @click.argument("checkpoint")
@click.option("--checkpoint")
@click.option("--revision", default="")
@click.option("--baseline_path")
# @click.argument("baseline_path")
@click.option("--val_files", default=None)
@click.option("--sp_mode", default="basic")
@click.option("--at", default=1000)
@click.option("--path_to_save", default=".")
def main(checkpoint, revision, baseline_path, val_files, sp_mode, at, path_to_save):

    # flag to check if the model is pairwise
    is_pairwise = ("pairwise" in checkpoint) or ("pairwise" in revision)
    print(f"Running in {'pairwise' if is_pairwise else 'pointwise'} mode")

    model_checkpoint = checkpoint

    print(model_checkpoint, revision)

    if revision != "":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            revision=revision,
            cache_dir="/data/bioasq13/phaseA-reranker/HF_CACHE",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            revision=revision,
            cache_dir="/data/bioasq13/phaseA-reranker/HF_CACHE/.dummy",
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.model_max_length = 512

    ## TODO: get that info from the model config
    if sp_mode == "basic":
        sample_preprocessing = BasicSamplePreprocessing(tokenizer)
    else:
        raise RuntimeError("Invalid sample class")

    test_inference_dataset = create_test_dataset(
        baseline_path, sample_preprocessing, val_files
    )

    # model = BetterTransformer.transform(model, keep_original_model=True)
    model = model.to("cuda")

    test_dl = torch.utils.data.DataLoader(
        test_inference_dataset,
        batch_size=128,
        collate_fn=RankingCollator(tokenizer=tokenizer),
    )

    run_dict = defaultdict(dict)

    with torch.no_grad():
        for sample in tqdm(test_dl):
            if is_pairwise:
                doc_score = model(**sample["inputs"].to("cuda")).logits.squeeze().cpu()
            else:
                logits = model(**sample["inputs"].to("cuda")).logits
                doc_score = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu()

            for i in range(doc_score.shape[0]):
                run_dict[sample["id"][i]][sample["doc_id"][i]] = doc_score[i].item()

    run = Run(run_dict)

    # if is_pairwise:
    #     new_run_dict = {}
    #     for q_id, docs_dict in run.items():
    #         new_run_dict[q_id] = {}

    #         for doc_id, doc_score in docs_dict.items():
    #             if doc_score<=0.01:
    #                 continue
    #             new_run_dict[q_id][doc_id]=doc_score
    #     run = Run(new_run_dict)
    # save ranx
    # flat_name = checkpoint.replace("/","_")
    flat_name = checkpoint.split("/")[-2] + "_" + checkpoint.split("/")[-1]

    baseline_name = baseline_path.split("/")[-1][:-6]

    name_begining = f"ranx_{flat_name}_{baseline_name}"

    if revision != "":
        name_begining = f"ranx_{flat_name}_{revision}_{baseline_name}"

    run.save(os.path.join(path_to_save, f"{name_begining}_{at}.json"))


if __name__ == "__main__":
    main()
