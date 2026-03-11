import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from data2 import (
    create_bioASQ_datasets,
    BioASQPointwiseIterator,
    BioASQPairwiseIterator,
    BioASQRelevanceAwarePairwiseIterator,
)
from sampler import (
    BasicSampler,
    HigherConfidenceNegativesSampler,
    ExponentialWeightSampler,
    BasicV2Sampler,
)
from collator import RankingCollator, PairwiseCollator

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, Trainer
from ranker_trainerv2 import PairwiseTrainer
from metrics import RanxMetrics
from utils import setup_wandb, create_config
from collections import defaultdict
import torch
import argparse
from sample_preprocessing import BasicSamplePreprocessing
# from optimum.bettertransformer import BetterTransformer

from trainer_callbacks import ResampleByRerankerCallback
import glob
import os
from ranx import Qrels, Run
from ranx import evaluate
from functools import partial


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("val", type=str)  # action='store_true'
    parser.add_argument("data", type=str)

    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--sampler", type=str, default="basic")
    parser.add_argument("--sample_preprocessing", type=str, default="basic")
    parser.add_argument("--results_file", type=str, default="val_results_b02.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_neg_samples", type=int, default=1)
    parser.add_argument("--pairwise", action="store_true")
    parser.add_argument("--use_expanded_pos", action="store_true")
    parser.add_argument("--warmup_ratio", action="store_true")

    parser.add_argument("--callback", action="store_true")

    args = parser.parse_args()

    model_checkpoint = args.checkpoint

    BASE_DIR = "/home/ucloud/BioASQ13B/phaseA-reranker/"

    if args.data == "quality":
        if args.use_expanded_pos:
            positives = "../data/quality/training14b_inflated_clean_wContents_dense_expanded.jsonl"
        else:
            positives = (
                "../data/quality/training14b_inflated_clean_wContents.jsonl"  # this
            )
        # all_path = "/data/bioasq13/data/old_data.jsonl"
        all_path = "../data/quality/hard_negatives_IA_clean.jsonl"  # this
        # all_path = "/data/bioasq13/data/quality/hard_negatives_IA_clean_fixed_new.jsonl"

    # if args.data == "quantity":
    #     if args.use_expanded_pos:
    #         positives = "../data/quality/training13b_inflated_clean_wContents_dense_expanded.jsonl"
    #     else:
    #         positives = "../data/quantity/training13b_inflated_full_wContents.jsonl"
    #     # all_path = "/data/bioasq13/data/old_data.jsonl"
    #     all_path = "../data/quantity/hard_negatives_IA_complete.jsonl"
    #     # all_path = "/data/bioasq13/data/quantity/hard_negatives_IA_complete_fixed.jsonl"

    if args.val == "True":
        # print("we need files")
        val = "val"
        eval_files = [
            "../data/val_data/13B1_golden.json",
            "../data/val_data/13B2_golden.json",
            # "../data/val_data/12B1_golden.json",
            # "../data/val_data/12B2_golden.json",
            # "../data/val_data/12B3_golden.json",
            # "../data/val_data/12B4_golden.json"
        ]

    elif args.val == "False":
        val = "full"
        eval_files = []

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.model_max_length = 512

    if args.use_expanded_pos:
        relevance_mapping = {
            "documents": 5,
            "expanded_docs_095": 4,
            # "expanded_docs_09": 3,
            # "expanded_docs_085":2,
            # "expanded_docs_08":1
        }
    else:
        relevance_mapping = {"documents": 1}

    if args.pairwise:
        if args.use_expanded_pos:
            iterator_class = BioASQRelevanceAwarePairwiseIterator
        else:
            iterator_class = BioASQPairwiseIterator
        trainer_class = PairwiseTrainer
        train_data_collator = PairwiseCollator(tokenizer=tokenizer)
        trainer_def = "pairwise"
        print("WE MANUALLY HALVE THE BATCH TO ACCUMUDATE THE PAIRWISE TRAINING")
        args.batch = args.batch // 2

        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=1
        )
    else:
        iterator_class = BioASQPointwiseIterator
        trainer_class = Trainer
        trainer_def = "pointwise"
        train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        id2label = {0: "IRRELEVANT", 1: "RELEVANT"}
        label2id = {"IRRELEVANT": 0, "RELEVANT": 1}

        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
        )

    if args.sampler == "basic":
        sampler = BasicSampler
    elif args.sampler == "basicv2":
        sampler = BasicV2Sampler
    elif args.sampler == "exponential":
        assert args.use_expanded_pos, (
            "exponential sample only works with use_expanded_pos"
        )
        sampler = ExponentialWeightSampler
    else:
        raise RuntimeError("Invalid sample class")

    if args.sample_preprocessing == "basic":
        sample_preprocessing = BasicSamplePreprocessing(tokenizer)
    else:
        raise RuntimeError("Invalid sample class")

    if args.warmup_ratio:
        warmup_ratio = 0.1
    else:
        warmup_ratio = 0.0

    out_dir_name = f"{model_checkpoint.replace('/', '-')}-{args.seed}-E{args.epoch}-S{args.sampler}-SP{args.sample_preprocessing}-{val}-{args.data}_data-CB{args.callback}-KN{args.num_neg_samples}-GA{args.gradient_accumulation_steps}-TR{trainer_def}-ExPOS{args.use_expanded_pos}-warmup{args.warmup_ratio}"
    print(out_dir_name)
    # setup_wandb("BioASQ 13b - Batch 2 - "+val, out_dir_name)
    training_args = create_config(
        BASE_DIR + "bert_trainer_config.yaml",
        output_dir=BASE_DIR + "/trained_models_tmp/" + out_dir_name,
        dataloader_num_workers=1,
        seed=args.seed,
        #   eval_steps = 10,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="no",
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch,
        remove_unused_columns=False,
        warmup_ratio=warmup_ratio,
    )

    train_ds, test_ds = create_bioASQ_datasets(
        positive_data_path=positives,
        all_data_path=all_path,
        test_sample_preprocessing=sample_preprocessing,
        iterator=iterator_class(
            sample_preprocessing=sample_preprocessing,
            sampler_class=sampler,
            num_neg_samples=args.num_neg_samples,
        ),
        val_files=eval_files,
        relevance_mapping=relevance_mapping,
    )

    # .to("cuda")

    model.config.sample_preprocessing = args.sample_preprocessing
    model.config.num_neg_samples = args.num_neg_samples
    model.config.sampler = args.sampler

    if args.callback:
        callback = [
            ResampleByRerankerCallback(
                train_dataset=train_ds,
                tokenizer=tokenizer,
                start_epoch=0,
                interval=-1,
                num_high_confidence_to_remove=1,
            )
        ]

    else:
        callback = []

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=train_data_collator,
        callbacks=callback,
    )

    trainer.train()

    if args.val == "True":
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=512, collate_fn=RankingCollator(tokenizer=tokenizer)
        )

        run_dict = defaultdict(dict)

        model = model.to("cuda")

        results = {}
        with torch.no_grad():
            for sample in tqdm(test_dl):
                if args.pairwise:
                    doc_score = (
                        model(**sample["inputs"].to("cuda")).logits.squeeze().cpu()
                    )
                else:
                    logits = model(**sample["inputs"].to("cuda")).logits
                    doc_score = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu()

                for i in range(doc_score.shape[0]):
                    run_dict[sample["id"][i]][sample["doc_id"][i]] = doc_score[i].item()

        # evaluate
        qrels = Qrels(test_ds.get_qrels())
        run = Run(run_dict)
        results["total"] = evaluate(
            qrels,
            run,
            [
                "ndcg@5",
                "mrr",
                "recall@10",
                "recall@100",
                "recall@1000",
                "map@10",
                "map-bioasq@10",
            ],
        )

        for file in eval_files:
            with open(file) as f:
                data = json.load(f)
            qids = {i["id"] for i in data["questions"]}

            b_run_dict = {k: v for k, v in run_dict.items() if k in qids}
            b_qrels_dict = {k: v for k, v in test_ds.get_qrels().items() if k in qids}
            results[os.path.basename(file)] = evaluate(
                Qrels(b_qrels_dict),
                Run(b_run_dict),
                [
                    "ndcg@5",
                    "mrr",
                    "recall@10",
                    "recall@100",
                    "recall@1000",
                    "map@10",
                    "map-bioasq@10",
                ],
            )

        print(results)

        with open(BASE_DIR + args.results_file, "a") as f:
            metadata = {
                "model": model_checkpoint.replace("/", "-"),
                "seed": args.seed,
                "epoch": args.epoch,
                "sampler": args.sampler,
                "sample_preprocessing": args.sample_preprocessing,
                "callback": args.callback,
                "data_type": args.data + "_data",
                "num_neg_samples": args.num_neg_samples,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "trainer": trainer_def,
                "use_expanded_pos_docs": args.use_expanded_pos,
                "relevance_mapping": relevance_mapping,
                "warmup_ratio": warmup_ratio,
            }
            f.write(f"{json.dumps(metadata | results)}\n")

    files = glob.glob(f"{BASE_DIR}/trained_models_b02/{out_dir_name}/*/*.pt")

    for file_path in files:
        os.remove(file_path)


if __name__ == "__main__":
    main()
