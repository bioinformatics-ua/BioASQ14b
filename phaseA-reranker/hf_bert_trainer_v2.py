import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer,
)
from pathlib import Path
import json
from tqdm import tqdm
from data2 import (
    create_bioASQ_datasets,
    BioASQPointwiseIterator,
)
from sampler import (
    BasicSampler,
    ExponentialWeightSampler,
    BasicV2Sampler,
)
from collator import RankingCollator, RankingCollatorForCasualLM

from utils import create_config
from collections import defaultdict
import torch
from sample_preprocessing import BasicSamplePreprocessing, CausalLMSamplePreprocessing

from trainer_callbacks import ResampleByRerankerCallback
import glob
import os
from ranx import Qrels, Run
from ranx import evaluate
import typer
from typing import Optional

app = typer.Typer()

BASE_DIR = Path(__file__).parent.resolve()


_LLM_FAMILY_KEYWORDS = ("llama", "mistral", "gemma", "falcon", "mpt", "phi", "gpt", "nemotron", "qwen")


def _is_causal_lm(config: AutoConfig) -> bool:
    """Detect whether a model belongs to an LLM family (decoder-only or bidirectional LLM).

    Matches on architecture name substrings so that custom variants like
    LlamaBidirectionalForSequenceClassification are captured correctly.
    """
    architectures = getattr(config, "architectures", []) or []
    arch_str = " ".join(architectures).lower()
    if any(kw in arch_str for kw in _LLM_FAMILY_KEYWORDS):
        return True
    model_type = getattr(config, "model_type", "").lower()
    return any(model_type.startswith(kw) for kw in _LLM_FAMILY_KEYWORDS)


def _parse_torch_dtype(dtype_str: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, None)


def _dtype_load_kwargs(dtype, causal: bool) -> dict:
    """Return the correct dtype kwarg for from_pretrained.

    Custom LLM model repos (e.g. Nemotron) deprecated `torch_dtype` in favour
    of `dtype`.  Standard HuggingFace models use `torch_dtype`.  We pass both
    so whichever the model's from_pretrained accepts takes effect.
    """
    if dtype is None:
        return {}
    if causal:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


@app.command()
def main(
    model_name: str,
    val: str,
    data: str,
    batch: int = 32,
    gradient_accumulation_steps: int = 1,
    epoch: int = 10,
    sampler: str = "basic",
    sample_preprocessing: str = "basic",
    results_file: str = "val_results_b02.jsonl",
    seed: int = 42,
    num_neg_samples: int = 1,
    use_expanded_pos: bool = False,
    warmup_ratio: bool = False,
    callback: bool = False,
    # LLM-specific options
    torch_dtype: str = "bfloat16",
    max_length: int = 512,
    prompt_template: Optional[str] = None,
    # Evaluation-only mode
    eval_only: bool = False,
):
    if data == "quality":
        if use_expanded_pos:
            positives = "../data/quality/training14b_inflated_clean_wContents_dense_expanded.jsonl"
        else:
            positives = (
                "../data/quality/training14b_inflated_clean_wContents.jsonl"
            )
        all_path = "../data/quality/hard_negatives_IA_clean.jsonl"

    if val == "True":
        val = "val"
        eval_files = [
            "../data/val_data/13B1_golden.json",
            "../data/val_data/13B2_golden.json",
        ]
    elif val == "False":
        val = "full"
        eval_files = []

    # ------------------------------------------------------------------ #
    # Model-type detection (load config once; reuse for model loading)
    # ------------------------------------------------------------------ #
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    causal = _is_causal_lm(model_config)
    print(f"Model type detected: {'LLM / decoder-style' if causal else 'encoder (BERT-style)'}")

    # ------------------------------------------------------------------ #
    # Tokenizer
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = max_length

    if causal:
        # LLM rerankers score via the last real token — padding must go left.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # ------------------------------------------------------------------ #
    # Sample preprocessing
    # ------------------------------------------------------------------ #
    if sample_preprocessing == "basic":
        if causal:
            _template = prompt_template or None  # uses DEFAULT_CAUSAL_LM_PROMPT
            kwargs = {} if _template is None else {"prompt_template": _template}
            sample_preprocessing_obj = CausalLMSamplePreprocessing(
                tokenizer, model_max_length=max_length, **kwargs
            )
        else:
            sample_preprocessing_obj = BasicSamplePreprocessing(tokenizer)
    else:
        raise RuntimeError(f"Unknown sample_preprocessing: {sample_preprocessing!r}")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    dtype = _parse_torch_dtype(torch_dtype)
    load_kwargs = {"trust_remote_code": True, "config": model_config, **_dtype_load_kwargs(dtype, causal)}

    if causal:
        # Respect the checkpoint's num_labels so the score head loads cleanly.
        ckpt_num_labels = getattr(model_config, "num_labels", 1)
        model_config.num_labels = ckpt_num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    else:
        id2label = {0: "IRRELEVANT", 1: "RELEVANT"}
        label2id = {"IRRELEVANT": 0, "RELEVANT": 1}
        model_config.num_labels = 2
        model_config.id2label = id2label
        model_config.label2id = label2id
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        )

    # ------------------------------------------------------------------ #
    # Data collators
    # ------------------------------------------------------------------ #
    if causal:
        # No token_type_ids; left-padding already set on tokenizer
        train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        eval_collator = RankingCollatorForCasualLM(tokenizer=tokenizer)
    else:
        train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        eval_collator = RankingCollator(tokenizer=tokenizer)

    # ------------------------------------------------------------------ #
    # Sampler
    # ------------------------------------------------------------------ #
    match sampler:
        case "basic":
            sampler_cls = BasicSampler
        case "basicv2":
            sampler_cls = BasicV2Sampler
        case "exponential":
            assert use_expanded_pos, "exponential sampler only works with use_expanded_pos"
            sampler_cls = ExponentialWeightSampler
        case _:
            raise RuntimeError(f"Invalid sampler: {sampler!r}")

    if use_expanded_pos:
        relevance_mapping = {
            "documents": 5,
            "expanded_docs_095": 4,
        }
    else:
        relevance_mapping = {"documents": 1}

    # ------------------------------------------------------------------ #
    # Training configuration
    # ------------------------------------------------------------------ #
    out_dir_name = (
        f"{model_name.replace('/', '-')}-{seed}-E{epoch}"
        f"-S{sampler_cls.__name__}-SP{sample_preprocessing_obj.__class__.__name__}"
        f"-{val}-{data}_data-CB{callback}-KN{num_neg_samples}"
        f"-GA{gradient_accumulation_steps}-ExPOS{use_expanded_pos}-warmup{warmup_ratio}"
    )
    print(out_dir_name)

    extra_training_kwargs = {}
    if causal:
        if torch_dtype == "bfloat16":
            extra_training_kwargs["bf16"] = True
        elif torch_dtype == "float16":
            extra_training_kwargs["fp16"] = True

    training_args = create_config(
        BASE_DIR / "bert_trainer_config.yaml",
        output_dir=BASE_DIR / "trained_models_tmp" / out_dir_name,
        dataloader_num_workers=1,
        seed=seed,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="no",
        num_train_epochs=epoch,
        per_device_train_batch_size=batch,
        remove_unused_columns=False,
        warmup_ratio=0.1 if warmup_ratio else 0.0,
        **extra_training_kwargs,
    )

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    train_ds, test_ds = create_bioASQ_datasets(
        positive_data_path=positives,
        all_data_path=all_path,
        test_sample_preprocessing=sample_preprocessing_obj,
        iterator=BioASQPointwiseIterator(
            sample_preprocessing=sample_preprocessing_obj,
            sampler_class=sampler_cls,
            num_neg_samples=num_neg_samples,
        ),
        val_files=eval_files,
        relevance_mapping=relevance_mapping,
    )

    model.config.sample_preprocessing = sample_preprocessing
    model.config.num_neg_samples = num_neg_samples
    model.config.sampler = sampler_cls.__name__

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=train_data_collator,
        callbacks=[
            ResampleByRerankerCallback(
                train_dataset=train_ds,
                tokenizer=tokenizer,
                start_epoch=0,
                interval=-1,
                num_high_confidence_to_remove=1,
            )
        ]
        if callback
        else None,
    )

    if eval_only:
        print(f"Running evaluation only from checkpoint: {model_name}")
    else:
        trainer.train()

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    # Run evaluation if val mode is enabled OR if eval_only mode is requested
    if val == "val" or eval_only:
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=512, collate_fn=eval_collator
        )

        run_dict = defaultdict(dict)
        model = model.to("cuda")

        results = {}
        with torch.no_grad():
            for sample in tqdm(test_dl):
                logits = model(**sample["inputs"].to("cuda")).logits
                if causal or logits.shape[-1] == 1:
                    # Scalar score from a single output node
                    doc_score = logits.squeeze(-1).cpu()
                else:
                    # Two-class BERT-style: take prob of the positive class
                    doc_score = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu()

                for i in range(doc_score.shape[0]):
                    run_dict[sample["id"][i]][sample["doc_id"][i]] = doc_score[i].item()

        metrics = ["ndcg@5", "mrr", "recall@10", "recall@100", "recall@1000", "map@10", "map-bioasq@10"]

        qrels = Qrels(test_ds.get_qrels())
        run = Run(run_dict)
        results["total"] = evaluate(qrels, run, metrics)

        for file in eval_files:
            with open(file) as f:
                file_data = json.load(f)
            qids = {i["id"] for i in file_data["questions"]}

            b_run_dict = {k: v for k, v in run_dict.items() if k in qids}
            b_qrels_dict = {k: v for k, v in test_ds.get_qrels().items() if k in qids}
            results[os.path.basename(file)] = evaluate(
                Qrels(b_qrels_dict), Run(b_run_dict), metrics
            )

        print(results)

        with open(BASE_DIR / results_file, "a") as f:
            metadata = {
                "model": model_name.replace("/", "-"),
                "seed": seed,
                "epoch": epoch,
                "sampler": sampler,
                "sample_preprocessing": sample_preprocessing,
                "callback": callback,
                "data_type": data + "_data",
                "num_neg_samples": num_neg_samples,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "use_expanded_pos_docs": use_expanded_pos,
                "relevance_mapping": relevance_mapping,
                "warmup_ratio": warmup_ratio,
                "causal_lm": causal,
                "torch_dtype": torch_dtype,
            }
            f.write(f"{json.dumps(metadata | results)}\n")

    files = glob.glob(f"{BASE_DIR}/trained_models_b02/{out_dir_name}/*/*.pt")
    for file_path in files:
        os.remove(file_path)


if __name__ == "__main__":
    app()
