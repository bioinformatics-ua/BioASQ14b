import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
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
    BioASQPairwiseIterator,
)
from sampler import (
    BasicSampler,
    ExponentialWeightSampler,
    BasicV2Sampler,
)
from collator import PairwiseSentenceCollator, RankingCollator, RankingCollatorForCasualLM, PairwiseCollator, RankingSentenceCollator

from utils import create_config, MaxPPoolingReranker
from collections import defaultdict
import torch
from sample_preprocessing import (
    BasicSamplePreprocessing,
    SentencePreprocessing2,
    CausalLMSamplePreprocessing,
    Qwen3RerankerSamplePreprocessing,
)

from trainer_callbacks import ResampleByRerankerCallback
import glob
import os
from ranx import Qrels, Run
from ranx import evaluate
import typer
from typing import Optional

app = typer.Typer()

BASE_DIR = Path(__file__).parent.resolve()

_LLM_FAMILY_KEYWORDS = (
    "llama",
    "mistral",
    "gemma",
    "falcon",
    "mpt",
    "phi",
    "gpt",
    "nemotron",
    "qwen",
)

def _is_causal_lm(config: AutoConfig) -> bool:
    architectures = getattr(config, "architectures", []) or []
    arch_str = " ".join(architectures).lower()
    if any(kw in arch_str for kw in _LLM_FAMILY_KEYWORDS):
        return True
    model_type = getattr(config, "model_type", "").lower()
    return any(model_type.startswith(kw) for kw in _LLM_FAMILY_KEYWORDS)

_GENERATIVE_RERANKER_NAMES = ("qwen3-reranker",)

def _is_generative_reranker(model_name: str) -> bool:
    return any(kw in model_name.lower() for kw in _GENERATIVE_RERANKER_NAMES)

def _parse_torch_dtype(dtype_str: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, None)

def _dtype_load_kwargs(dtype, causal: bool) -> dict:
    if dtype is None:
        return {}
    if causal:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}

def _short_model_name(model_name: str) -> str:
    if "://" in model_name or "/" not in model_name:
        return model_name
    return Path(model_name).name

class RerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        inputs.pop("id", None)          # remove non-model keys
        model_inputs = inputs.pop("inputs")  # unwrap the collator's nesting
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1), labels.float().view(-1))
        return (loss, outputs) if return_outputs else loss


class PairwiseRerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_inputs = inputs["pos_inputs"]
        neg_inputs = inputs["neg_inputs"]

        pos_outputs = model(**pos_inputs)
        neg_outputs = model(**neg_inputs)

        pos_logits = pos_outputs.logits
        neg_logits = neg_outputs.logits

        if pos_logits.shape[-1] == 1:
            pos_scores = pos_logits.view(-1)
            neg_scores = neg_logits.view(-1)
        else:
            pos_scores = pos_logits[:, 1]
            neg_scores = neg_logits[:, 1]

        loss_fct = torch.nn.MarginRankingLoss(margin=1.0)
        target = torch.ones_like(pos_scores)
        loss = loss_fct(pos_scores, neg_scores, target)

        return (loss, pos_outputs) if return_outputs else loss

class Qwen3PointwiseRerankerTrainer(Trainer):
    def __init__(self, *args, token_true_id: int, token_false_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_true_id = token_true_id
        self.token_false_id = token_false_id

    def _log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        last = logits[:, -1, :]
        pair = torch.stack(
            [last[:, self.token_false_id], last[:, self.token_true_id]], dim=1
        )
        return torch.nn.functional.log_softmax(pair, dim=1)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        log_probs = self._log_probs(outputs.logits)
        loss = -(labels * log_probs[:, 1] + (1.0 - labels) * log_probs[:, 0]).mean()
        return (loss, outputs) if return_outputs else loss

class Qwen3PairwiseRerankerTrainer(Trainer):
    def __init__(self, *args, token_true_id: int, token_false_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_true_id = token_true_id
        self.token_false_id = token_false_id

    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        last = logits[:, -1, :]
        pair = torch.stack(
            [last[:, self.token_false_id], last[:, self.token_true_id]], dim=1
        )
        return torch.nn.functional.log_softmax(pair, dim=1)[:, 1]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_inputs = inputs["pos_inputs"]
        neg_inputs = inputs["neg_inputs"]
        pos_outputs = model(**pos_inputs)
        neg_outputs = model(**neg_inputs)
        pos_scores = self._score(pos_outputs.logits)
        neg_scores = self._score(neg_outputs.logits)
        loss_fct = torch.nn.MarginRankingLoss(margin=1.0)
        target = torch.ones_like(pos_scores)
        loss = loss_fct(pos_scores, neg_scores, target)
        return (loss, pos_outputs) if return_outputs else loss


@app.command()
def main(
    model_name: str,
    val: str,
    data: str,
    batch: int = 32,
    gradient_accumulation_steps: int = 2,
    epoch: int = 5,
    sampler: str = "basic",
    sample_preprocessing: str = "basic",
    results_file: str = "val_results_b02.jsonl",
    seed: int = 42,
    num_neg_samples: int = 4,
    use_expanded_pos: bool = False,
    warmup_ratio: bool = True,
    callback: bool = False,
    torch_dtype: str = "bfloat16",
    max_length: int = 512,
    prompt_template: Optional[str] = None,
    pairwise: bool = True,
    qwen3_instruction: Optional[str] = None,
    eval_only: bool = False,
    eval_batch_size: int = 512,
    base_model: Optional[str] = None,
):
    if data == "quality":
        if use_expanded_pos:
            positives = "../data/quality/training14b_inflated_clean_wContents_dense_expanded.jsonl"
        else:
            positives = "../data/quality/training14b_inflated_clean_wContents.jsonl"
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

    # Use base_model for config/tokenizer if provided (e.g., when loading from checkpoint without config.json)
    config_source = base_model if base_model else model_name
    model_config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    causal = _is_causal_lm(model_config)
    generative_reranker = _is_generative_reranker(config_source)
    print(
        f"Model type detected: {'LLM / decoder-style' if causal else 'encoder (BERT-style)'}"
    )
    if generative_reranker:
        print("Generative reranker detected (Qwen3-style yes/no scoring)")

    tokenizer = AutoTokenizer.from_pretrained(config_source, trust_remote_code=True)
    tokenizer.model_max_length = max_length

    if causal:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    if sample_preprocessing == "basic":
        if generative_reranker:
            _instruction = qwen3_instruction or None
            kwargs = {} if _instruction is None else {"instruction": _instruction}
            sample_preprocessing_obj = Qwen3RerankerSamplePreprocessing(
                tokenizer, model_max_length=max_length, **kwargs
            )
        elif causal:
            _template = prompt_template or None
            kwargs = {} if _template is None else {"prompt_template": _template}
            sample_preprocessing_obj = CausalLMSamplePreprocessing(
                tokenizer, model_max_length=max_length, **kwargs
            )
        else:
            sample_preprocessing_obj = BasicSamplePreprocessing(tokenizer)
    elif sample_preprocessing == "sentence":
        sample_preprocessing_obj = SentencePreprocessing2(
            tokenizer, 
            model_max_length=max_length,
            sentence_length=512 # You can expose this as a Typer arg later
        )
    else:
        raise RuntimeError(f"Unknown sample_preprocessing: {sample_preprocessing!r}")

    dtype = _parse_torch_dtype(torch_dtype)
    load_kwargs = {
        "trust_remote_code": True,
        "config": model_config,
        **_dtype_load_kwargs(dtype, causal),
    }

    ckpt_num_labels = getattr(model_config, "num_labels", 1)

    # FIX: Force num_labels to 1 for causal rerankers (like mxbai) to prevent weight dropping
    if causal and not generative_reranker:
        model_config.num_labels = 1
    else:
        model_config.num_labels = ckpt_num_labels

    if generative_reranker:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    elif causal:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # FIX: Force num_labels to 1 if we are running Pointwise to prevent BCE shape crashes
        if ckpt_num_labels == 1 or not pairwise:
            id2label = {0: "SCORE"}
            label2id = {"SCORE": 0}
            model_config.num_labels = 1
        else:
            id2label = {0: "IRRELEVANT", 1: "RELEVANT"}
            label2id = {"IRRELEVANT": 0, "RELEVANT": 1}
            model_config.num_labels = 2
        model_config.id2label = id2label
        model_config.label2id = label2id

        # For eval-only with sentence preprocessing, checkpoint has wrapped model weights
        # We need to handle the base_model. prefix in the checkpoint
        if eval_only and sample_preprocessing == "sentence":
            # Load base model from base_model path (not checkpoint)
            # Exclude torch_dtype from load_kwargs as we apply it after loading
            base_load_kwargs = {k: v for k, v in load_kwargs.items() if k != 'torch_dtype'}
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model if base_model else model_name, **base_load_kwargs
            )
            # Then load checkpoint weights with prefix stripping
            import safetensors.torch
            checkpoint_path = Path(model_name) / "model.safetensors"
            if checkpoint_path.exists():
                state_dict = safetensors.torch.load_file(str(checkpoint_path))
                # The checkpoint was saved from MaxPPoolingReranker which wraps the base model
                # Keys have format: base_model.<actual_model_key>
                # We need to strip the "base_model." prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("base_model."):
                        new_key = k[11:]  # Remove "base_model." prefix (11 chars)
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v

                # Load the state dict and check for mismatches
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded checkpoint weights from {checkpoint_path}")
                if missing_keys:
                    print(f"WARNING: Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"WARNING: Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
                if not missing_keys and not unexpected_keys:
                    print("All checkpoint weights loaded successfully!")
            else:
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            # Set torch dtype
            if torch_dtype == "bfloat16":
                model = model.to(torch.bfloat16)
            elif torch_dtype == "float16":
                model = model.to(torch.float16)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **load_kwargs
            )

    token_true_id = (
        tokenizer.convert_tokens_to_ids("yes") if generative_reranker else None
    )
    token_false_id = (
        tokenizer.convert_tokens_to_ids("no") if generative_reranker else None
    )

    if sample_preprocessing == "sentence":
        # 1. Wrap the base model
        model = MaxPPoolingReranker(model)
        
        # 2. Swap to the Sentence collators
        if pairwise:
            train_data_collator = PairwiseSentenceCollator(tokenizer=tokenizer) #
        else:
            # You will need to build a PointwiseSentenceCollator or use RankingSentenceCollator
            train_data_collator = RankingSentenceCollator(tokenizer=tokenizer) #
            
        if not causal:
            eval_collator = RankingSentenceCollator(tokenizer=tokenizer) #
    else:
        # Keep your standard collators
        if pairwise:
            train_data_collator = PairwiseCollator(tokenizer=tokenizer)
        else:
            train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if causal:
            eval_collator = RankingCollatorForCasualLM(tokenizer=tokenizer)
        else:
            eval_collator = RankingCollator(tokenizer=tokenizer)

    match sampler:
        case "basic":
            sampler_cls = BasicSampler
        case "basicv2":
            sampler_cls = BasicV2Sampler
        case "exponential":
            assert use_expanded_pos, (
                "exponential sampler only works with use_expanded_pos"
            )
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

    loss_mode = ("GR-" if generative_reranker else "") + (
        "Pairwise" if pairwise else "Pointwise"
    )
    _model_identifier = _short_model_name(model_name)
    out_dir_name = (
        f"{_model_identifier.replace('/', '-')}-{seed}-E{epoch}"
        f"-S{sampler_cls.__name__}-SP{sample_preprocessing_obj.__class__.__name__}"
        f"-{val}-{data}_data-CB{callback}-KN{num_neg_samples}"
        f"-GA{gradient_accumulation_steps}-ExPOS{use_expanded_pos}-warmup{warmup_ratio}"
        f"-{loss_mode}"
    )
    print(out_dir_name)

    extra_training_kwargs = {}
    if causal:
        if torch_dtype == "bfloat16":
            extra_training_kwargs["bf16"] = True
        elif torch_dtype == "float16":
            extra_training_kwargs["fp16"] = True
    if generative_reranker:
        extra_training_kwargs["gradient_checkpointing"] = True
        extra_training_kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": False
        }

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

    iterator_cls = BioASQPairwiseIterator if pairwise else BioASQPointwiseIterator
    train_ds, test_ds = create_bioASQ_datasets(
        positive_data_path=positives,
        all_data_path=all_path,
        test_sample_preprocessing=sample_preprocessing_obj,
        iterator=iterator_cls(
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

    _callbacks = (
        [
            ResampleByRerankerCallback(
                train_dataset=train_ds,
                tokenizer=tokenizer,
                start_epoch=0,
                interval=-1,
                num_high_confidence_to_remove=1,
            )
        ]
        if callback
        else None
    )
    _common_trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=train_data_collator,
        callbacks=_callbacks,
    )

    if generative_reranker:
        trainer_cls = (
            Qwen3PairwiseRerankerTrainer if pairwise else Qwen3PointwiseRerankerTrainer
        )
        trainer = trainer_cls(
            **_common_trainer_kwargs,
            token_true_id=token_true_id,
            token_false_id=token_false_id,
        )
    else:
        trainer_cls = PairwiseRerankerTrainer if pairwise else RerankerTrainer
        trainer = trainer_cls(**_common_trainer_kwargs)

    if eval_only:
        print(f"Running evaluation only from checkpoint: {model_name}")
    else:
        trainer.train()

    if val == "val" or eval_only:
        _eval_batch_size = 16 if (causal or generative_reranker) else eval_batch_size
        print(f"Eval batch size: {_eval_batch_size}")
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=_eval_batch_size, collate_fn=eval_collator
        )

        run_dict = defaultdict(dict)
        model = model.to("cuda")

        results = {}
        with torch.no_grad():
            for sample in tqdm(test_dl):
                logits = model(**sample["inputs"].to("cuda")).logits
                if generative_reranker:
                    last = logits[:, -1, :]
                    pair = torch.stack(
                        [last[:, token_false_id], last[:, token_true_id]], dim=1
                    )
                    doc_score = torch.nn.functional.softmax(pair, dim=1)[:, 1].cpu()
                elif logits.shape[-1] == 1:
                    doc_score = logits.squeeze(-1).cpu()
                else:
                    # FIX: Removed softmax here so it matches the MarginRankingLoss logic!
                    doc_score = logits[:, 1].cpu()

                for i in range(doc_score.shape[0]):
                    run_dict[sample["id"][i]][sample["doc_id"][i]] = doc_score[i].item()

        metrics = [
            "ndcg@5",
            "mrr",
            "recall@10",
            "recall@100",
            "recall@1000",
            "map@10",
            "map-bioasq@10",
        ]

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
                "generative_reranker": generative_reranker,
                "torch_dtype": torch_dtype,
                "pairwise": pairwise,
            }
            f.write(f"{json.dumps(metadata | results)}\n")

    files = glob.glob(f"{BASE_DIR}/trained_models_b02/{out_dir_name}/*/*.pt")
    for file_path in files:
        os.remove(file_path)

if __name__ == "__main__":
    app()