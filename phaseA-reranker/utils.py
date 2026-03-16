import os
from transformers.trainer_utils import set_seed
from transformers import TrainingArguments
import yaml
import json


def setup_wandb(project, name):
    # print(os.getcwd())
    os.environ["WANDB_NAME"] = name

    os.environ["WANDB_API_KEY"] = open(".api").read().strip()
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_ENTITY"] = "bitua"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    # print("If you want to use wandb please change the in setup_wandb function on utils file. And update the bert_config_yaml to report to wandb")


def get_negative_positive_index_from_dataset(dataset):
    _sample = dataset[next(iter(dataset.keys()))]
    relevance_order = [k for k in _sample.keys() if isinstance(k, int)]
    # print(relevance_order)
    return min(relevance_order), max(relevance_order)


def get_relevance_order_from_dataset(dataset):
    _sample = dataset[next(iter(dataset.keys()))]
    relevance_order = sorted(
        [k for k in _sample.keys() if isinstance(k, int)], reverse=True
    )
    # print(relevance_order)
    return relevance_order


def _load_flat_config(path):
    assert path is not None, "`path` cannot be none"
    with open(path) as fp:
        config = yaml.safe_load(fp)

    return _flatten(config)


def _flatten(d):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(_flatten(v).items())
            elif isinstance(v, list):
                for x in v:
                    items.extend(_flatten(x).items())
            else:
                try:
                    items.append((k, eval(v)))
                except (NameError, TypeError):
                    items.append((k, v))
    else:
        raise ValueError(
            f"Found leaf value ({repr(d)}) that is not a dictionary. Please convert it to a dictionary."
        )
    return dict(items)


def create_config(base_config_path="bert_trainer_config.yaml", **update_config):
    # assert isinstance(update_config, dict), "update_config config must be a dictionary."
    print(
        f"Combining values supplied as `keywords arguments` with base config from {base_config_path}"
        if update_config
        else f"Using base config from {base_config_path}"
    )

    base_config = _load_flat_config(base_config_path)
    joint_config = base_config | update_config
    return TrainingArguments(**joint_config)


class EmptyEncodeBatch:
    def __init__(self):
        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []


def split_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def load_rank_data(bm25_rank_path, at=1000, qrels=None):

    dataset = {}
    with open(bm25_rank_path) as f:
        for line in f:
            q_data = json.loads(line)
            if qrels:
                if q_data["id"] not in qrels:
                    continue
            dataset[q_data["id"]] = {
                "documents": q_data["documents"][:at],
                "question": q_data["question"],
            }
    return dataset



import torch
from transformers.modeling_outputs import SequenceClassifierOutput

class MaxPPoolingReranker(torch.nn.Module):
    """
    Wraps a base HuggingFace Sequence Classification model.
    Takes flattened chunks, runs them through the model, and max-pools 
    the logits back to the original batch size based on sentences_count.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # The Hugging Face Trainer needs access to the config
        self.config = base_model.config 

    def forward(self, input_ids, attention_mask, sentences_count, token_type_ids=None, **kwargs):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        # 1. Run all chunks through the base model
        outputs = self.base_model(**inputs)
        chunk_logits = outputs.logits  # Shape: [total_chunks_in_batch, num_labels]

        # 2. Pool the logits back to the original batch size
        pooled_logits = []
        start_idx = 0
        for count in sentences_count:
            end_idx = start_idx + count
            doc_chunk_logits = chunk_logits[start_idx:end_idx]

            # MaxP: Take the maximum logit score across all chunks for this document
            # Handle edge case: if count is 0, doc_chunk_logits is empty
            if doc_chunk_logits.numel() == 0:
                # Use zeros as default logit for empty documents
                num_labels = chunk_logits.shape[1] if chunk_logits.dim() > 1 else 1
                doc_pooled_logit = torch.zeros(num_labels, device=chunk_logits.device, dtype=chunk_logits.dtype)
            else:
                doc_pooled_logit, _ = torch.max(doc_chunk_logits, dim=0)
            pooled_logits.append(doc_pooled_logit)

            start_idx = end_idx

        # Stack back into a tensor of shape [batch_size, num_labels]
        pooled_logits = torch.stack(pooled_logits)

        # Return a standard HF output object so your Trainer doesn't crash
        return SequenceClassifierOutput(logits=pooled_logits)
