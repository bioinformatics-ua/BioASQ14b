from typing import Any


class BasicSamplePreprocessing:
    def __init__(self, tokenizer, model_max_length=-1) -> None:
        self.tokenizer = tokenizer
        self.model_max_length = (
            model_max_length if model_max_length != -1 else tokenizer.model_max_length
        )

    def __call__(self, sample) -> Any:

        if "label" in sample:
            # training
            inputs = self.tokenizer(sample["query_text"], sample["doc_text"])

            # skip sample if too big during training
            if len(inputs.input_ids) > self.model_max_length:
                return None

            return inputs | {"labels": int(sample["label"])}
        else:
            # inference
            inputs = self.tokenizer(
                sample["query_text"],
                sample["doc_text"],
                truncation=True,
                max_length=self.model_max_length,
            )

            return inputs | {"id": sample["id"], "doc_id": sample["doc_id"]}


from typing import Any
from sample_preprocessing import BasicSamplePreprocessing
import re
import math


def split_sentence(sentence, n_parts):
    words = sentence.split()
    part_length = len(words) // n_parts  # Number of words in each part
    remainder = len(words) % n_parts  # Remainder to distribute one extra word as needed

    parts = []
    current_index = 0

    for i in range(n_parts):
        # Calculate the end index for this part, adding an extra word if needed
        extra_word = 1 if i < remainder else 0
        next_index = current_index + part_length + extra_word

        # Join words to form a part and add it to the result list
        part = " ".join(words[current_index:next_index])
        parts.append(part)

        # Update current index to the start of the next part
        current_index = next_index

    return parts


import time
import nltk


class SentencePreprocessing(BasicSamplePreprocessing):
    def __init__(self, tokenizer, model_max_length=-1, sentence_length=128) -> None:
        super().__init__(tokenizer=tokenizer, model_max_length=model_max_length)
        self.sentence_length = sentence_length

    def __call__(self, sample) -> Any:
        # st = time.time()
        _sentences = nltk.sent_tokenize(sample["doc_text"])
        # print("ntlk", time.time()-st)
        # st = time.time()
        _tokenized_sentences = self.tokenizer(
            _sentences, add_special_tokens=False
        ).input_ids
        tokenized_question = self.tokenizer(
            sample["query_text"], add_special_tokens=False
        ).input_ids

        sentences = []
        tokenized_sentences = []

        for i, tok_sent in enumerate(_tokenized_sentences):
            if len(tok_sent) + len(tokenized_question) + 3 <= self.sentence_length:
                sentences.append(_sentences[i])
                tokenized_sentences.append(tok_sent)
            else:
                number_chuncks = (
                    round(
                        len(tok_sent)
                        + len(tokenized_question)
                        + 3 / self.sentence_length
                    )
                    + 1
                )
                for sent in split_sentence(_sentences[i], number_chuncks):
                    sentences.append(sent)
                    tokenized_sentences.append(
                        self.tokenizer(sent, add_special_tokens=False).input_ids
                    )

        # sentences = _sentences
        # tokenized_sentences = _tokenized_sentences

        paragraphs = []
        paragraph = (
            self.tokenizer.cls_token + sample["query_text"] + self.tokenizer.sep_token
        )
        num_tokens = len(tokenized_question) + 2
        for i in range(len(sentences)):
            sen = sentences[i]
            # print(num_tokens, paragraph)
            # print(len(tokenized_sentences[i]), len(self.tokenizer(paragraph, add_special_tokens=False).input_ids))
            if len(tokenized_sentences[i]) + num_tokens + 1 > self.sentence_length:
                # print("finish")
                paragraphs.append(paragraph + self.tokenizer.sep_token)
                paragraph = (
                    self.tokenizer.cls_token
                    + sample["query_text"]
                    + self.tokenizer.sep_token
                    + sen
                )
                num_tokens = len(tokenized_question) + len(tokenized_sentences[i]) + 2
            else:
                # print("concat")
                paragraph += " " + sen
                num_tokens += len(tokenized_sentences[i])

        paragraphs.append(paragraph)

        inputs = self.tokenizer(paragraphs, add_special_tokens=False)
        # inputs = {}
        # print("all",time.time()-st)
        if "label" in sample:
            return inputs | {"labels": int(sample["label"])}
        else:
            return inputs | {"id": sample["id"], "doc_id": sample["doc_id"]}


DEFAULT_CAUSAL_LM_PROMPT = "question:{query} \n \n passage:{doc}"


class CausalLMSamplePreprocessing:
    """Sample preprocessing for decoder-only reranker models (e.g. Nemotron, LLaMA-based).

    Formats query and document into a single prompt string instead of a BERT-style
    pair encoding. No token_type_ids are produced.
    """

    def __init__(
        self,
        tokenizer,
        model_max_length: int = 512,
        prompt_template: str = DEFAULT_CAUSAL_LM_PROMPT,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.prompt_template = prompt_template

    def _format(self, sample) -> str:
        return self.prompt_template.format(
            query=sample["query_text"], doc=sample["doc_text"]
        )

    def __call__(self, sample) -> Any:
        text = self._format(sample)

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.model_max_length,
            add_special_tokens=True,
        )

        if "label" in sample:
            return inputs | {"labels": float(sample["label"])}
        else:
            return inputs | {"id": sample["id"], "doc_id": sample["doc_id"]}


# ── Qwen3-Reranker ────────────────────────────────────────────────────────────

QWEN3_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
    "\n<|im_start|>user\n"
)
QWEN3_RERANKER_SUFFIX = "\n<|im_start|>assistant\n\n\n\n\n"
QWEN3_RERANKER_DEFAULT_INSTRUCTION = "Given a biomedical question, retrieve relevant PubMed passages that answer the query"


class Qwen3RerankerSamplePreprocessing:
    """Sample preprocessing for Qwen3-Reranker generative rerankers.

    Formats the query + document into the yes/no judge prompt used by
    Qwen3-Reranker models. The system prefix and assistant suffix are
    prepended/appended so the model's final token logits reflect
    relevance via the "yes"/"no" vocabulary entries.
    """

    def __init__(
        self,
        tokenizer,
        model_max_length: int = 8192,
        instruction: str = QWEN3_RERANKER_DEFAULT_INSTRUCTION,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.instruction = instruction
        self.prefix_tokens = tokenizer.encode(
            QWEN3_RERANKER_PREFIX, add_special_tokens=False
        )
        self.suffix_tokens = tokenizer.encode(
            QWEN3_RERANKER_SUFFIX, add_special_tokens=False
        )

    def _format_body(self, sample) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {sample['query_text']}\n"
            f"<Document>: {sample['doc_text']}"
        )

    def __call__(self, sample) -> Any:
        body = self._format_body(sample)
        max_body_len = (
            self.model_max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        body_ids = self.tokenizer.encode(
            body,
            truncation=True,
            max_length=max_body_len,
            add_special_tokens=False,
        )
        input_ids = self.prefix_tokens + body_ids + self.suffix_tokens
        inputs = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if "label" in sample:
            return inputs | {"labels": float(sample["label"])}
        else:
            return inputs | {"id": sample["id"], "doc_id": sample["doc_id"]}
