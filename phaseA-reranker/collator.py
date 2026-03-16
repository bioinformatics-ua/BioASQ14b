class RankingCollator:
    def __init__(
        self,
        tokenizer,
        model_inputs=None,  # Auto-detect from batch if None
        padding=True,
        max_length=None,
    ):
        self.tokenizer = tokenizer
        self.model_inputs = model_inputs
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        batch = {key: [i[key] for i in batch] for key in batch[0]}

        # Auto-detect model inputs if not specified
        if self.model_inputs is None:
            standard_inputs = {"input_ids", "attention_mask", "token_type_ids"}
            present_inputs = set(batch.keys()) & standard_inputs
            # Must have at least input_ids and attention_mask
            model_inputs = present_inputs
        else:
            model_inputs = self.model_inputs

        reminder_keys = set(batch.keys()) - model_inputs
        return {
            "inputs": self.tokenizer.pad(
                {k: batch[k] for k in model_inputs},
                padding=self.padding,
                max_length=self.max_length,
                return_tensors="pt",
            )
        } | {k: batch[k] for k in reminder_keys}


class PairwiseCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch = {key: [i[key] for i in batch] for key in batch[0]}
        # print(batch.keys())
        return {
            "pos_inputs": self.tokenizer.pad(
                batch["pos_inputs"], padding=True, return_tensors="pt"
            ),
            "neg_inputs": self.tokenizer.pad(
                batch["neg_inputs"],
                padding=True,
                # max_length=512,
                return_tensors="pt",
            ),
        }


class RankingCollatorForCasualLM(RankingCollator):
    def __init__(self, tokenizer, model_inputs={"input_ids", "attention_mask"}):
        super().__init__(tokenizer, model_inputs=model_inputs)


class RankingCollatorForSeq2Seq(RankingCollator):
    def __init__(
        self,
        tokenizer,
        model_inputs={"input_ids", "attention_mask", "decoder_input_ids"},
    ):
        super().__init__(tokenizer, model_inputs=model_inputs)


class SentenceCollator:
    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        # Dynamically detect keys from the first sample (token_type_ids may not exist)
        first_sample = batch[0]
        standard_keys = {"input_ids", "attention_mask", "token_type_ids"}
        present_keys = [k for k in standard_keys if k in first_sample]

        expanded_batch = {k: [] for k in present_keys}

        sentences_count = []

        for i, sample in enumerate(batch):
            for k in expanded_batch.keys():
                expanded_batch[k].extend(sample[k])
            sentences_count.append(len(sample["input_ids"]))

        expanded_batch_padded = self.tokenizer.pad(
            expanded_batch,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # batch = {key: [i[key] for i in batch] for key in batch[0]}
        expanded_batch_padded["sentences_count"] = sentences_count

        return expanded_batch_padded


class PairwiseSentenceCollator(SentenceCollator):
    def __call__(self, batch):
        pos_batch = []
        neg_batch = []
        for sample in batch:
            pos_batch.append(sample["pos_inputs"])
            neg_batch.append(sample["neg_inputs"])

        return {
            "pos_inputs": super().__call__(pos_batch),
            "neg_inputs": super().__call__(neg_batch),
        }


class RankingSentenceCollator(SentenceCollator):
    def __init__(
        self,
        tokenizer,
        model_inputs_keys={"input_ids", "attention_mask", "token_type_ids"},
        padding=True,
        max_length=None,
    ):
        self.tokenizer = tokenizer
        self.model_inputs_keys = model_inputs_keys
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        # Filter model_inputs_keys to only include keys present in the batch
        # (token_type_ids may not be present for some tokenizers)
        first_sample = batch[0]
        actual_model_keys = {k for k in self.model_inputs_keys if k in first_sample}

        model_inputs = []
        reminder_inputs = {
            k: [] for k in first_sample.keys() if k not in actual_model_keys
        }

        for sample in batch:
            model_inputs.append({k: sample[k] for k in actual_model_keys})

            for k in reminder_inputs.keys():
                reminder_inputs[k].append(sample[k])

        return {"inputs": super().__call__(model_inputs)} | reminder_inputs
