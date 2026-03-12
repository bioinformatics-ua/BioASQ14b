import argparse
import json
import os

os.environ["TMPDIR"] = "/data/lmdeploy/tmp"
os.environ["HF_HOME"] = "/data/lmdeploy/hf_cache"

os.environ["TRITON_CACHE_DIR"] = "/data/lmdeploy/tmp"
import re
from tqdm import tqdm
from datasets import Dataset
from unsloth import FastModel, FastLanguageModel
from unsloth.chat_templates import get_chat_template
import gc
import torch
import triton
# print("Using Triton cache dir:", triton.runtime.config.cache_dir)


def parse_json(text):
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if matches:
        last_json_str = matches[-1]
        try:
            parsed_json = json.loads(last_json_str, strict=False)
            if "answer" in parsed_json:
                return True, parsed_json["answer"]
        except json.JSONDecodeError:
            pass
    return False, text


# ------------------------------------------------
# Parse arguments
# ------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run inference on all checkpoints in a model directory."
)
parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Path to directory containing checkpoints.",
)
parser.add_argument(
    "--output_template",
    type=str,
    required=True,
    help="Base name for output files (e.g., tmp → tmp_E1.json).",
)
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Base name for output files (e.g., tmp → tmp_E1.json).",
)

args = parser.parse_args()

# ------------------------------------------------
# Load constants and dataset
# ------------------------------------------------

print("Loading data...")
with open(args.input_file, "r") as file:
    raw_data = json.load(file)

# Prompt template
prompt = """Act as a biomedical expert. You will receive several {d_type} summarizing research findings and methodologies. Along with this, a question will be provided. Your role is to analyze the {d_type} and provide a scientifically accurate, concise answer to the question, leveraging the information from the {d_type}.\n\nFirst read and understand the relevant inforamtion present in the several {d_type}, extracting all relevant facts, explaining all your reasoning.\n\nAfter thinking about the information presented, present a final json containing your answer {{"answer": answer}}.\n\nAnswer in around 50 - 150 words, use a concise format without only with plain text (no lists or markdown).\n\n For example:\n\nQuestion: "What is the use of P85-Ab?" \n\nInsert your thinking here.\n\n{{"answer": "P85-Ab is a promising novel biomarker for nasopharyngeal carcinoma screening."}}\n\nQuestion: {question}\n\n{context}"""


# Prepare inference samples
def test_gen():
    for k, i in raw_data.items():
        data_type = "abstracts"
        _data = i[data_type][:5]
        context = "\n\n".join(f"{data_type}: {x}" for x in _data)
        question = i["question"]
        user_text = prompt.format(d_type=data_type, context=context, question=question)
        yield {
            "id": k,
            "conversations": [
                {"content": user_text, "role": "user"},
            ],
        }


# ------------------------------------------------
# Inference function
# ------------------------------------------------
def run_inference(model_name, output_path):
    print(f"\n\nRunning inference with model: {model_name}")
    # base_model_path = "/data/lmdeploy/gemma-base"

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-27b-it",  # base model
        max_seq_length=2048,
        full_finetuning=False,
    )

    # Step 2: Load the LoRA adapter
    model.load_adapter(model_name)

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    FastLanguageModel.for_inference(model)

    dataset = Dataset.from_generator(test_gen)
    dataset = dataset.map(
        lambda x: {
            "id": x["id"],
            "text": tokenizer.apply_chat_template(
                x["conversations"], add_generation_prompt=True
            ),
        }
    )

    predictions = {}

    for example in tqdm(dataset):
        inputs = tokenizer(example["text"], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)

        prompt_len = inputs["input_ids"].shape[1]
        assistant_output = tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()
        valid, parsed_text = parse_json(assistant_output)

        predictions[str(example["id"])] = {"text": parsed_text, "valid": valid}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as outfile:
        json.dump(predictions, outfile, indent=2)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------
# Main: Run inference on all checkpoints
# ------------------------------------------------
all_checkpoints = sorted(
    [
        os.path.join(args.model_dir, d)
        for d in os.listdir(args.model_dir)
        if os.path.isdir(os.path.join(args.model_dir, d)) and "checkpoint" in d
    ],
    key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)),
)

print(f"Found {len(all_checkpoints)} checkpoints in {args.model_dir}")

for idx, checkpoint in enumerate(all_checkpoints, 1):
    output_file = f"bioasq_b04/{args.output_template}_E{idx}.json"
    run_inference(checkpoint, output_file)
