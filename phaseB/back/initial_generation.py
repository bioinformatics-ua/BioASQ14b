import json
import time
import re
import click
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

MODEL_NAMES = {
    "OpenBioLLM": "/data/lmdeploy/Llama3-OpenBioLLM-70B-AWQ-INT4-TurboMind",
    "Nemotron": "/data/lmdeploy/Nemotron",
}


# prompt = f"""Context: {context}\nQuestion:{tmp['question']}\n\nAnswer in less than 150 words:"""
# prompt=f"""Act as a biomedical expert. You will receive several abstracts ('[abstract: Abstract]') summarizing research findings and methodologies. Along with this, a question will be provided('[question]').\nYour role is to analyze the abstract and provide a scientifically accurate, concise answer to the question, leveraging the information from the abstracts.\n\n[Abstract: {context}]\n\n[Question: {tmp['question']}]\n\nAnswer in less than 150 words:"""

PROMPT_TEMPLATES = {
    1: """Context: {context}\nQuestion: {question}\n\nAnswer in less than 150 words, present a final json containing your answer {{"answer": answer}}""",
    2: """Act as a biomedical expert. You will receive several {d_type} summarizing research findings and methodologies. Along with this, a question will be provided('[question]').\nYour role is to analyze the {d_type} and provide a scientifically accurate, concise answer to the question, leveraging the information from the {d_type}.\n\nAnswer in less than 150 words, present a final json containing your answer {{"answer": answer}}.\n\nQuestion: {question} \n\n {context}""",
    3: """Act as a biomedical expert. You will receive several {d_type} summarizing research findings and methodologies. Along with this, a question will be provided. Your role is to analyze the {d_type} and provide a scientifically accurate, concise answer to the question, leveraging the information from the {d_type}.\n\nFirst read and understand the relevant inforamtion present in the several {d_type}, extracting all relevant facts.\n\nAfter thinking about the information presented, and present a final json containing your answer {{"answer": answer}}. Please show all your reasoning first.\n\nAnswer in around 50 - 150 words, without using any markdown.\n\nQuestion: {question}\n\n{context}""",
    4: """Act as a biomedical expert. You will receive several {d_type} summarizing research findings and methodologies. Along with this, a question will be provided. Your role is to analyze the {d_type} and provide a scientifically accurate, concise answer to the question, leveraging the information from the {d_type}.\n\nFirst read and understand the relevant inforamtion present in the several {d_type}, extracting all relevant facts, explaining all your reasoning.\n\nAfter thinking about the information presented, present a final json containing your answer {{"answer": answer}}.\n\nAnswer in around 50 - 150 words, use a concise format without only with plain text (no lists or markdown).\n\nQuestion: {question}\n\n{context}""",
    5: """Act as a biomedical expert. You will receive several {d_type} summarizing research findings and methodologies. Along with this, a question will be provided. Your role is to analyze the {d_type} and provide a scientifically accurate, concise answer to the question, leveraging the information from the {d_type}.\n\nFirst read and understand the relevant inforamtion present in the several {d_type}, extracting all relevant facts, explaining all your reasoning.\n\nAfter thinking about the information presented, present a final json containing your answer {{"answer": answer}}.\n\nAnswer in around 50 - 150 words, use a concise format without only with plain text (no lists or markdown).\n\n For example:\n\nQuestion: "What is the use of P85-Ab?" \n\nInsert your thinking here.\n\n{{"answer": "P85-Ab is a promising novel biomarker for nasopharyngeal carcinoma screening."}}\n\nQuestion: {question}\n\n{context}""",
}


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


@click.command()
@click.option("--model-name", help="Model name (e.g., OpenBioLLM, Nemotron).")
@click.option("--data-type", help="abstract/snippets")
@click.option("--data-path", help="Path to input JSON data.")
@click.option(
    "--num-abstracts",
    help="Comma-separated list of numbers of abstracts to process per question.",
)
@click.option("--output-dir", help="Directory to save outputs.")
@click.option("--prompt-ids", help="Comma-separated list of prompt types to use.")
def main(model_name, data_type, data_path, num_abstracts, output_dir, prompt_ids):
    model = MODEL_NAMES.get(model_name, None)
    if not model:
        raise ValueError(
            f"Invalid model name. Available models: {', '.join(MODEL_NAMES.keys())}"
        )

    backend_config = TurbomindEngineConfig(
        model_format="awq",
        cache_max_entry_count=0.6,
        quant_policy=4,
    )

    generation_config = GenerationConfig(
        max_new_tokens=1000,
        # Added in  B2 phase B
        temperature=0.5,
    )

    with open(data_path) as f:
        data = json.load(f)

    selected_prompts = [int(pid) for pid in prompt_ids.split(",")]
    selected_abstract_counts = [int(n) for n in num_abstracts.split(",")]

    answer_dict = {
        num_abstract: {pid: {} for pid in selected_prompts}
        for num_abstract in selected_abstract_counts
    }
    prompts = []
    id_list = []
    prompt_info = []

    for _id, tmp in data.items():
        for num_abstract in selected_abstract_counts:
            _data = tmp[data_type][:num_abstract]
            context = "\n\n".join(f"{data_type}: {x}" for x in _data)
            question = tmp["question"]
            for pid in selected_prompts:
                if pid in PROMPT_TEMPLATES:
                    prompts.append(
                        PROMPT_TEMPLATES[pid].format(
                            d_type=data_type, context=context, question=question
                        )
                    )
                    prompt_info.append((_id, num_abstract, pid))

    # prompts = prompts[:10]
    print(f"Number of promtps: {len(prompts)}", flush=True)
    print("Loading model...", flush=True)
    pipe = pipeline(model, backend_config=backend_config)
    print("Model loaded. Running inference...", flush=True)

    t0 = time.time()
    response = pipe(prompts, gen_config=generation_config)
    response_time = time.time() - t0
    print(f"Response generated in {response_time:.2f} seconds", flush=True)

    total_tokens = 0

    for i, (response_text, (qid, num_abstract, pid)) in enumerate(
        zip(response, prompt_info)
    ):
        total_tokens += response_text.generate_token_len
        valid, parsed_text = parse_json(response_text.text)

        # here we trau
        text = response_text.text
        if valid:
            text = parsed_text

        answer_dict[num_abstract][pid][qid] = {"text": text, "valid": valid}

    print(
        f"Total tokens: {total_tokens}, Tokens per sec: {total_tokens / response_time:.2f}",
        flush=True,
    )
    print(
        f"Total questions: {len(prompts)}, Avg answer length: {total_tokens / len(prompts):.2f}, qpd: {86400 * (len(prompts) / response_time)}",
        flush=True,
    )

    for num_abstract in answer_dict:
        for key in answer_dict[num_abstract]:
            with open(
                f"{output_dir}/{model_name}_{data_type}_{num_abstract}_{key}.json", "w"
            ) as file:
                json.dump(answer_dict[num_abstract][key], file)

    print(f"Results saved in {output_dir}", flush=True)
    print(f"Model used: {model_name}", flush=True)


if __name__ == "__main__":
    main()
