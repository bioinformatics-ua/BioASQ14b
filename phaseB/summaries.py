import json
import time
import re
import click
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

MODEL_NAMES = {
    "OpenBioLLM": "/data/lmdeploy/Llama3-OpenBioLLM-70B-AWQ-INT4-TurboMind",
    "Nemotron": "/data/lmdeploy/Nemotron",
}


PROMPT_TEMPLATES = {
    1: """Act as a biomedical expert. You will receive several answers to a question. Given these answers you should construct a short and concise full answer (50-150 words, at least one sentence). \n\nFirst read and understand the relevant inforamtion present in the several answers, extracting all relevant facts.\n\nAfter thinking about the information presented, present a final JSON containing your concise answer {{"answer": answer}}. Please show all your reasoning first.\n\nAnswer in around 50 - 150 words, without using any markdown.\n\nQuestion: {question}\n\n{answers}""",
    2: """Act as a biomedical expert. You will receive multiple answers to a given question. Your task is to analyze these responses, extract all relevant information, and synthesize a concise yet comprehensive final answer (50-150 words, at least one complete sentence).

1. Carefully read and understand the key facts and insights from the provided answers.  
2. Thoughtfully evaluate the information to form a well-reasoned conclusion.  
3. Present your reasoning step-by-step before delivering the final response.  

Finally, output a JSON object in the following format:  
{{"answer": "<your concise answer>"}}  

Guidelines:  
- Ensure the answer is informative, clear, and medically accurate.  
- Do not use Markdown formatting.  
- Keep the response within the word limit.  

Question: {question}  

Answers: {answers}""",
    3: """Act as a biomedical expert. You will receive multiple answers to a given question. Your task is to analyze these responses, extract all relevant information, and synthesize a concise yet comprehensive final answer (50-150 words, at least one complete sentence).

1. Carefully read and understand the key facts and insights from the provided answers.  
2. Thoughtfully evaluate the information to form a well-reasoned conclusion.  
3. Present your reasoning step-by-step before delivering the final response.  

Finally, output a JSON object in the following format:  
{{"answer": "<your concise answer>"}}  

Guidelines:  
- Ensure the answer is informative, clear, and medically accurate.  
- Do not use Markdown formatting.  
- Keep the response within the word limit.  

For example:

Question: "What is the use of P85-Ab?"

Insert your thinking here.

{{"answer": "P85-Ab is a promising novel biomarker for nasopharyngeal carcinoma screening."}}

Question: {question}  

Answers: {answers}""",
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
@click.argument("runs", nargs=-1)
@click.option("--model-name", help="Model name (e.g., OpenBioLLM, Nemotron).")
@click.option("--data-path", help="Path to input JSON data.")
@click.option("--output-dir", help="Directory to save outputs.")
@click.option("--prompt-ids", help="Comma-separated list of prompt types to use.")
@click.option("--out-id", help="file identifier")
def main(runs, model_name, data_path, output_dir, prompt_ids, out_id):
    model = MODEL_NAMES.get(model_name, None)
    if not model:
        raise ValueError(
            f"Invalid model name. Available models: {', '.join(MODEL_NAMES.keys())}"
        )

    backend_config = TurbomindEngineConfig(
        model_format="awq",
        cache_max_entry_count=0.5,
        quant_policy=4,
    )

    generation_config = GenerationConfig(
        max_new_tokens=1000,
        # Added in  B2 phase B
        temperature=0.5,
    )

    data = []
    for _file in runs:
        with open(_file) as f:
            data.append(json.load(f))

    answer_dict = {}

    with open(data_path) as f:
        source_data = json.load(f)

    selected_prompts = [int(pid) for pid in prompt_ids.split(",")]

    answer_dict = {pid: {} for pid in selected_prompts}
    prompts = []
    prompt_info = []
    id_list = list(data[0].keys())

    for _key in id_list:
        answers = ""
        for i in range(len(data)):
            answers += str(data[i][_key]["text"]) + "\n"

        question = source_data[_key]["question"]
        for pid in selected_prompts:
            if pid in PROMPT_TEMPLATES:
                prompts.append(
                    PROMPT_TEMPLATES[pid].format(answers=answers, question=question)
                )
                prompt_info.append((_key, pid))

    # prompts = prompts[:10]
    print(f"Number of promtps: {len(prompts)}")
    print("Loading model...")
    pipe = pipeline(model, backend_config=backend_config)
    print("Model loaded. Running inference...")

    t0 = time.time()
    response = pipe(prompts, gen_config=generation_config)
    response_time = time.time() - t0
    print(f"Response generated in {response_time:.2f} seconds")

    total_tokens = 0

    for i, (response_text, (qid, pid)) in enumerate(zip(response, prompt_info)):
        total_tokens += response_text.generate_token_len
        valid, parsed_text = parse_json(response_text.text)

        text = response_text.text
        if valid:
            text = parsed_text
        try_counter = 0
        while not valid:
            if try_counter > 1:
                print("reached try conuter")
                break

            print(f"invalid, regeneratin {pid} {qid} {i}", flush=True)

            p = (
                "Please use the answer I give you and correct the JSON error. The model did not contain a valid json please generate a valid json, which must contain only {'answer': answer}.\n\n Previous Answer: "
                + text
            )
            new_r = pipe([p], gen_config=generation_config)
            valid, parsed_text = parse_json(new_r[0].text)
            try_counter += 1

            if valid:
                text = parsed_text

        answer_dict[pid][qid] = {"text": text, "valid": valid}

    print(
        f"Total tokens: {total_tokens}, Tokens per sec: {total_tokens / response_time:.2f}"
    )
    print(
        f"Total questions: {len(prompts)}, Avg answer length: {total_tokens / len(prompts):.2f}, qpd: {86400 * (len(prompts) / response_time)}"
    )

    for pid in answer_dict.keys():
        with open(
            f"{output_dir}/{out_id}_{model_name}_summary_{len(runs)}_{pid}.json", "w"
        ) as file:
            json.dump(answer_dict[pid], file)

    print(f"Results saved in {output_dir}")
    print(f"Model used: {model_name}")


if __name__ == "__main__":
    main()
