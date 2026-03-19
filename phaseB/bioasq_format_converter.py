import click
import json
import re


def clean_text(value):
    cleaned_value = re.sub(r"\n(?=\w)", ", ", value)
    cleaned_value = re.sub(r"^[^a-zA-Z]+", "", cleaned_value)
    cleaned_value = re.sub(r"\s+", " ", cleaned_value)
    cleaned_value = re.sub(r"\n", " ", cleaned_value)
    cleaned_value = re.sub(r"Long answer: .*?(?=\n|$)", "", cleaned_value)
    cleaned_value = re.sub(r"Longer answer:(.*)", "", cleaned_value)
    return cleaned_value


def default_exact_answer(qtype):
    if qtype == "yesno":
        return "no"
    elif qtype in ("factoid", "list"):
        return []
    return ""


@click.command()
@click.argument("test_set")
@click.argument("ideal_answer")
@click.argument("out_file")
@click.option("--fallback-ideal-answer", default=None, help="Fallback ideal answer file (used when primary is invalid)")
@click.option("--exact-answer", default=None, help="Exact answer file (per-question, with valid flag)")
def main(test_set, ideal_answer, out_file, fallback_ideal_answer, exact_answer):
    with open(test_set) as f:
        testset = json.load(f)

    with open(ideal_answer) as f:
        ideal_predictions = json.load(f)

    fallback_predictions = {}
    if fallback_ideal_answer:
        with open(fallback_ideal_answer) as f:
            fallback_predictions = json.load(f)

    exact_predictions = {}
    if exact_answer:
        with open(exact_answer) as f:
            exact_predictions = json.load(f)

    new_data = {"questions": []}

    for q in testset["questions"]:
        qid = q["id"]
        qtype = q["type"]

        # Ideal answer: use primary if valid, else fallback
        pred = ideal_predictions.get(qid, {})
        if pred.get("valid", False):
            text = str(pred["text"]).replace("*", "")
        elif qid in fallback_predictions:
            text = str(fallback_predictions[qid]["text"]).replace("*", "")
        else:
            text = str(pred.get("text", "")).replace("*", "")

        # Exact answer: use file if valid, else hardcoded default
        exact_pred = exact_predictions.get(qid, {})
        if exact_pred.get("valid", False):
            ea = exact_pred["exact_answer"]
        else:
            ea = default_exact_answer(qtype)

        new_data["questions"].append(q | {"ideal_answer": text, "exact_answer": ea})

    with open(out_file, "w") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    main()
