import re
from pathlib import Path
from typing import Annotated, Any

import orjson
import typer

app = typer.Typer()


def clean_text(value: str) -> str:
    cleaned_value = re.sub(r"\n(?=\w)", ", ", value)
    cleaned_value = re.sub(r"^[^a-zA-Z]+", "", cleaned_value)
    cleaned_value = re.sub(r"\s+", " ", cleaned_value)
    cleaned_value = re.sub(r"\n", " ", cleaned_value)
    cleaned_value = re.sub(r"Long answer: .*?(?=\n|$)", "", cleaned_value)
    return re.sub(r"Longer answer:(.*)", "", cleaned_value)


def default_exact_answer(qtype: str) -> str:
    match qtype:
        case "yesno":
            return "no"
        case "factoid" | "list":
            return []
        case _:
            return ""


@app.command()
def main(
    test_set: Annotated[Path, typer.Argument(..., help="Path to test set JSONL file")],
    ideal_answer: Annotated[Path, typer.Argument(..., help="Path to ideal answer JSON file")],
    out_file: Annotated[Path, typer.Argument(..., help="Path to output JSON file")],
    fallback_ideal_answer: Annotated[
        Path | None,
        typer.Option(
            default=None,
            help="Path to fallback ideal answer JSON file, which is used when primary is invalid",
        ),
    ] = None,
    exact_answer: Annotated[
        Path | None,
        typer.Option(
            default=None,
            help="Path to exact answer JSON file (per-question, with valid flag)",
        ),
    ] = None,
) -> None:
    testset = orjson.loads(test_set.read_bytes())
    ideal_predictions = orjson.loads(ideal_answer.read_bytes())
    fallback_predictions = orjson.loads(fallback_ideal_answer.read_bytes() or b"{}")
    exact_predictions = orjson.loads(exact_answer.read_bytes() or b"{}")

    new_data: dict[str, list[dict[str, Any]]] = {"questions": []}

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

    with out_file.open("wb") as f:
        f.write(orjson.dumps(new_data))


if __name__ == "__main__":
    main()
