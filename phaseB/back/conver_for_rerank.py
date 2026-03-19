import click
import json
from collections import defaultdict

import re


def clean_text(value):
    """
    Clean the values of a dictionary by removing trailing spaces or tabs,
    line breaks, replacing line breaks between items of a list with commas,
    and removing specific text patterns.

    Args:
        dictionary (dict): The input dictionary.

    Returns:
        dict: A new dictionary with cleaned values.
    """

    # Replace line breaks between list items with commas
    cleaned_value = re.sub(r"\n(?=\w)", ", ", value)
    # Remove specific text patterns like "Long answer: ..."

    cleaned_value = re.sub(r"^[^a-zA-Z]+", "", cleaned_value)

    # Remove trailing spaces or tabs
    cleaned_value = re.sub(r"\s+", " ", cleaned_value)
    # Remove line breaks
    cleaned_value = re.sub(r"\n", " ", cleaned_value)

    cleaned_value = re.sub(r"Long answer: .*?(?=\n|$)", "", cleaned_value)
    cleaned_value = re.sub(r"Longer answer:(.*)", "", cleaned_value)

    return cleaned_value


# prepare file for  bm25
@click.command()
@click.argument("files", nargs=-1, type=click.Path())
@click.option("--out")
@click.option("--testset")
def main(files, out, testset):
    print(files)
    data = []

    # files = ["outs/0_llama_1.json", "outs/1_llama_1.json","outs/2_llama_1.json","outs/3_llama_1.json",]

    number_of_removed_answers = 0
    answer = defaultdict(list)
    for file in files:
        with open(file, "r") as f:
            tmp = json.load(f)
            for k, v in tmp.items():
                if len(clean_text(v).split(" ")) <= 200:
                    answer[k].append({"id": file, "score": 1.0, "text": clean_text(v)})
                else:
                    number_of_removed_answers += 1
                    # print(clean_text(v))

    print(f"{number_of_removed_answers} answers were removed")

    # answer

    bm25_format = []
    # Batch01/BioASQ-task12bPhaseA-testset1
    with open(testset, "r") as f:
        data = json.load(f)
        for q in data["questions"]:
            if len(answer[q["id"]]) == 0:
                print("error no answeer")
            bm25_format.append(
                {"id": q["id"], "query_text": q["body"], "bm25": answer[q["id"]]}
            )

    with open(out, "w") as f:
        for l in bm25_format:
            f.write(json.dumps(l) + "\n")


if __name__ == "__main__":
    main()
