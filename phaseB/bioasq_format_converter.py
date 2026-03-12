import click
import json


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


@click.command()
@click.argument("test_set")
@click.argument("run")
@click.argument("out_file")
@click.argument("baseline_file")
def main(test_set, run, out_file, baseline_file):
    with open(test_set) as f:
        testset = json.load(f)

    # testset
    with open(run) as f:
        predictions = json.load(f)

    with open(baseline_file) as f:
        predictions_baselin = json.load(f)
    new_data = {"questions": []}

    for q in testset["questions"]:
        # print(predictions[q['id']['text']])
        text_without_asterix = str(predictions[q["id"]]["text"]).replace("*", "")
        if not predictions[q["id"]]["valid"]:
            text_without_asterix = str(predictions_baselin[q["id"]]["text"]).replace(
                "*", ""
            )

        exact_answer = ""
        if q["type"] == "yesno":
            exact_answer = "yes"
        elif q["type"] == "factoid" or q["type"] == "list":
            exact_answer = []

        # if len(text_without_asterix.split(" "))>200:

        #     print("splitting answer "+q['id'])

        #     new_data['questions'].append(q|{'ideal_answer':text_without_asterix[:940], "exact_answer":exact_answer})
        # else:
        #     new_data['questions'].append(q|{'ideal_answer':text_without_asterix, "exact_answer":exact_answer})

        new_data["questions"].append(
            q | {"ideal_answer": text_without_asterix, "exact_answer": exact_answer}
        )
        # [q['id']]

    with open(out_file, "w") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    main()
