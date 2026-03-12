import json

""" 

The aim of this script is to distinguish between the content of the training files from last year and this year. The contents of last year's files are already sanitized and can be found in the './training11b' folder. The new content has been cleaned up and expanded and has been merged with last year's content.

"""


def read_prev_training_file(previous_training_file):
    """read inflated file from previous editions"""

    with open(previous_training_file, "r") as prev_file:
        old_questions = [question for question in map(json.loads, prev_file)]

    return old_questions


def read_new_training_file(current_training_file, old_questions):
    """read new questions and add new keys"""
    old_questions_ids = [question["id"] for question in old_questions]

    with open(current_training_file, "r") as curr_file:
        data = json.load(curr_file)
        new_questions = [
            question
            for question in data["questions"]
            if question["id"] not in old_questions_ids
        ]

        for question in new_questions:
            question["edition"] = "13B"
            question["baseline"] = "2025"

    print(f"duplicate questions = {len(data['questions']) - len(new_questions)}")
    print(f"new questions = {len(new_questions)}")
    return new_questions


def join_and_write(filename, old_questions, new_questions):
    """write the inflated file ('edition' and 'baselines' keys added)"""

    joined_questions = old_questions + new_questions
    print(f"joined questions = {len(joined_questions)}")

    with open(filename, "w") as file:
        for question in joined_questions:
            json.dump(question, file)
            file.write("\n")

    return joined_questions


def clean_questions(questions):
    """clean the inflated file, in other words, simplify the doc_id"""

    for question in questions:
        question["documents"] = [doc.split("/")[-1] for doc in question["documents"]]

    return questions


def add_content(questions):
    """add the documents content to training file"""

    def get_file_path(baseline):
        return "../baselines/pubmed_baseline_" + str(baseline) + ".jsonl"

    def load_content(baseline):
        with open(get_file_path(baseline), "r") as f:
            for doc in map(json.loads, f):
                yield doc

    set_question = set()
    for question in questions:
        doc_ids = question["documents"]
        set_question.update(doc_ids)

    map_doc = {}
    for doc in load_content("2025"):  # should be 2024
        if doc["pmid"] in set_question:
            map_doc[doc["pmid"]] = " ".join([doc["title"], doc["abstract"]])
            set_question.remove(doc["pmid"])

        if len(set_question) == 0:
            break

    for question in questions:
        question["documents"] = [
            {"id": id, "text": text}
            for id, text in map_doc.items()
            if id in question["documents"]
        ]

    return questions


def main():
    """main funtion"""

    # 1) inflated: add new key-value pairs (baseline and edition) and join this new questions to the previous questions
    prev_questions = read_prev_training_file("training13b/training13b_inflated.jsonl")
    new_questions = read_new_training_file("training14b.json", prev_questions)

    join_and_write("training14b_inflated.jsonl", prev_questions, new_questions)

    # 2) clean: clean the document list for each new question and join this new questions to the previous questions
    cleaned_prev_questions = read_prev_training_file(
        "training13b/training13b_inflated_clean.jsonl"
    )
    cleaned_new_questions = clean_questions(new_questions)

    join_and_write(
        "training14b_inflated_clean.jsonl",
        cleaned_prev_questions,
        cleaned_new_questions,
    )

    # 3) wContent: replace the list of documents with their content and join this new questions to the previous questions
    # cleaned_prev_questions_with_content = read_prev_training_file("training12b/training12b_inflated_wContents_IA_complete.jsonl")

    cleaned_prev_questions_with_content = read_prev_training_file(
        "training13b/training13b_inflated_clean_wContents.jsonl"
    )
    cleaned_new_questions_with_content = add_content(cleaned_new_questions)

    join_and_write(
        "training14b_inflated_clean_wContents.jsonl",
        cleaned_prev_questions_with_content,
        cleaned_new_questions_with_content,
    )


if __name__ == "__main__":
    main()
