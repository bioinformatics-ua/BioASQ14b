import click
import json


def load_content(baseline, docs_set):
    total = len(docs_set)
    doc_content = {}

    with open(baseline, "r") as f:
        for line in f:
            doc = json.loads(line)
            if doc["pmid"] in docs_set:
                doc_content[doc["pmid"]] = " ".join([doc["title"], doc["abstract"]])
                docs_set.remove(doc["pmid"])
                print(f"{total - len(docs_set)}/{total}", end="\r")

                if len(doc_content) == total:
                    break

    print(f"{baseline} done")
    return doc_content


@click.command()
@click.argument("test_set")
@click.argument("out_file")
def main(test_set, out_file):
    print("here")
    with open(test_set) as f:
        testset = json.load(f)

    doc_ids = set()

    for q in testset["questions"]:
        for d in q["documents"]:
            doc_ids.add(d.split("/")[-1])
    doc_dictionary = load_content("../../../data/pubmed_baseline_2025.jsonl", doc_ids)
    new_format = {}

    for q in testset["questions"]:
        abstracts = []
        for doc in q["documents"]:
            id = doc.split("/")[-1]
            abstracts.append(doc_dictionary[id])
        new_format[q["id"]] = {
            "question": q["body"],
            "abstracts": abstracts,
            "snippets": [snippet["text"] for snippet in q["snippets"]],
        }

    print(len(new_format.keys()))
    print("writing to :", out_file)
    with open(out_file, "w") as f:
        json.dump(new_format, f)


if __name__ == "__main__":
    main()
