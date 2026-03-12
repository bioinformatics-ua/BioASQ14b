import click
import json
from tqdm import tqdm


@click.command()
@click.argument("run_path")
@click.argument("og_path")
@click.argument("out_file")
def main(run_path, og_path, out_file):

    doc_ids_needed = set()

    print("here")
    with open(run_path) as f:
        run = json.load(f)
        for q_data in run["questions"]:
            for doc_id in map(lambda x: x.split("/")[-1], q_data["documents"]):
                doc_ids_needed.add(doc_id)

    old_docs = {}
    with open(og_path) as f:
        for line in f.readlines():
            tmp = json.loads(line)
            for cand in tmp["bm25"]:
                old_docs[cand["id"]] = cand["text"]

    doc_ids_needed = doc_ids_needed - set(old_docs.keys())

    if len(doc_ids_needed) > 0:
        with open("/data/bioasq13/data/pubmed_baseline_2025.jsonl") as f:
            for article in tqdm(map(json.loads, f), total=36555429):
                if article["pmid"] in doc_ids_needed:
                    old_docs[article["pmid"]] = (
                        article["title"] + " " + article["abstract"]
                    )

    new_format = {}

    for q in run["questions"]:
        abstracts = []
        for doc in q["documents"]:
            id = doc.split("/")[-1]
            abstracts.append(old_docs[id])
        new_format[q["id"]] = {"question": q["body"], "abstracts": abstracts}

    print(len(new_format.keys()))
    print("writing to :", out_file)
    with open(out_file, "w") as f:
        json.dump(new_format, f)


if __name__ == "__main__":
    main()
