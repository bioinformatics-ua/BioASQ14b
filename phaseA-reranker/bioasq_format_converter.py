import click
import json
from ranx import Run
import os


@click.command()
@click.argument("run_path")
@click.argument("testset")
def main(run_path, testset):
    print("RUN???")
    with open(testset) as f:
        bioasq_testset = {q_data["id"]: q_data for q_data in json.load(f)["questions"]}

    run = Run.from_file(run_path).to_dict()
    bioasq_struct = {"questions": []}

    for q_id, docs_dict in run.items():
        prev_score = 1
        doc_list = []
        for doc_id, doc_score in docs_dict.items():
            assert doc_score <= prev_score
            prev_score = doc_score

            doc_list.append(f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_id}")
            # http://www.ncbi.nlm.nih.gov/pubmed/
            # http://pubmed.ncbi.nlm.nih.gov/

        bioasq_struct["questions"].append(
            {
                "id": q_id,
                "type": bioasq_testset[q_id]["type"],
                "body": bioasq_testset[q_id]["body"],
                "documents": doc_list[:10],
                "snippets": [],
            }
        )

    outdir = f"{os.path.dirname(run_path)}/runs_bioasq_format"
    outfile_name = os.path.basename(run_path)
    path_to_w = os.path.join(outdir, outfile_name)
    print(path_to_w)
    print(run_path)
    with open(path_to_w, "w") as fOut:
        json.dump(bioasq_struct, fOut)


if __name__ == "__main__":
    main()
