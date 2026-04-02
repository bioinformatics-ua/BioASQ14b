import os
from pathlib import Path
from typing import Annotated

import orjson
import typer
from ranx import Run

app = typer.Typer()


@app.command()
def main(
    run_path: Annotated[Path, typer.Argument()], testset: Annotated[Path, typer.Argument()]
) -> None:
    bioasq_testset = {
        q_data["id"]: q_data for q_data in orjson.loads(testset.read_bytes())["questions"]
    }

    run = Run.from_file(str(run_path)).to_dict()
    bioasq_struct = {"questions": []}

    for q_id, docs_dict in run.items():
        prev_score = 1
        doc_list = []
        for doc_id, doc_score in docs_dict.items():
            assert doc_score <= prev_score
            prev_score = doc_score
            doc_list.append(f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_id}")

        bioasq_struct["questions"].append(
            {
                "id": q_id,
                "type": bioasq_testset[q_id]["type"],
                "body": bioasq_testset[q_id]["body"],
                "documents": doc_list[:10],
                "snippets": [],
            }
        )

    outfile = run_path.with_suffix(".bioasq.json")
    outfile.write_bytes(orjson.dumps(bioasq_struct, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    app()
