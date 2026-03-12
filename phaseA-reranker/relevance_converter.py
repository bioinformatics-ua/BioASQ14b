import click
import json
from ranx import Run
import os


@click.command()
@click.argument("run_paths", nargs=-1)
def main(run_paths):
    print(run_paths)
    for run_path in run_paths:
        print(run_path)
        run = Run.from_file(run_path).to_dict()
        new_run_dict = {}
        for q_id, docs_dict in run.items():
            new_run_dict[q_id] = {}

            for doc_id, doc_score in docs_dict.items():
                if doc_score <= 0.01:
                    continue
                new_run_dict[q_id][doc_id] = doc_score

        outdir = os.path.dirname(run_path)
        outfile_name, _ = os.path.splitext(os.path.basename(run_path))

        Run(new_run_dict).save(os.path.join(outdir, f"{outfile_name}_relevance.json"))


if __name__ == "__main__":
    main()
