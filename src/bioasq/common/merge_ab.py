"""Merge a Phase-A BioASQ file with a Quorum or LLM-as-a-judge output to produce an A+B submission."""

import sys
from pathlib import Path
from typing import Annotated

import orjson
import typer

app = typer.Typer(help="Merge Phase-A BioASQ file with Phase-B answer outputs")


def merge(phase_a: dict, quorum: dict, *, keep_meta: bool = False) -> dict:
    """Merge phase-A documents/snippets with quorum ideal_answer & exact_answer."""
    quorum_by_id = {q["id"]: q for q in quorum["questions"]}

    merged_questions = []
    for q in phase_a["questions"]:
        qid = q["id"]
        if qid not in quorum_by_id:
            print(
                f"WARNING: question {qid} not found in quorum output, skipping answers",
                file=sys.stderr,
            )
            merged_questions.append(q)
            continue

        qb = quorum_by_id[qid]
        merged = dict(q)
        if "ideal_answer" in qb:
            merged["ideal_answer"] = qb["ideal_answer"]
        if "exact_answer" in qb:
            merged["exact_answer"] = qb["exact_answer"]
        if keep_meta and "quorum_meta" in qb:
            merged["quorum_meta"] = qb["quorum_meta"]

        merged_questions.append(merged)

    return {"questions": merged_questions}


@app.command()
def main(
    phase_a: Annotated[
        Path, typer.Argument(..., help="Phase-A BioASQ JSON file (documents/snippets)")
    ],
    quorum: Annotated[
        Path, typer.Argument(..., help="Quorum output JSON file (ideal_answer/exact_answer)")
    ],
    output: Annotated[
        Path | None, typer.Option(..., "-o", "--output", help="Output path (default: stdout)")
    ] = None,
    keep_meta: Annotated[
        bool, typer.Option(..., "--keep-meta", help="Include quorum_meta in output")
    ] = False,
) -> None:
    phase_a_data = orjson.loads(phase_a.read_bytes())
    quorum_data = orjson.loads(quorum.read_bytes())

    result = merge(phase_a_data, quorum_data, keep_meta=keep_meta)

    output_json = orjson.dumps(result, option=orjson.OPT_INDENT_2)
    if output:
        output.write_bytes(output_json)
        print(f"Wrote {len(result['questions'])} questions to {output}", file=sys.stderr)
    else:
        sys.stdout.buffer.write(output_json)
        sys.stdout.buffer.write(b"\n")


def _merge_from_judge(phase_a: dict, judge: dict) -> dict:
    """Merge phase-A with an LLM-as-a-judge dict (keyed by question ID)."""
    merged_questions = []
    for q in phase_a["questions"]:
        qid = q["id"]
        if qid not in judge:
            print(
                f"WARNING: question {qid} not found in judge output, skipping answers",
                file=sys.stderr,
            )
            merged_questions.append(q)
            continue

        jq = judge[qid]
        merged = dict(q)
        if "ideal_answer" in jq:
            merged["ideal_answer"] = jq["ideal_answer"]
        if "exact_answer" in jq:
            merged["exact_answer"] = jq["exact_answer"]

        merged_questions.append(merged)

    return {"questions": merged_questions}


@app.command("from-judge")
def from_judge(
    phase_a: Annotated[
        Path, typer.Argument(..., help="Phase-A BioASQ JSON file (documents/snippets)")
    ],
    judge_dir: Annotated[
        Path, typer.Argument(..., help="Directory containing LLM-as-a-judge JSON files")
    ],
) -> None:
    """Merge a Phase-A BioASQ file with all LLM-as-a-judge files in a directory."""
    phase_a_data = orjson.loads(phase_a.read_bytes())
    out_dir = judge_dir

    judge_files = sorted(judge_dir.glob("*.json"))
    if not judge_files:
        print(f"No JSON files found in {judge_dir}", file=sys.stderr)
        raise typer.Exit(1)

    for judge_file in judge_files:
        judge_data = orjson.loads(judge_file.read_bytes())
        result = _merge_from_judge(phase_a_data, judge_data)
        out_path = out_dir / f"{judge_file.stem}.bioasq.json"
        out_path.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
        print(
            f"Wrote {len(result['questions'])} questions to {out_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    app()
