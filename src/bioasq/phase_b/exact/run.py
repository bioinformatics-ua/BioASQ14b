"""
Exact-answer inference for BioASQ Phase B.

For yesno / factoid / list: extracts exact_answer from model JSON.

For summary: no exact answer — model outputs reasoning + ideal_answer (same narrative
role as ideal_answer elsewhere). Stored as ideal_answer plus exact_answer \"\".

Usage:
    python inference/run_exact.py \\
        --input  data/training14b/training14b.json \\
        --output dev/outputs/exact_p1.json \\
        --model  /path/to/model
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import orjson
import typer

from bioasq.phase_b.backends.cloud import OpenRouterBackend
from bioasq.phase_b.backends.local import VLLMBackend
from bioasq.phase_b.dataloader import BioASQDataLoader

if TYPE_CHECKING:
    from bioasq.phase_b.types import Prompts

app = typer.Typer()


def build_context(
    question: dict[str, Any], num_context: int, source: Literal["abstracts", "snippets"]
) -> str:
    if source == "abstracts":
        items = [d["text"] for d in question["documents"] if d.get("text")][:num_context]
        label = "Abstract"
    else:
        items = question["snippets"][:num_context]
        label = "Snippet"
    if not items:
        return "(No context available)"
    return "\n\n".join(f"{label} {i}: {s}" for i, s in enumerate(items, 1))


def parse_exact(
    text: str, qtype: Literal["yesno", "factoid", "list", "summary"]
) -> tuple[bool, str | list[str] | None]:
    """
    Find the last JSON block in the model output and extract exact_answer.
    Promotes bare strings / flat lists to the expected shape per type.
    Returns (valid, exact_answer).
    """
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return False, None
    try:
        exact = orjson.loads(matches[-1]).get("exact_answer")
        if exact is None:
            return False, None
        if qtype == "yesno" and isinstance(exact, str) and exact.lower() in ("yes", "no"):
            return True, exact.lower()
        if qtype == "factoid":
            return True, [exact] if isinstance(exact, str) else exact
        if qtype == "list":
            if isinstance(exact, list) and exact and not isinstance(exact[0], list):
                return True, [[item] for item in exact]  # promote flat → nested
            return True, exact
    except orjson.JSONDecodeError:
        pass
    return False, None


def parse_summary_ideal(text: str) -> tuple[bool, str | None]:
    """Extract ideal_answer from JSON (summary questions). Returns (valid, ideal_paragraph)."""
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return False, None
    try:
        obj = orjson.loads(matches[-1])
        ideal = obj.get("ideal_answer") if isinstance(obj, dict) else None
        if isinstance(ideal, str) and ideal.strip():
            return True, ideal.strip()
    except orjson.JSONDecodeError:
        pass
    return False, None


@app.command()
def main(
    input_: Annotated[Path, typer.Argument(..., help="Input file path")],
    output: Annotated[
        Path | None, typer.Option(default=None, help="Output file path (single prompt mode)")
    ],
    output_dir: Annotated[
        Path | None, typer.Option(default=None, help="Output directory (multi-prompt grid mode)")
    ],
    model: Annotated[str | None, typer.Option(default=None, help="Model name")],
    prompt_ids: Annotated[
        list[str],
        typer.Option(
            help="One or more prompt IDs to run with one model load, e.g. 1 2 3 4."
            "Use 'all' to run every prompt in the prompts file.",
        ),
    ],
    backend: Annotated[
        Literal["local", "openrouter"],
        typer.Option(default="local", choices=["local", "openrouter"]),
    ],
    num_snippets: Annotated[
        int, typer.Option(default=5, help="Number of snippets to use for context")
    ],
    prompts_file: Annotated[
        Path,
        typer.Option(
            default=str(Path(__file__).parent / "prompts_exact.json"), help="Prompts file path"
        ),
    ],
    max_tokens: Annotated[
        int, typer.Option(default=500, help="Maximum number of tokens to generate")
    ],
    temperature: Annotated[float, typer.Option(default=0.0, help="Temperature for generation")],
    types: Annotated[
        list[Literal["yesno", "factoid", "list", "summary"]],
        typer.Option(
            default=["yesno", "factoid", "list"],
            help="Question types to run. summary → ideal_answer paragraph JSON.",
        ),
    ],
    context_source: Annotated[
        Literal["abstracts", "snippets"],
        typer.Option(
            default="abstracts",
            help="abstracts = Phase A+ golden docs; snippets = Phase B passages.",
        ),
    ],
    tensor_parallel_size: Annotated[int, typer.Option(default=1, help="Tensor parallel size")],
    gpu_memory_utilization: Annotated[
        float, typer.Option(default=0.85, help="GPU memory utilization")
    ],
    max_model_len: Annotated[int, typer.Option(default=8192, help="Maximum model length")],
    enforce_eager: Annotated[bool, typer.Option(default=False, help="Enforce eager mode")],
    request_delay: Annotated[
        float,
        typer.Option(
            default=0.0,
            help="Seconds between requests — "
            "use ~4.0 for free OpenRouter models (16 req/min limit).",
        ),
    ],
) -> None:
    loader = BioASQDataLoader(path=input_)
    prompts: Prompts = orjson.loads(prompts_file.read_text())

    if prompt_ids == ["all"]:
        prompt_ids = sorted(
            {qtype_prompts.keys() for qtype_prompts in prompts.values()},
            key=lambda x: int(x) if x.isdigit() else x,
        )

    questions = [q for q in loader if q["type"] in set(types)]
    model_slug = (model or "unknown").replace("/", "-").replace(".", "-").replace(":", "-")

    # In grid mode: compute output path per (pid, qtype) and skip existing files
    def grid_output_path(pid: str, qtype: str) -> Path:
        return Path(output_dir) / f"{model_slug}_p{pid}_{context_source}_{qtype}.json"

    # Determine which (pid, qtype) combos still need work
    combos_to_run: list[tuple[str, str]] = []
    for pid in prompt_ids:
        qtypes_needed = {q["type"] for q in questions if pid in prompts.get(q["type"], {})}
        for qtype in sorted(qtypes_needed):
            if grid_output_path(pid, qtype).exists():
                print(f"Skipping prompt {pid} / {qtype} — output already exists")
                continue
            combos_to_run.append((pid, qtype))

    if not combos_to_run:
        print("All outputs already exist — nothing to run.")
        return

    # Non-grid mode: single output file — skip if it already exists
    if output and output.exists():
        print(f"Output already exists: {output} — skipping.")
        return

    print(f"{len(combos_to_run)} combo(s) to run. Loading model...")

    backend = (
        VLLMBackend(
            model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )
        if backend == "local"
        else OpenRouterBackend(
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            request_delay=request_delay,
        )
    )

    backend.load()

    # Process each (pid, qtype) combo separately — save immediately after each
    saved_files: list[str] = []
    for pid, qtype in combos_to_run:
        combo_prompts, combo_qids = [], []
        for q in questions:
            if q["type"] != qtype:
                continue
            if pid not in prompts.get(qtype, {}):
                continue
            template = prompts[qtype][pid]["template"]
            combo_prompts.append(
                template.format(
                    question=q["body"],
                    context=build_context(q, num_snippets, context_source),
                )
            )
            combo_qids.append(q["id"])

        if not combo_prompts:
            continue

        print(f"\nPrompt {pid} / {qtype} — {len(combo_prompts)} questions...")
        responses = backend.generate_batch(combo_prompts)

        results = {}
        for raw, qid in zip(responses, combo_qids, strict=True):
            if qtype == "summary":
                valid, ideal = parse_summary_ideal(raw)
                results[qid] = {
                    "exact_answer": "",
                    "ideal_answer": ideal if valid else "",
                    "valid": valid,
                    "raw": raw,
                }
            else:
                valid, exact = parse_exact(raw, qtype)
                results[qid] = {"exact_answer": exact, "valid": valid, "raw": raw}

        n_valid = sum(1 for r in results.values() if r["valid"])
        print(f"Prompt {pid} / {qtype} — {n_valid}/{len(results)} valid")

        out = grid_output_path(pid, qtype)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            f.write(orjson.dumps(results))
        print(f"Saved to {out}")
        saved_files.append(str(out))

    backend.unload()
    print(f"Inference done. {len(saved_files)} file(s) saved.")


if __name__ == "__main__":
    main()
