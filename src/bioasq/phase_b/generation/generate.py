"""CLI entrypoint for Phase B Answer Generation.

This module provides a Typer CLI that consolidates standard and
exact answer generation, supporting both local and cloud model backends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, cast

import msgspec
import typer
from rich import print as rprint

from bioasq.common.io import load_json, load_jsonl
from bioasq.phase_b.backends import get_backend
from bioasq.phase_b.generation.run import load_prompts, run_generation

app = typer.Typer(help="BioASQ Phase B Generation CLI")


def _resolve_prompts_file(prompts_file: Path | None, extract_exact: bool) -> Path:
    if prompts_file is not None:
        return prompts_file

    p_base = Path(__file__).resolve().parent.parent

    if extract_exact:
        return p_base / "prompts" / "prompts_exact.json"

    return p_base / "prompts" / "prompts_generic.json"


def _load_questions(data_path: Path) -> list[Any]:
    if str(data_path).endswith(".jsonl"):
        return cast("list[Any]", load_jsonl(data_path))

    obj = load_json(data_path)
    if isinstance(obj, dict):
        obj_dict = cast("dict[str, Any]", obj)
        return cast("list[Any]", obj_dict.get("questions", []))

    return cast("list[Any]", obj)


def _filter_questions(loader: list[Any], q_types: set[str]) -> list[dict[str, Any]]:
    return [q for q in loader if isinstance(q, dict) and q.get("type", "summary") in q_types]


def _resolve_prompt_ids(prompt_ids: str, prompts_templates: dict[str, Any]) -> list[str]:
    if prompt_ids.lower() != "all":
        return prompt_ids.split(",")

    p_ids = set()
    for key, val in prompts_templates.items():
        if isinstance(val, dict) and "template" not in val:
            p_ids.update(val.keys())
            continue
        p_ids.add(key)

    return sorted(p_ids, key=lambda x: int(x) if x.isdigit() else x)


def _save_outputs(
    answer_dict: dict[int, dict[str, dict[str, Any]]],
    output_dir: Path,
    model: str,
    input_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model).name.replace("/", "-").replace(".", "-").replace(":", "-")

    for n in answer_dict:
        for pid in answer_dict[n]:
            results = {qid: msgspec.to_builtins(ans) for qid, ans in answer_dict[n][pid].items()}
            if not results:
                continue

            out_path = output_dir / f"{model_name}_p{pid}_{input_type}_{n}.json"
            save_path = output_dir / out_path.name
            with save_path.open("w") as f:
                json.dump(results, f, indent=2)
            rprint(f"Saved: [green]{save_path}[/green]")


@app.command("run")
def generate_command(
    data_path: Annotated[Path, typer.Option(help="Path to BioASQ JSON or JSONL data.")],
    output_dir: Annotated[Path, typer.Option(help="Directory for output files.")],
    model: Annotated[str, typer.Option(help="Model path or name.")],
    backend: Annotated[str, typer.Option(help="'local' or 'openrouter'")] = "local",
    input_type: Annotated[
        str, typer.Option(help="'abstracts', 'snippets', or 'snippets_with_thinking'")
    ] = "abstracts",
    num_support: Annotated[str, typer.Option(help="Context counts, e.g. '3,5'")] = "5",
    prompt_ids: Annotated[str, typer.Option(help="Comma-separated IDs, e.g. '1,2'")] = "1",
    prompts_file: Annotated[Path | None, typer.Option(help="Path to prompts JSON")] = None,
    max_tokens: Annotated[int, typer.Option(help="Max new tokens")] = 1000,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.5,
    gpu_memory_utilization: Annotated[float, typer.Option(help="vLLM GPU RAM util")] = 0.90,
    tensor_parallel_size: Annotated[int, typer.Option(help="vLLM tensor parallel size")] = 1,
    max_model_len: Annotated[int, typer.Option(help="vLLM max sequence length")] = 8192,
    types: Annotated[
        str, typer.Option(help="Comma-separated question types")
    ] = "yesno,factoid,list,summary",
    extract_exact: Annotated[bool, typer.Option(help="Extract exact answers.")] = False,
    request_delay: Annotated[float, typer.Option(help="OpenRouter request delay")] = 0.0,
) -> None:
    """Run Phase B answer generation for a collection of questions."""
    resolved_prompts_file = _resolve_prompts_file(prompts_file, extract_exact)

    rprint(f"Loading data from [cyan]{data_path}[/cyan]...")
    loader = _load_questions(data_path)

    q_types = set(types.split(","))
    questions = _filter_questions(loader, q_types)
    rprint(f"Found [green]{len(questions)}[/green] questions matching types {q_types}.")

    selected_counts = [int(n) for n in num_support.split(",")]
    prompts_templates = load_prompts(resolved_prompts_file)
    selected_prompts = _resolve_prompt_ids(prompt_ids, prompts_templates)

    rprint(f"Using prompts format from {resolved_prompts_file}")

    rprint(f"Loading [bold]{backend}[/bold] backend with model [cyan]{model}[/cyan]...")
    model_backend = get_backend(
        model,
        max_tokens,
        temperature,
        gpu_memory_utilization,
        tensor_parallel_size,
        max_model_len,
        request_delay,
    )

    model_backend.load()

    answer_dict = run_generation(
        questions=questions,
        backend=model_backend,
        prompts_templates=prompts_templates,
        input_type=input_type,
        selected_counts=selected_counts,
        selected_prompts=selected_prompts,
        extract_exact=extract_exact,
    )

    model_backend.unload()
    _save_outputs(answer_dict, output_dir, model, input_type)


if __name__ == "__main__":
    app()
