"""Build distillation-ready chat datasets from teacher traces.

Current defaults target snippet extraction, but prompt rendering and teacher
response selection stay configurable so same pipeline can later distill normal
answer generation as well.

Input JSONL can either:
  1. contain a full teacher response field, or
  2. contain task-specific fields used to synthesize that response.

Output JSONL format:
    {
        "task": "snippets",
        "prompt_messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "teacher_response": "...",
        "messages": [... prompt + assistant ...],
        "metadata": {"question_id": "...", "doc_pmid": "..."}
    }

Usage:
    python -m bioasq.snippets.prepare_distillation_data \
        --input data/training/snippet_extraction/gold_pairs_with_rationale.jsonl \
        --output-dir data/training/distillation/snippets
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer()


DEFAULT_SYSTEM_PROMPTS = {
    "snippets": (
        "You are a biomedical snippet extractor. Given a question and a PubMed "
        "abstract, extract the most relevant text spans that help answer the question. "
        "Output valid JSON with two fields:\n"
        '- "thinking": 1-2 sentences explaining why the snippets are relevant.\n'
        '- "snippets": a list of verbatim text spans copied exactly from the abstract.'
    )
}

DEFAULT_USER_TEMPLATES = {
    "snippets": """\
Question: {question_body}

Abstract:
{doc_text}"""
}

DEFAULT_METADATA_FIELDS = ("question_id", "doc_pmid", "question_type")

SNIPPET_ASSISTANT_TEMPLATE = '{{"thinking": {thinking}, "snippets": {snippets}}}'


def _resolve_override_text(inline_value: str, file_path: Path | None) -> str | None:
    """Return inline text or file contents if an override was provided."""
    if inline_value and file_path is not None:
        raise typer.BadParameter("Use either inline override or file override, not both.")
    if inline_value:
        return inline_value
    if file_path is not None:
        return file_path.read_text()
    return None


def _format_user_prompt(template: str, example: dict[str, Any]) -> str:
    """Render a user template against one example with a clear error on missing keys."""
    try:
        return template.format_map(example)
    except KeyError as exc:
        missing = exc.args[0]
        raise typer.BadParameter(f"User template needs missing field: {missing}") from exc


def _build_snippet_teacher_response(example: dict[str, Any]) -> str | None:
    """Build default teacher JSON for snippet distillation."""
    snippets = example.get("snippets")
    if not snippets:
        return None

    thinking = example.get("thinking") or (
        "These snippets contain information relevant to the question."
    )
    return SNIPPET_ASSISTANT_TEMPLATE.format(
        thinking=json.dumps(thinking, ensure_ascii=False),
        snippets=json.dumps(snippets, ensure_ascii=False),
    )


def _build_teacher_response(
    example: dict[str, Any],
    task: str,
    teacher_response_field: str,
) -> str | None:
    """Return teacher response from explicit field or task-specific builder."""
    if teacher_response_field:
        value = example.get(teacher_response_field)
        if value in (None, ""):
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    if task == "snippets":
        return _build_snippet_teacher_response(example)

    raise typer.BadParameter(
        "Unknown task without --teacher-response-field. Provide teacher text explicitly."
    )


def _prepare_example(
    example: dict[str, Any],
    task: str,
    system_prompt: str,
    user_template: str,
    teacher_response_field: str,
    metadata_fields: tuple[str, ...],
    truncate_field: str,
    max_field_chars: int,
) -> dict[str, Any] | None:
    """Convert one raw example into distillation format."""
    working = dict(example)
    value = working.get(truncate_field)
    if isinstance(value, str) and len(value) > max_field_chars:
        working[truncate_field] = value[:max_field_chars]

    teacher_response = _build_teacher_response(working, task, teacher_response_field)
    if teacher_response is None:
        return None

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _format_user_prompt(user_template, working)},
    ]
    assistant_message = {"role": "assistant", "content": teacher_response}
    metadata = {field: working[field] for field in metadata_fields if field in working}

    return {
        "task": task,
        "prompt_messages": prompt_messages,
        "teacher_response": teacher_response,
        "messages": [*prompt_messages, assistant_message],
        "metadata": metadata,
    }


@app.command()
def main(
    input_path: Annotated[
        Path, typer.Option("--input", help="JSONL with teacher traces or raw task data")
    ] = Path("data/training/snippet_extraction/gold_pairs_with_rationale.jsonl"),
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for train/val distillation splits")
    ] = Path("data/training/distillation/snippets"),
    task: Annotated[
        str, typer.Option(help="Task preset. Use snippets now; override prompts later.")
    ] = "snippets",
    output_prefix: Annotated[
        str, typer.Option(help="Prefix for output JSONL files")
    ] = "distill",
    teacher_response_field: Annotated[
        str,
        typer.Option(
            help="Field containing full teacher response. If empty, task-specific builder is used."
        ),
    ] = "",
    system_prompt: Annotated[
        str, typer.Option(help="Inline system prompt override")
    ] = "",
    system_prompt_file: Annotated[
        Path | None, typer.Option(help="Path to a system prompt override")
    ] = None,
    user_template: Annotated[
        str, typer.Option(help="Inline user template override")
    ] = "",
    user_template_file: Annotated[
        Path | None, typer.Option(help="Path to a user template override")
    ] = None,
    metadata_fields: Annotated[
        str,
        typer.Option(help="Comma-separated fields to keep under metadata in output"),
    ] = ",".join(DEFAULT_METADATA_FIELDS),
    val_fraction: Annotated[
        float, typer.Option(help="Fraction of data used for validation")
    ] = 0.1,
    seed: Annotated[int, typer.Option(help="Random seed for splitting")] = 42,
    truncate_field: Annotated[
        str, typer.Option(help="Text field to truncate before prompt rendering")
    ] = "doc_text",
    max_field_chars: Annotated[
        int, typer.Option(help="Max chars kept from truncate_field")
    ] = 3500,
) -> None:
    """Create train/val JSONL files for student distillation."""
    if not 0.0 < val_fraction < 1.0:
        raise typer.BadParameter("--val-fraction must be between 0 and 1.")

    random.seed(seed)

    resolved_system_prompt = _resolve_override_text(system_prompt, system_prompt_file)
    resolved_user_template = _resolve_override_text(user_template, user_template_file)

    if resolved_system_prompt is None:
        if task not in DEFAULT_SYSTEM_PROMPTS:
            raise typer.BadParameter(
                f"No default system prompt for task '{task}'. Pass --system-prompt."
            )
        resolved_system_prompt = DEFAULT_SYSTEM_PROMPTS[task]

    if resolved_user_template is None:
        if task not in DEFAULT_USER_TEMPLATES:
            raise typer.BadParameter(
                f"No default user template for task '{task}'. Pass --user-template."
            )
        resolved_user_template = DEFAULT_USER_TEMPLATES[task]

    metadata_field_names = tuple(
        field.strip() for field in metadata_fields.split(",") if field.strip()
    )

    prepared: list[dict[str, Any]] = []
    skipped = 0
    with input_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            formatted = _prepare_example(
                example=example,
                task=task,
                system_prompt=resolved_system_prompt,
                user_template=resolved_user_template,
                teacher_response_field=teacher_response_field,
                metadata_fields=metadata_field_names,
                truncate_field=truncate_field,
                max_field_chars=max_field_chars,
            )
            if formatted is None:
                skipped += 1
                continue
            prepared.append(formatted)

    if len(prepared) < 2:
        raise typer.BadParameter("Need at least 2 usable examples to build train/val splits.")

    random.shuffle(prepared)
    val_size = min(max(1, int(len(prepared) * val_fraction)), len(prepared) - 1)
    val_records = prepared[:val_size]
    train_records = prepared[val_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in (("train", train_records), ("val", val_records)):
        out_path = output_dir / f"{output_prefix}_{split_name}.jsonl"
        with out_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} examples to {out_path}")

    print(f"Prepared {len(prepared)} examples ({skipped} skipped)")


if __name__ == "__main__":
    app()