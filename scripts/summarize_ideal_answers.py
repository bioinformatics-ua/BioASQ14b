#!/usr/bin/env python3
"""Summarize overly long ideal_answers in a BioASQ submission file.

For each question whose ideal_answer exceeds MAX_WORDS, this script calls
OpenRouter to produce a concise summary (≤200 words) using only the
ideal_answer text — no snippets, documents, or external context.

Usage
-----
    python scripts/summarize_ideal_answers.py \
        --input  data/batch02/generation/quorum/submission_1_quorum-v2.json \
        --output data/batch02/generation/quorum/submission_1_quorum-v2-short.json \
        --model  google/gemini-2.5-flash \
        --max-words 200

Requires
--------
    OPENROUTER_API_KEY environment variable to be set.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import typer
from openai import OpenAI
from tqdm import tqdm

MAX_WORDS_DEFAULT = 200
SYSTEM_PROMPT = (
    "You are a biomedical writing assistant. "
    "Your task is to condense the provided answer into a clear, accurate summary "
    "of at most {max_words} words. "
    "Use only the information present in the answer — do not add, infer, or omit "
    "key facts. Return only the summary text, with no preamble or explanation."
)

app = typer.Typer(add_completion=False)


def _word_count(text: str) -> int:
    return len(text.split())


def _summarize(
    client: OpenAI,
    ideal_answer: str,
    model: str,
    max_words: int,
    max_tokens: int,
    temperature: float,
    request_delay: float,
) -> str:
    """Call the LLM to summarize a single ideal_answer."""
    if request_delay > 0:
        time.sleep(request_delay)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(max_words=max_words),
            },
            {
                "role": "user",
                "content": (
                    f"Please summarize the following answer in at most {max_words} words:\n\n"
                    f"{ideal_answer}"
                ),
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Path to the input submission JSON file."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Path to write the updated submission JSON file."
    ),
    model: str = typer.Option(
        "google/gemini-2.5-flash",
        "--model",
        "-m",
        help="OpenRouter model identifier.",
    ),
    max_words: int = typer.Option(
        MAX_WORDS_DEFAULT,
        "--max-words",
        help="Ideal answers exceeding this word count will be summarized.",
    ),
    max_tokens: int = typer.Option(
        512,
        "--max-tokens",
        help="Maximum tokens for each LLM response.",
    ),
    temperature: float = typer.Option(
        0.3,
        "--temperature",
        help="Sampling temperature.",
    ),
    request_delay: float = typer.Option(
        0.5,
        "--request-delay",
        help="Seconds to wait between API requests (rate limiting).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print statistics without calling the LLM.",
    ),
) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not dry_run:
        typer.echo("ERROR: OPENROUTER_API_KEY environment variable is not set.", err=True)
        raise typer.Exit(1)

    with input_path.open() as f:
        data: dict = json.load(f)

    questions: list[dict] = data["questions"]
    to_summarize = [
        i
        for i, q in enumerate(questions)
        if _word_count(
            q["ideal_answer"] if isinstance(q["ideal_answer"], str) else q["ideal_answer"][0]
        )
        > max_words
    ]

    typer.echo(
        f"Found {len(to_summarize)}/{len(questions)} questions with ideal_answer > {max_words} words."
    )

    if dry_run:
        for i in to_summarize:
            q = questions[i]
            ia = q["ideal_answer"] if isinstance(q["ideal_answer"], str) else q["ideal_answer"][0]
            typer.echo(f"  [{i}] {_word_count(ia)} words — {q['id']}")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    summarized = 0
    failed = 0

    for idx in tqdm(to_summarize, desc="Summarizing", unit="answer"):
        q = questions[idx]
        is_list = isinstance(q["ideal_answer"], list)
        ia: str = q["ideal_answer"][0] if is_list else q["ideal_answer"]  # type: ignore[index]

        try:
            summary = _summarize(
                client=client,
                ideal_answer=ia,
                model=model,
                max_words=max_words,
                max_tokens=max_tokens,
                temperature=temperature,
                request_delay=request_delay,
            )
            q["ideal_answer"] = [summary] if is_list else summary
            summarized += 1
            wc_after = _word_count(summary)
            wc_before = _word_count(ia)
            tqdm.write(f"  [{idx}] {wc_before} → {wc_after} words  ({q['id']})")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            tqdm.write(f"  [{idx}] FAILED ({q['id']}): {exc}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    typer.echo(
        f"\nDone. Summarized {summarized} answers, {failed} failed. "
        f"Output written to {output_path}"
    )


if __name__ == "__main__":
    app()
