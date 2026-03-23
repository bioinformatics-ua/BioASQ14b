"""BioASQ data loading for Phase B.

Loads BioASQ JSON format (training or competition batch JSON) and yields
unified dicts with resolved document and snippet information.

Refactored from ``phaseB/loaders/dataloader.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from bioasq.common.types import Question, QuestionType, Snippet


class BioASQDataLoader:
    """Loads BioASQ JSONL or JSON data files.

    Supports both JSONL (one question per line) and standard BioASQ JSON
    (``{"questions": [...]}`` format) formats.

    Yields :class:`~bioasq.common.types.Question` instances.
    """

    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        self._questions: list[Question] = self._load()

    def _load(self) -> list[Question]:
        """Load questions from file, handling both JSON and JSONL formats."""
        raw_text: str = self.path.read_text()

        # Try standard JSON format first
        raw_questions: list[dict[str, str | list[str] | list[dict[str, str]]]]
        if raw_text.strip().startswith("{"):
            data: dict[str, list[dict[str, str | list[str] | list[dict[str, str]]]]] = json.loads(raw_text)
            raw_questions = data.get("questions", [])
        else:
            # JSONL format
            raw_questions = [
                json.loads(line)
                for line in raw_text.splitlines()
                if line.strip()
            ]

        questions: list[Question] = []
        for q in raw_questions:
            q_id: str = str(q["id"])
            q_body: str = str(q["body"])
            q_type_str: str = str(q.get("type", "summary"))

            # Parse question type
            try:
                q_type: QuestionType = QuestionType(q_type_str)
            except ValueError:
                q_type = QuestionType.SUMMARY

            # Parse documents
            raw_docs: list[str] = []
            docs_field = q.get("documents", [])
            if isinstance(docs_field, list):
                for d in docs_field:
                    raw_docs.append(str(d))

            # Parse snippets
            snippets: list[Snippet] = []
            snippets_field = q.get("snippets", [])
            if isinstance(snippets_field, list):
                for s in snippets_field:
                    if isinstance(s, dict):
                        snippets.append(Snippet(
                            text=str(s.get("text", "")),
                            document=str(s.get("document", "")),
                            offset_in_begin_section=str(s.get("offsetInBeginSection", "")),
                            offset_in_end_section=str(s.get("offsetInEndSection", "")),
                            begin_section=str(s.get("beginSection", "")),
                            end_section=str(s.get("endSection", "")),
                        ))
                    else:
                        snippets.append(Snippet(text=str(s)))

            # Parse answers
            ideal_answer: list[str] | str | None = None
            raw_ideal = q.get("ideal_answer")
            if isinstance(raw_ideal, list):
                ideal_answer = [str(a) for a in raw_ideal]
            elif raw_ideal is not None:
                ideal_answer = str(raw_ideal)

            exact_answer: list[str] | list[list[str]] | str | None = None
            raw_exact = q.get("exact_answer")
            if isinstance(raw_exact, str):
                exact_answer = raw_exact
            elif isinstance(raw_exact, list):
                if raw_exact and isinstance(raw_exact[0], list):
                    exact_answer = [[str(s) for s in group] for group in raw_exact]
                else:
                    exact_answer = [str(e) for e in raw_exact]

            questions.append(Question(
                id=q_id,
                body=q_body,
                type=q_type,
                documents=raw_docs,
                snippets=snippets,
                ideal_answer=ideal_answer,
                exact_answer=exact_answer,
            ))

        return questions

    def __iter__(self) -> Iterator[Question]:
        return iter(self._questions)

    def __len__(self) -> int:
        return len(self._questions)

    def __getitem__(self, idx: int) -> Question:
        return self._questions[idx]
