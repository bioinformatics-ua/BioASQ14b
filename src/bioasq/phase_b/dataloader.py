from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson


class BioASQDataLoader:
    """
    Loads BioASQ JSON format (training14b.json or competition batch JSON).

    Both formats share the same structure:
        {"questions": [{"body", "documents", "snippets", "type", "id", ...}]}

    Documents are PubMed URLs — resolve them separately with lookup_abstract_B.py
    before running inference.

    Gold fields (ideal_answer, exact_answer) are present in training data,
    absent in competition test batches.

    Yields unified dicts:
        {
            "id":           str,
            "body":         str,
            "type":         "yesno" | "factoid" | "list" | "summary",
            "documents":    list[dict(id,text)],   # PubMed URLs
            "snippets":     list[str],   # snippet texts
            "ideal_answer": list[str] | None,
            "exact_answer": str | list | list[list] | None,
        }
    """

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path
        self._questions: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        return [
            {
                "id": q["id"],
                "body": q["body"],
                "type": q["type"],
                # Documents are raw PubMed URLs at this stage — resolve them
                # separately with lookup_abstract_B.py before running inference
                "documents": q.get("documents", []),
                "snippets": [s["text"] for s in q.get("snippets", [])],
                "ideal_answer": q.get("ideal_answer") or None,  # arr
                # exact_answer format differs by type:
                "exact_answer": q.get("exact_answer") or None,
            }
            for line in self.path.open("rb")
            if (q := orjson.loads(line).strip())
        ]

    def __iter__(self) -> Iterator[dict]:
        return iter(self._questions)

    def __len__(self) -> int:
        return len(self._questions)

    def __getitem__(self, idx: int) -> dict:
        return self._questions[idx]


if __name__ == "__main__":
    from typing import Annotated

    import typer

    app = typer.Typer()

    @app.command()
    def main(path: Annotated[Path, typer.Argument(..., help="Path to BioASQ JSON file")]) -> None:
        loader = BioASQDataLoader(path)

        print(f"Loaded {len(loader)} questions\n")
        for q in loader:
            print(f"[{q['type']}] {q['body']}")
            print(f"  id:           {q['id']}")
            print(f"  documents:    {len(q['documents'])} urls")
            print(f"  snippets:     {len(q['snippets'])}")
            print(f"  ideal_answer: {str(q['ideal_answer'])[:80] if q['ideal_answer'] else None}")
            print(f"  exact_answer: {q['exact_answer']}")
            print()

    app()
