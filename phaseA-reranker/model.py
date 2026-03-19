from typing import Literal
import msgspec

type PMID = str


class PMIDLine(msgspec.Struct):
    pmid: PMID
    title: str
    abstract: str


class LookupEntry(msgspec.Struct):
    """One (pmid, score) pair; decodes from JSON array [pmid, score] via dec_hook."""

    pmid: PMID
    score: float


type Lookup = dict[PMID, list[LookupEntry]]


def lookup_dec_hook(type: type, obj: object) -> object:
    """Convert JSON array [pmid, score] to LookupEntry."""
    if type is LookupEntry and isinstance(obj, (list, tuple)) and len(obj) == 2:
        return LookupEntry(pmid=obj[0], score=obj[1])
    raise NotImplementedError(f"Cannot decode {obj!r} as {type}")


class _BaseQuestion(msgspec.Struct):
    id: PMID
    body: str
    type: Literal["yesno", "factoid", "list", "summary"]


class TestQuestion(_BaseQuestion):
    pass

class TrainingQuestion(_BaseQuestion):
    documents: list[str]
    snippets: list[str]
    ideal_answer: list[str]
    exact_answer: list[str]


type Question = TestQuestion | TrainingQuestion


class _Collection[T: Question](msgspec.Struct):
    questions: list[T]


class TrainingCollection(_Collection[TrainingQuestion]):
    pass


class TestCollection(_Collection[TestQuestion]):
    pass


type Collection = TrainingCollection | TestCollection

