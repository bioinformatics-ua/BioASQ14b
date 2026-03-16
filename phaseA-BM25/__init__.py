from typing import TypedDict


class Doc(TypedDict):
    id: str
    text: str


class NegDoc(Doc):
    score: float


class QuestionWithNegatives(TypedDict):
    id: str
    body: str
    pos_docs: list[Doc]
    neg_docs: list[NegDoc]
