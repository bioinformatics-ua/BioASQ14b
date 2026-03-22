import msgspec


class JudgeScores(msgspec.Struct):
    correctness: float
    faithfulness: float
    completeness: float
    overall: float
    rationale: str
