from typing import Literal, TypedDict


class Prompt(TypedDict):
    id: str
    template: str


type PromptsForType = dict[str, Prompt]

type Prompts = dict[Literal["yesno", "factoid", "list", "summary"], PromptsForType]
