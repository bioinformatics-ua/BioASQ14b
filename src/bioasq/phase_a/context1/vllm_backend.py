"""OpenAI-compatible vLLM client for Context-1 tool-calling inference."""

import json
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI

from bioasq.phase_a.context1.types import ToolCallSpec, ToolName


@dataclass(slots=True)
class ModelTurn:
    """Assistant response returned by the vLLM-served model."""

    content: str
    tool_calls: list[ToolCallSpec]
    raw_response: dict[str, Any]


class Context1VLLMOpenAIBackend:
    """Thin client over a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        temperature: float = 0.2,
        max_completion_tokens: int = 2_048,
        timeout: float = 300.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        normalized_base_url = base_url.rstrip("/")
        if not normalized_base_url.endswith("/v1"):
            normalized_base_url = f"{normalized_base_url}/v1"
        self._client = AsyncOpenAI(
            base_url=normalized_base_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""

        await self._client.close()

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        seed: int | None = None,
    ) -> ModelTurn:
        """Run one assistant turn with tool calling enabled."""

        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=cast("Any", messages),
            tools=cast("Any", tools),
            tool_choice="auto",
            parallel_tool_calls=True,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            seed=seed,
        )
        message = response.choices[0].message

        tool_calls: list[ToolCallSpec] = []
        for tool_call in message.tool_calls or []:
            try:
                arguments = json.loads(tool_call.function.arguments)
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
            except json.JSONDecodeError:
                arguments = {"raw": tool_call.function.arguments}
            tool_calls.append(
                ToolCallSpec(
                    call_id=tool_call.id,
                    name=ToolName(tool_call.function.name),
                    arguments=arguments,
                )
            )

        return ModelTurn(
            content=self._coalesce_content(message.content),
            tool_calls=tool_calls,
            raw_response=cast("dict[str, Any]", response.model_dump(mode="json")),
        )

    @staticmethod
    def _coalesce_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        return str(content)
