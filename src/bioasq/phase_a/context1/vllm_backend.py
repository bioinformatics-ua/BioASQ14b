"""OpenAI-compatible vLLM client for Context-1 tool-calling inference."""

import json
import re
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI

from bioasq.phase_a.context1.types import ToolCallSpec, ToolName

_RECOVERABLE_TOOL_NAMES = (
    ToolName.SEARCH_CORPUS,
    ToolName.GREP_CORPUS,
    ToolName.READ_DOCUMENT,
    ToolName.PRUNE_CHUNKS,
)


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
        for index, tool_call in enumerate(message.tool_calls or []):
            function = getattr(tool_call, "function", None)
            raw_name = getattr(function, "name", None)
            arguments = self._parse_tool_arguments(getattr(function, "arguments", None))
            call_id = self._coerce_tool_call_id(getattr(tool_call, "id", None), index=index)
            parsed_name = self._normalize_tool_name(raw_name)
            if parsed_name is None:
                invalid_arguments = dict(arguments)
                invalid_arguments["__raw_tool_name__"] = self._stringify_tool_name(raw_name)
                tool_calls.append(
                    ToolCallSpec(
                        call_id=call_id,
                        name=ToolName.INVALID,
                        arguments=invalid_arguments,
                    )
                )
                continue
            tool_calls.append(
                ToolCallSpec(
                    call_id=call_id,
                    name=parsed_name,
                    arguments=arguments,
                )
            )

        return ModelTurn(
            content=self._coalesce_content(message.content),
            tool_calls=tool_calls,
            raw_response=cast("dict[str, Any]", response.model_dump(mode="json")),
        )

    @staticmethod
    def _coalesce_content(content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if not all(isinstance(key, str) for key in item):
                    continue
                typed_item = cast("dict[str, object]", item)
                if typed_item.get("type") == "text":
                    text = typed_item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        return str(content)

    @staticmethod
    def _parse_tool_arguments(raw_arguments: object) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            if not all(isinstance(key, str) for key in raw_arguments):
                return {"value": raw_arguments}
            typed_arguments = cast("dict[str, Any]", raw_arguments)
            return dict(typed_arguments)
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, str):
            stripped = raw_arguments.strip()
            if not stripped:
                return {}
            try:
                decoded = json.loads(stripped)
            except (TypeError, ValueError, json.JSONDecodeError):
                return {"raw": stripped}
            if isinstance(decoded, dict):
                return decoded
            return {"value": decoded}
        return {"value": raw_arguments}

    @staticmethod
    def _coerce_tool_call_id(raw_call_id: object, *, index: int) -> str:
        if isinstance(raw_call_id, str) and raw_call_id.strip():
            return raw_call_id
        return f"tool_call_{index}"

    @classmethod
    def _normalize_tool_name(cls, raw_name: object) -> ToolName | None:
        if not isinstance(raw_name, str):
            return None
        candidate = raw_name.strip().strip("`\"'")
        if not candidate:
            return None
        lowered = candidate.lower()
        for tool_name in _RECOVERABLE_TOOL_NAMES:
            if lowered == tool_name.value:
                return tool_name

        matches = [
            tool_name
            for tool_name in _RECOVERABLE_TOOL_NAMES
            if re.search(
                rf"(?<![a-z0-9_]){re.escape(tool_name.value)}(?![a-z0-9_])",
                lowered,
            )
        ]
        if len(matches) == 1:
            return matches[0]

        for tool_name in _RECOVERABLE_TOOL_NAMES:
            if lowered.startswith(tool_name.value) or lowered.endswith(tool_name.value):
                return tool_name
        return None

    @staticmethod
    def _stringify_tool_name(raw_name: object) -> str:
        if isinstance(raw_name, str) and raw_name:
            return raw_name
        return repr(raw_name)
