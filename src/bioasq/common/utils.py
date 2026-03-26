import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any


def typer_async[T](f: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    @wraps(f)
    def wrapper(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> T:
        return asyncio.run(f(*args, **kwargs))

    return wrapper
