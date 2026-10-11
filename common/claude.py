"""The one Claude client. Claude Opus 5.5, adaptive thinking, streaming.

No fallbacks: a refusal or a truncated response raises.
"""

import asyncio
import os
from typing import Awaitable, Callable, TypeVar

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 16000

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


def get_client() -> anthropic.AsyncAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment or .env")
    return anthropic.AsyncAnthropic()


def _check_stop(message: anthropic.types.Message) -> None:
    if message.stop_reason == "refusal":
        details = message.stop_details
        raise RuntimeError(
            f"Claude refused: category={getattr(details, 'category', None)} "
            f"explanation={getattr(details, 'explanation', None)}"
        )
    if message.stop_reason == "max_tokens":
        raise RuntimeError("Claude hit max_tokens; raise max_tokens for this call")


async def complete(
    system: str,
    user: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = "medium",
    client: anthropic.AsyncAnthropic | None = None,
) -> str:
    """Plain text completion, streamed to avoid timeouts."""
    client = client or get_client()
    async with client.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort},
    ) as stream:
        message = await stream.get_final_message()
    _check_stop(message)
    text = "".join(block.text for block in message.content if block.type == "text")
    if not text.strip():
        raise RuntimeError("Claude returned no text")
    return text


async def parse(
    system: str,
    user: str,
    output_format: type[T],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = "medium",
    client: anthropic.AsyncAnthropic | None = None,
) -> T:
    """Structured output validated into a pydantic model."""
    client = client or get_client()
    message = await client.messages.parse(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort},
        output_format=output_format,
    )
    _check_stop(message)
    if message.parsed_output is None:
        raise RuntimeError("Claude returned no parsed output")
    return message.parsed_output


async def gather_limited(
    items: list[R],
    fn: Callable[[R], Awaitable[T]],
    *,
    concurrency: int = 10,
) -> list[T]:
    """Run fn over items with bounded concurrency. Any failure raises."""
    sem = asyncio.Semaphore(concurrency)

    async def run(item: R) -> T:
        async with sem:
            return await fn(item)

    return list(await asyncio.gather(*(run(i) for i in items)))
