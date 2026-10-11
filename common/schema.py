"""Every contract that crosses a workstream boundary. See SCOPE.md §3.

One function, `render_user_message`, builds the model's user turn for BOTH
training and inference. Never build that string anywhere else.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Literal, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

Source = Literal["x", "linkedin", "gmail", "blog"]
Format = Literal["tweet", "thread", "reply", "linkedin_post", "email", "blog", "section"]

TRAINABLE_FORMATS: tuple[Format, ...] = ("blog", "section", "email", "linkedin_post", "thread", "tweet", "reply")

SYSTEM_PROMPT_TEMPLATE = "You are {name}. Write in your own voice."


def author_name() -> str:
    name = os.environ.get("AUTHOR_NAME")
    if not name:
        raise RuntimeError("AUTHOR_NAME is not set in the environment or .env")
    return name


def system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(name=author_name())


# ---------- data/corpus.jsonl ----------


class CorpusRecord(BaseModel):
    id: str
    source: Source
    format: Format
    text: str = Field(min_length=1)
    created_at: str
    title: str | None = None


# ---------- data/pairs.jsonl ----------


class PairRecord(BaseModel):
    id: str
    format: Format
    notes: str = Field(min_length=1)
    keep_phrases: list[str] = Field(default_factory=list, max_length=4)
    text: str = Field(min_length=1)


# ---------- data/train.jsonl, data/val.jsonl (Tinker conversation format) ----------


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class TrainRecord(BaseModel):
    messages: list[Message] = Field(min_length=3, max_length=3)


# ---------- runs/<id>/transcript.json ----------


class TranscriptTurn(BaseModel):
    role: Literal["assistant", "user"]
    text: str


class Transcript(BaseModel):
    id: str
    started_at: str
    turns: list[TranscriptTurn]


# ---------- runs/<id>/outline.json ----------


class Outline(BaseModel):
    title: str
    format: Format
    notes: str = Field(min_length=1)
    keep_phrases: list[str] = Field(default_factory=list, max_length=2)


# ---------- the one prompt renderer ----------


def render_user_message(format: Format, notes: str, keep_phrases: list[str]) -> str:
    """The user turn the voice model sees. Used by make_train.py AND sample.py.

    Lines to echo are soft: the writer keeps their meaning and key words and may fix spoken grammar.
    An empty list renders as "(none)" so the model sees that case in training too.
    """
    lines = "\n".join(f"- {p}" for p in keep_phrases) if keep_phrases else "- (none)"
    return (
        f"Format: {format}\n\n"
        f"Notes:\n{notes.strip()}\n\n"
        f"Lines to echo (fix spoken grammar if needed):\n{lines}\n"
    )


def build_messages(
    format: Format, notes: str, keep_phrases: list[str], text: str | None = None
) -> list[Message]:
    """System + user, plus the assistant turn when building a training record."""
    messages = [
        Message(role="system", content=system_prompt()),
        Message(role="user", content=render_user_message(format, notes, keep_phrases)),
    ]
    if text is not None:
        messages.append(Message(role="assistant", content=text))
    return messages


# ---------- jsonl io ----------

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: str | Path, model: type[T]) -> list[T]:
    path = Path(path)
    records: list[T] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(model.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"{path}:{line_no} does not validate as {model.__name__}: {e}") from e
    return records


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    """Atomic write: full file to a temp sibling, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for r in records:
            f.write(r.model_dump_json())
            f.write("\n")
            count += 1
    os.replace(tmp, path)
    return count


def write_json(path: str | Path, record: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(record.model_dump(), indent=2))
    os.replace(tmp, path)
