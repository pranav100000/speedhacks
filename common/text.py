"""Text helpers shared by the drafter and the judge."""

import os
import re
import tempfile
from pathlib import Path


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


SURVIVE_THRESHOLD = 0.7
_STOP = {"the", "and", "that", "this", "with", "for", "was", "were", "are", "you", "your", "they", "them",
         "have", "has", "had", "but", "not", "its", "it's", "just", "then", "than", "into", "from", "would"}


def _content_tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2 and w not in _STOP]


def survived(draft: str, keep_phrases: list[str], threshold: float = SURVIVE_THRESHOLD) -> int:
    """How many keep_phrases are echoed in the draft.

    A phrase survives when some window of the draft (phrase length plus three content words)
    contains at least `threshold` of the phrase's content words. Exact copies always count;
    "the future of coding agents is going to be in the cloud" still counts for the spoken
    "future of coding agents is gonna be in cloud".
    """
    body = _content_tokens(draft)
    hits = 0
    for phrase in keep_phrases:
        toks = _content_tokens(phrase)
        if not toks:
            continue
        need = len(toks) * threshold
        width = len(toks) + 3
        for i in range(max(1, len(body) - width + 1)):
            window = set(body[i : i + width])
            if sum(t in window for t in toks) >= need:
                hits += 1
                break
    return hits


def write_text(path: Path, text: str) -> None:
    """Atomic write: temp sibling, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)
