"""corpus.jsonl -> voice_profile.md: a short style profile Claude can use for few-shot drafting.

Selects all blogs first, then linkedin_posts, then threads, then tweets, most recent first
within each format, until --n samples.

Usage:
  uv run python -m synth.voice_profile --corpus data/corpus.jsonl --out data/voice_profile.md --n 40
"""

import argparse
import asyncio
import os
import tempfile
from pathlib import Path

from common.claude import complete
from common.schema import CorpusRecord, read_jsonl
from synth.util import created_at_key

SELECTION_ORDER = ("blog", "email", "linkedin_post", "thread", "tweet")

SECTIONS = (
    "Sentence length and rhythm",
    "How pieces open and close",
    "Recurring words and punctuation habits",
    "Things this writer never does",
    "AI tells to avoid",
)

SYSTEM = (
    "You are a writing-style analyst. You read a writer's real posts and describe, concretely and "
    "briefly, how they write, so that a ghostwriter can match their voice. You cite habits you can actually "
    "see in the samples, with short quoted examples. You do not flatter, you do not generalize, "
    "and you do not describe the topics; only the voice."
)


def select_samples(corpus: list[CorpusRecord], n: int) -> list[CorpusRecord]:
    selected: list[CorpusRecord] = []
    for fmt in SELECTION_ORDER:
        group = sorted((r for r in corpus if r.format == fmt), key=created_at_key, reverse=True)
        selected.extend(group)
        if len(selected) >= n:
            break
    return selected[:n]


def user_prompt(samples: list[CorpusRecord]) -> str:
    blocks = []
    for i, r in enumerate(samples, start=1):
        title = f" | title: {r.title}" if r.title else ""
        blocks.append(f"### Sample {i} ({r.format}{title})\n{r.text}")
    sample_text = "\n\n".join(blocks)
    headings = "\n".join(f"## {s}" for s in SECTIONS)
    return (
        f"Here are {len(samples)} real pieces by one writer.\n\n{sample_text}\n\n"
        "Write a voice profile of this writer in markdown, at most 350 words total (hard limit), using exactly these "
        f"five level-2 headings in this order and nothing before the first one:\n{headings}\n\n"
        "Under each heading use terse bullets with short quoted examples from the samples. "
        "The last section is a blacklist: specific phrases and structures this writer would never use "
        "(for example stock AI openers, tidy triplets, summary closers, hedging words), so a ghostwriter "
        "matching their voice can avoid those. Be concrete. No preamble, no closing remarks."
    )


def check_profile(profile: str) -> None:
    lowered = profile.lower()
    missing = [s for s in SECTIONS if s.lower() not in lowered]
    if missing:
        raise RuntimeError(f"voice profile is missing sections: {missing}")


def write_atomic(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="input data/corpus.jsonl")
    ap.add_argument("--out", required=True, help="output data/voice_profile.md")
    ap.add_argument("--n", type=int, default=40, help="number of samples to show Claude")
    args = ap.parse_args()

    corpus = read_jsonl(args.corpus, CorpusRecord)
    samples = select_samples(corpus, args.n)
    if not samples:
        raise ValueError(f"no samples with format in {SELECTION_ORDER} found in {args.corpus}")
    by_format = {fmt: sum(1 for r in samples if r.format == fmt) for fmt in SELECTION_ORDER}
    print(f"selected {len(samples)} samples: {by_format}")

    profile = asyncio.run(complete(SYSTEM, user_prompt(samples), effort="medium"))
    profile = profile.strip() + "\n"
    check_profile(profile)
    write_atomic(args.out, profile)
    print(f"wrote {args.out} ({len(profile.split())} words)")


if __name__ == "__main__":
    main()
