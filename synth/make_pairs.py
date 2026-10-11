"""corpus.jsonl -> pairs.jsonl: the spoken notes behind each piece + verbatim phrases from it.

Long-form records get extra pairs: note variants at different densities (same target text)
and section-level pairs (format "section") so the model sees more long-form voice.

Usage:
  uv run python -m synth.make_pairs --corpus data/corpus.jsonl --out data/pairs.jsonl --concurrency 10 [--limit N]
"""

import argparse
import asyncio
import re
from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, Field

from common.claude import gather_limited, get_client, parse
from common.schema import TRAINABLE_FORMATS, CorpusRecord, Format, PairRecord, read_jsonl, write_jsonl

MIN_WORDS = 5
VARIANT_MIN_WORDS = 150
SECTION_MIN_WORDS = 300
SECTION_TARGET_WORDS = 180
SECTION_MAX_WORDS = 260

SYSTEM = (
    "You turn a finished piece of writing by a human author back into the terse spoken notes "
    "they would have said out loud to an interviewer before drafting it. First person, fragments, "
    "the specifics, none of the structure. Notes are CONTENT ONLY: what happened, who, when, the "
    "numbers, the names, the opinion, the joke. Never describe the piece itself: no notes about tone, "
    "length, format, emoji, style, audience, or what to do (never 'keep it short', 'it's a reply', "
    "'end with a link', 'deadpan'). You never summarize and you never polish. You also copy a few "
    "exact phrases out of the piece, character for character."
)

BULLETS = {
    "tweet": "1 to 2 bullets",
    "reply": "1 to 2 bullets",
    "section": "2 to 5 bullets",
    "short": "1 to 3 bullets",
    "medium": "2 to 5 bullets",
    "normal": "5 to 8 bullets",
    "sparse": "3 to 4 bullets, only the biggest points",
    "dense": "9 to 12 bullets, every specific detail",
}
PHRASES = {"tweet": "1 to 2", "reply": "1 to 2", "section": "1 to 3", "short": "1 to 2", "medium": "1 to 3", "normal": "2 to 4", "sparse": "2 to 4", "dense": "2 to 4"}


@dataclass(frozen=True)
class Job:
    pair_id: str
    format: Format
    density: str
    text: str


class PairOutput(BaseModel):
    notes: str = Field(description='Spoken notes as bullet lines, each starting with "- ". First person, fragments, content only.')
    keep_phrases: list[str] = Field(description="Exact substrings of the text, 3 to 12 words each, copied character for character.")


def user_prompt(job: Job) -> str:
    return (
        f"Format: {job.format}\n\n"
        f"Text:\n<<<\n{job.text}\n>>>\n\n"
        "Give me two things.\n\n"
        "1. notes: the spoken notes the writer gave an interviewer before writing this. "
        "First-person fragments, the way someone talks out loud. "
        f"{BULLETS[job.density]}, one per line, each starting with \"- \". Content only: names, numbers, "
        "what happened, the opinion. No topic sentences, no summary, and nothing about the style, tone, "
        "length, or format of the piece.\n\n"
        f"2. keep_phrases: {PHRASES[job.density]} short phrases copied character-for-character from the text, "
        "3 to 12 words each, that a reader would quote: an opinion, a joke, a blunt verdict, a concrete number. "
        "Exact substrings only: same punctuation, same casing, same spacing. Do not paraphrase, do not drop "
        "words from the middle of a phrase, do not add quotation marks."
    )


def split_sections(text: str) -> list[str]:
    """Group paragraphs into chunks near SECTION_TARGET_WORDS; a short tail merges into the previous chunk."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[list[str]] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        n = len(para.split())
        if current and count + n > SECTION_MAX_WORDS:
            chunks.append(current)
            current, count = [], 0
        current.append(para)
        count += n
        if count >= SECTION_TARGET_WORDS:
            chunks.append(current)
            current, count = [], 0
    if current:
        if chunks and count < 60:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return ["\n\n".join(c) for c in chunks]


def expand_jobs(records: list[CorpusRecord]) -> list[Job]:
    jobs: list[Job] = []
    too_short = 0
    for r in records:
        words = len(r.text.split())
        if words < MIN_WORDS:
            too_short += 1
            continue
        if r.format in ("tweet", "reply"):
            jobs.append(Job(r.id, r.format, r.format, r.text))
            continue
        density = "short" if words < 80 else "medium" if words < 200 else "normal"
        jobs.append(Job(r.id, r.format, density, r.text))
        if words >= VARIANT_MIN_WORDS:
            jobs.append(Job(f"{r.id}__sparse", r.format, "sparse", r.text))
            jobs.append(Job(f"{r.id}__dense", r.format, "dense", r.text))
        if words >= SECTION_MIN_WORDS:
            sections = split_sections(r.text)
            if len(sections) < 2:
                raise ValueError(f"{r.id}: {words} words but split_sections produced {len(sections)} chunk")
            for i, chunk in enumerate(sections, start=1):
                jobs.append(Job(f"{r.id}__s{i}", "section", "section", chunk))
    print(f"filter: dropped {too_short} records under {MIN_WORDS} words")
    return jobs


META_WORDS = re.compile(r"\b(keep it short|it'?s a reply|it'?s a tweet|deadpan|no emoji|end (it )?with|tone|casual|lowercase)\b", re.I)


def validate(job: Job, out: PairOutput) -> PairRecord:
    if not out.keep_phrases:
        raise ValueError(f"{job.pair_id}: Claude returned no keep_phrases")
    for phrase in out.keep_phrases:
        if not phrase.strip():
            raise ValueError(f"{job.pair_id}: empty keep_phrase")
        if phrase not in job.text:
            raise ValueError(f"{job.pair_id}: keep_phrase is not a substring of text: {phrase!r}")
    meta = [line for line in out.notes.splitlines() if META_WORDS.search(line)]
    if meta:
        raise ValueError(f"{job.pair_id}: notes describe the piece instead of its content: {meta}")
    return PairRecord(id=job.pair_id, format=job.format, notes=out.notes.strip(), keep_phrases=out.keep_phrases, text=job.text)


async def make_pairs(jobs: list[Job], concurrency: int) -> list[PairRecord]:
    client = get_client()
    done = 0

    async def one(job: Job) -> PairRecord:
        nonlocal done
        try:
            out = await parse(SYSTEM, user_prompt(job), PairOutput, effort="low", client=client)
        except Exception as e:
            raise RuntimeError(f"{job.pair_id}: {e}") from e
        pair = validate(job, out)
        done += 1
        if done % 25 == 0 or done == len(jobs):
            print(f"  {done}/{len(jobs)} done", flush=True)
        return pair

    return await gather_limited(jobs, one, concurrency=concurrency)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="input data/corpus.jsonl")
    ap.add_argument("--out", required=True, help="output data/pairs.jsonl")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N trainable records")
    args = ap.parse_args()

    corpus = read_jsonl(args.corpus, CorpusRecord)
    records = [r for r in corpus if r.format in TRAINABLE_FORMATS]
    skipped = Counter(r.format for r in corpus if r.format not in TRAINABLE_FORMATS)
    if args.limit is not None:
        records = records[: args.limit]
    jobs = expand_jobs(records)
    print(
        f"corpus: {len(corpus)} records, {len(records)} trainable, {len(jobs)} jobs after variants and sections, "
        f"skipped by format: {dict(skipped)}"
    )

    pairs = asyncio.run(make_pairs(jobs, args.concurrency))
    n = write_jsonl(args.out, pairs)
    counts = Counter(p.format for p in pairs)
    print(f"wrote {n} pairs to {args.out}")
    for fmt in TRAINABLE_FORMATS:
        print(f"  {fmt:14s} {counts.get(fmt, 0)}")


if __name__ == "__main__":
    main()
