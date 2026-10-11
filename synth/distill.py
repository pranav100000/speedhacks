"""Bootstrap long-form training data in the author's voice.

For each of N seeds: sample three real corpus records, have Claude write the interview-style
notes a founder like this would give on a topic those records support (facts only from the
records), have the few-shot writer (profile + real posts) draft the post, keep it only if the
claims judge finds at most --max-claims unsupported claims. Output is pairs in the usual shape,
format "blog", ids synth_<i>, with no echoed lines.

  uv run python -m synth.distill --corpus data/corpus.jsonl --profile data/voice_profile.md \
      --out data/pairs_synth.jsonl --n 80
"""

import argparse
import asyncio
import random
from pathlib import Path

from pydantic import BaseModel, Field

from common.claude import complete, gather_limited, parse
from common.schema import CorpusRecord, Outline, PairRecord, read_jsonl, render_user_message, write_jsonl
from draft.generate import count_unsupported, fewshot_system, select_examples

SEED_SYSTEM = (
    "You write the terse first-person notes a founder would give an interviewer before writing a post. "
    "You are shown a few things this founder has actually written. Pick ONE topic those pieces support "
    "and write 6 to 10 note bullets for a post on it. Every fact, number, name, opinion, and story in the "
    "notes must come from the pieces shown; you may connect them, you may not invent. Fragments, the way "
    "someone talks, no polish, no topic sentences."
)


class NoteSet(BaseModel):
    title: str = Field(description="Short working title in the founder's framing")
    notes: str = Field(description='6 to 10 bullets, one per line, each starting with "- "')


def seed_prompt(records: list[CorpusRecord]) -> str:
    blocks = "\n\n".join(f'<piece format="{r.format}">\n{r.text.strip()}\n</piece>' for r in records)
    return f"{blocks}\n\nWrite the title and the notes for one post these pieces support."


async def one(i: int, records: list[CorpusRecord], system: str, max_claims: int) -> PairRecord | None:
    seed = await parse(SEED_SYSTEM, seed_prompt(records), NoteSet, effort="low")
    outline = Outline(title=seed.title, format="blog", notes=seed.notes.strip(), keep_phrases=[])
    draft = await complete(system, render_user_message("blog", outline.notes, []), effort="low")
    claims = await count_unsupported(f"synth_{i}", draft, outline, None)
    words = len(draft.split())
    if claims > max_claims or words < 120:
        print(f"  synth_{i}: dropped ({claims} claims, {words} words)")
        return None
    print(f"  synth_{i}: kept ({claims} claims, {words} words) {seed.title[:60]}")
    return PairRecord(id=f"synth_{i}", format="blog", notes=outline.notes, keep_phrases=[], text=draft.strip())


async def run(args: argparse.Namespace) -> None:
    corpus = read_jsonl(args.corpus, CorpusRecord)
    long_form = [r for r in corpus if r.format in ("blog", "email")]
    short_form = [r for r in corpus if r.format in ("tweet", "reply") and len(r.text.split()) >= 12]
    if not long_form or not short_form:
        raise ValueError("need both long-form and substantive short-form records to seed from")
    rng = random.Random(args.seed)
    profile = Path(args.profile).read_text()
    system = fewshot_system(profile, select_examples(corpus, args.examples))

    seeds = []
    for i in range(args.n):
        picks = [rng.choice(long_form)] + rng.sample(short_form, 2)
        seeds.append((i, picks))

    results = await gather_limited(seeds, lambda s: one(s[0], s[1], system, args.max_claims), concurrency=args.concurrency)
    kept = [p for p in results if p is not None]
    n = write_jsonl(args.out, kept)
    print(f"wrote {n} synthetic pairs to {args.out} ({args.n - n} dropped), {sum(len(p.text.split()) for p in kept)} words")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--examples", type=int, default=25)
    ap.add_argument("--max-claims", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
