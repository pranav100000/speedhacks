"""Score the drafts of one run: voice likeness, unsupported claims, repeats, AI tells.

Per draft in runs/<id>/drafts/<name>.md (and runs/<id>/real.md as "real" with --include-real):
  voice_score            Claude rubric, 1-10, likeness to 15 corpus reference posts (not quality)
  unsupported_claims     Claude, factual claims not backed by outline notes / keep phrases / user turns
  repeated_ngrams        6-grams occurring more than once
  ai_tells               markdown headers, bold, built-in phrase list, "AI tells to avoid" from the profile
  keep_phrases_survived  same whitespace-normalized substring logic as draft.generate
  words
--pairwise: every pair of drafts, both orders, Claude picks which one the reference author wrote.

    uv run python -m draft.judge --run runs/<id> --corpus data/corpus.jsonl [--profile data/voice_profile.md]
        [--out runs/<id>/judge.json] [--drafts vanilla fewshot finetune] [--include-real] [--pairwise]
"""

import argparse
import asyncio
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.claude import gather_limited, parse
from common.schema import CorpusRecord, Outline, Transcript, read_jsonl
from common.text import normalize_ws, survived, write_text

DEFAULT_DRAFTS = ("vanilla", "fewshot", "finetune")
REFERENCE_FORMAT_ORDER = ("blog", "tweet", "reply")
REFERENCE_COUNT = 15
NGRAM = 6
BUILTIN_TELLS = (
    "in today's", "let's dive", "here's the thing", "game-changer", "game changer", "revolutioniz",
    "seamless", "in conclusion", "in short", "at the end of the day", "the bottom line", "it's not just",
    "it's not about", "this is the story of", "night and day", "journey",
)

# ---------- output models ----------


SCORE_DESC = "integer from 1 (nothing like the author) to 10 (indistinguishable)"


class RubricItem(BaseModel):
    score: int = Field(ge=0, le=10, description=SCORE_DESC)
    reason: str = Field(description="one sentence")


class VoiceScore(BaseModel):
    sentence_rhythm: RubricItem
    vocabulary: RubricItem
    structure_and_format: RubricItem
    openers_and_closers: RubricItem
    ai_tells: RubricItem
    overall: int = Field(ge=0, le=10, description=SCORE_DESC)


class Claim(BaseModel):
    span: str
    why: str


class Claims(BaseModel):
    unsupported: list[Claim]


class PairwiseChoice(BaseModel):
    choice: Literal["A", "B"]
    reason: str


# ---------- prompts ----------

VOICE_SYSTEM = (
    "You are a stylometry judge. You will be shown reference posts written by one author, a profile of that "
    "author's voice, and one candidate draft. Score how much the draft reads as if the reference author wrote it. "
    "Judge likeness to the reference author ONLY, never quality: a polished draft that does not sound like them "
    "scores low, a sloppy draft that sounds exactly like them scores high. Every rubric score is an integer from "
    "1 (nothing like the author) to 10 (indistinguishable) with a one-sentence reason. ai_tells is 10 when the "
    "draft has none of the generic LLM habits the author never uses and 1 when it is full of them. overall is "
    "your single 1-10 likeness verdict. The draft is a blog post. The author's tweets and replies in the "
    "references are lowercase and short because of that medium; judge word choice, rhythm, attitude, and how "
    "the author structures long-form writing as shown in the blog references, and do not penalize sentence case "
    "or length that match the author's own blog posts."
)

CLAIMS_SYSTEM = (
    "You are a fact-grounding checker. You will be shown the sources a draft was written from (interview notes, "
    "phrases the author said verbatim, and the author's own spoken turns when available) and the draft. List every "
    "factual claim in the draft that is not stated or directly implied by the sources: numbers, durations, events, "
    "named tools or people, comparisons, quoted consequences. Stylistic framing, opinions and rhetorical emphasis "
    "are not claims. For each one, span must be copied from the draft character-for-character (same quotes, same "
    "punctuation, no ellipsis) and why is one short reason. Return an empty list if everything is grounded."
)

PAIRWISE_SYSTEM = (
    "You will be shown reference posts written by one author, then two drafts labelled A and B. Exactly one of them "
    "is closer to what the reference author would write. Judge likeness to the author only, never quality, and "
    "answer with A or B plus a one-sentence reason. Both drafts are blog posts. The author's tweets and replies "
    "among the references are lowercase and short because of that medium; compare word choice, rhythm, attitude, "
    "and long-form structure against the author's blog references, and do not prefer a draft for sounding like "
    "a tweet or for echoing spoken phrasing."
)


def reference_blocks(refs: list[CorpusRecord]) -> str:
    return "\n\n".join(f'<reference format="{r.format}">\n{r.text.strip()}\n</reference>' for r in refs)


def voice_user(refs: list[CorpusRecord], profile: str | None, draft: str) -> str:
    profile_block = f"\n\n<voice_profile>\n{profile.strip()}\n</voice_profile>" if profile else ""
    return f"{reference_blocks(refs)}{profile_block}\n\n<draft>\n{draft.strip()}\n</draft>"


def claims_user(outline: Outline, user_turns: list[str], draft: str) -> str:
    phrases = "\n".join(f"- {p}" for p in outline.keep_phrases)
    turns = "\n\n".join(user_turns)
    turns_block = f"\n\n<user_turns>\n{turns}\n</user_turns>" if user_turns else ""
    return (
        f"<notes>\n{outline.notes.strip()}\n</notes>\n\n<keep_phrases>\n{phrases}\n</keep_phrases>"
        f"{turns_block}\n\n<draft>\n{draft.strip()}\n</draft>"
    )


def pairwise_user(refs: list[CorpusRecord], a: str, b: str) -> str:
    return f"{reference_blocks(refs)}\n\n<draft label=\"A\">\n{a.strip()}\n</draft>\n\n<draft label=\"B\">\n{b.strip()}\n</draft>"


# ---------- programmatic metrics ----------


def repeated_ngrams(draft: str, n: int = NGRAM) -> list[str]:
    words = draft.lower().split()
    grams = Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return sorted(g for g, c in grams.items() if c > 1)


def profile_tells(profile: str) -> list[str]:
    """Every quoted phrase in the profile's '## AI tells to avoid' section."""
    match = re.search(r"^## AI tells to avoid\s*$(.*?)(?=^## |\Z)", profile, re.M | re.S)
    if not match:
        raise ValueError('voice profile has no "## AI tells to avoid" section')
    phrases = re.findall(r'["“]([^"”]+)["”]', match.group(1))
    if not phrases:
        raise ValueError('"## AI tells to avoid" section has no quoted phrases')
    return phrases


def ai_tells(draft: str, extra_phrases: list[str]) -> dict:
    lines = draft.splitlines()
    headers = sum(1 for line in lines if line.lstrip().startswith("#"))
    bold = draft.count("**") // 2
    body = draft.lower()
    phrases = {p: body.count(p.lower()) for p in dict.fromkeys((*BUILTIN_TELLS, *extra_phrases))}
    phrases = {p: c for p, c in phrases.items() if c}
    return {"headers": headers, "bold": bold, "phrases": phrases, "total": headers + bold + sum(phrases.values())}


def check_rubric(name: str, score: VoiceScore) -> VoiceScore:
    for field, value in score.model_dump().items():
        n = value if field == "overall" else value["score"]
        if not 0 <= n <= 10:
            raise ValueError(f"{name}: voice rubric {field}={n} is not in 0..10")
    return score


def check_claims(name: str, draft: str, claims: Claims) -> Claims:
    body = normalize_ws(draft)
    for claim in claims.unsupported:
        if normalize_ws(claim.span) not in body:
            raise ValueError(f"{name}: claim span is not a substring of the draft: {claim.span!r}")
    return claims


# ---------- loading ----------


def load_drafts(run_dir: Path, names: list[str], include_real: bool) -> dict[str, str]:
    drafts: dict[str, str] = {}
    for name in names:
        path = run_dir / "drafts" / f"{name}.md"
        if path.exists():
            drafts[name] = path.read_text()
            if not drafts[name].strip():
                raise ValueError(f"{path} is empty")
    if include_real:
        real = run_dir / "real.md"
        if not real.exists():
            raise FileNotFoundError(f"--include-real but {real} does not exist")
        drafts["real"] = real.read_text()
    if not drafts:
        raise FileNotFoundError(f"none of {names} exist under {run_dir / 'drafts'}")
    return drafts


def select_references(records: list[CorpusRecord], drafts: dict[str, str], n: int) -> list[CorpusRecord]:
    """All blogs, then tweets, then replies, up to n; any record identical to a draft is a leak and is dropped."""
    draft_bodies = {normalize_ws(d) for d in drafts.values()}
    ordered: list[CorpusRecord] = []
    for fmt in REFERENCE_FORMAT_ORDER:
        for r in records:
            if r.format != fmt:
                continue
            if normalize_ws(r.text) in draft_bodies:
                print(f"leak guard: dropping reference {r.id} because it is identical to a draft")
                continue
            ordered.append(r)
    if len(ordered) < n:
        raise ValueError(f"corpus has only {len(ordered)} usable reference posts, need {n}")
    return ordered[:n]


def load_user_turns(run_dir: Path) -> list[str]:
    path = run_dir / "transcript.json"
    if not path.exists():
        print(f"{path} does not exist; grounding claims against outline notes and keep phrases only")
        return []
    transcript = Transcript.model_validate_json(path.read_text())
    return [t.text for t in transcript.turns if t.role == "user"]


# ---------- orchestration ----------


async def judge_draft(
    name: str, draft: str, refs: list[CorpusRecord], profile: str | None, outline: Outline,
    user_turns: list[str], extra_tells: list[str], effort: str,
) -> dict:
    voice, claims = await asyncio.gather(
        parse(VOICE_SYSTEM, voice_user(refs, profile, draft), VoiceScore, effort=effort),
        parse(CLAIMS_SYSTEM, claims_user(outline, user_turns, draft), Claims, effort=effort),
    )
    check_rubric(name, voice)
    check_claims(name, draft, claims)
    return {
        "voice_score": voice.overall,
        "voice": voice.model_dump(),
        "unsupported_claims": [c.model_dump() for c in claims.unsupported],
        "repeated_ngrams": repeated_ngrams(draft),
        "ai_tells": ai_tells(draft, extra_tells),
        "keep_phrases_survived": survived(draft, outline.keep_phrases),
        "keep_phrases_total": len(outline.keep_phrases),
        "words": len(draft.split()),
    }


async def judge_pairwise(drafts: dict[str, str], refs: list[CorpusRecord], seed: int, effort: str) -> dict:
    rng = random.Random(seed)
    matchups: list[tuple[str, str]] = []
    for pair in itertools.combinations(drafts, 2):
        first, second = rng.sample(pair, 2)
        matchups.extend([(first, second), (second, first)])

    async def one(m: tuple[str, str]) -> PairwiseChoice:
        return await parse(PAIRWISE_SYSTEM, pairwise_user(refs, drafts[m[0]], drafts[m[1]]), PairwiseChoice, effort=effort)

    choices = await gather_limited(matchups, one, concurrency=6)
    wins = {name: 0 for name in drafts}
    records = []
    for (a, b), choice in zip(matchups, choices):
        winner = a if choice.choice == "A" else b
        wins[winner] += 1
        records.append({"A": a, "B": b, "winner": winner, "reason": choice.reason})
    return {"wins": wins, "matchups": records}


def print_table(results: dict[str, dict], wins: dict[str, int] | None) -> None:
    header = f"{'draft':<22} {'voice':>5} {'claims':>6} {'repeats':>7} {'tells':>5} {'kept':>5} {'words':>5} {'pairwise':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        kept = f"{r['keep_phrases_survived']}/{r['keep_phrases_total']}"
        pw = "-" if wins is None else str(wins[name])
        print(
            f"{name:<22} {r['voice_score']:>5} {len(r['unsupported_claims']):>6} {len(r['repeated_ngrams']):>7} "
            f"{r['ai_tells']['total']:>5} {kept:>5} {r['words']:>5} {pw:>8}"
        )


async def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    outline = Outline.model_validate_json((run_dir / "outline.json").read_text())
    drafts = load_drafts(run_dir, args.drafts, args.include_real)
    refs = select_references(read_jsonl(args.corpus, CorpusRecord), drafts, REFERENCE_COUNT)
    print(f"{len(drafts)} drafts: {', '.join(drafts)}; {len(refs)} references: " + ", ".join(
        f"{k}={v}" for k, v in Counter(r.format for r in refs).items()))
    profile = None
    extra_tells: list[str] = []
    if args.profile:
        profile = Path(args.profile).read_text()
        if not profile.strip():
            raise ValueError(f"{args.profile} is empty")
        extra_tells = profile_tells(profile)
    user_turns = load_user_turns(run_dir)

    per_draft = await asyncio.gather(*(
        judge_draft(name, text, refs, profile, outline, user_turns, extra_tells, args.effort)
        for name, text in drafts.items()
    ))
    results = dict(zip(drafts, per_draft))
    pairwise = await judge_pairwise(drafts, refs, args.seed, args.effort) if args.pairwise else None

    out = Path(args.out) if args.out else run_dir / "judge.json"
    payload = {
        "run": run_dir.name,
        "references": [r.id for r in refs],
        "drafts": results,
        "pairwise": pairwise,
    }
    write_text(out, json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print_table(results, pairwise["wins"] if pairwise else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs/<id> directory with outline.json and drafts/")
    ap.add_argument("--corpus", required=True, help="corpus jsonl, e.g. data/corpus.jsonl")
    ap.add_argument("--profile", help="voice profile markdown; also supplies extra AI-tell phrases")
    ap.add_argument("--out", help="output json (default: runs/<id>/judge.json)")
    ap.add_argument("--drafts", nargs="+", default=list(DEFAULT_DRAFTS), help="draft names to judge if present")
    ap.add_argument("--include-real", action="store_true", help='also judge runs/<id>/real.md as "real"')
    ap.add_argument("--pairwise", action="store_true", help="run pairwise which-is-the-author comparisons")
    ap.add_argument("--seed", type=int, default=0, help="seed for pairwise A/B ordering (default: 0)")
    ap.add_argument("--effort", default="medium", help="Claude effort level (default: medium)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
