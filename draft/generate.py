"""outline.json -> drafts/{vanilla,fewshot,finetune}.md. See SCOPE.md §4E.

vanilla:  plain Claude, "write a blog post from these notes".
fewshot:  Claude with the voice profile plus corpus examples, fed the exact
          render_user_message() string the fine-tune sees.
finetune: shells out to train.sample (the Tinker sampler).

After each draft is written, prints how many keep_phrases survived verbatim.

    uv run python -m draft.generate --run runs/<id> --profile data/voice_profile.md \
        --corpus data/corpus.jsonl [--examples 25] [--only vanilla --only fewshot]
"""

import argparse
import asyncio
import subprocess
from collections import Counter
from pathlib import Path

from common.claude import complete, parse
from common.text import normalize_ws, survived, write_text
from common.schema import CorpusRecord, Outline, Transcript, read_jsonl, render_user_message
from draft.judge import CLAIMS_SYSTEM, Claims, check_claims, claims_user

VARIANTS = ("vanilla", "fewshot", "finetune")
EXAMPLE_FORMAT_ORDER = ("blog", "linkedin_post", "thread", "email")

VANILLA_SYSTEM = "You are a helpful writing assistant."
FEWSHOT_RULES = (
    "Rules: output only the post, in markdown, with no preamble and no commentary. "
    'Each line under "Lines to echo" must be recognizably present in the post: keep its meaning and its key '
    "words, and fix spoken grammar if it reads badly on the page. If the list says (none), ignore it. "
    "Use only facts, numbers, names, events, and comparisons that appear in the notes, the phrases, or the "
    "speaker's own words. Do not invent examples, figures, timelines, or consequences. If a detail is missing, "
    "leave it out rather than make one up. Structure, pacing, and sentence-level craft are yours; facts are not. "
    "Use markdown headers, bold, or bullet lists only if the example posts use them; otherwise plain paragraphs."
)
SPEAKER_WORDS_NOTE = (
    "Speaker's own words from the interview, for voice and facts. Quote beyond the listed phrases only if verbatim."
)


def select_examples(records: list[CorpusRecord], n: int) -> list[CorpusRecord]:
    """All blogs first, then linkedin posts, then threads, up to n."""
    ordered: list[CorpusRecord] = []
    for fmt in EXAMPLE_FORMAT_ORDER:
        group = [r for r in records if r.format == fmt]
        if fmt == "email":
            group.sort(key=lambda r: len(r.text.split()), reverse=True)
        ordered.extend(group)
    if not ordered:
        raise ValueError(f"corpus has no records with format in {EXAMPLE_FORMAT_ORDER}")
    return ordered[:n]


def vanilla_user(outline: Outline) -> str:
    return f"Write a blog post from these notes:\n\nTitle: {outline.title}\n\nNotes:\n{outline.notes.strip()}\n"


def fewshot_system(profile: str, examples: list[CorpusRecord]) -> str:
    blocks = "\n\n".join(f'<example format="{r.format}">\n{r.text.strip()}\n</example>' for r in examples)
    return f"{profile.strip()}\n\nWrite exactly like the author of these examples.\n\n{blocks}\n\n{FEWSHOT_RULES}"


async def draft_vanilla(outline: Outline, effort: str) -> str:
    return await complete(VANILLA_SYSTEM, vanilla_user(outline), effort=effort)


def speaker_words(run_dir: Path) -> str | None:
    """User turns from transcript_clean.json, else transcript.json, else None for outline-only runs."""
    for name in ("transcript_clean.json", "transcript.json"):
        path = run_dir / name
        if path.exists():
            transcript = Transcript.model_validate_json(path.read_text())
            return "\n".join(t.text for t in transcript.turns if t.role == "user")
    return None


async def draft_fewshot(
    outline: Outline, profile: str, examples: list[CorpusRecord], effort: str, words: str | None, cand_dir: Path
) -> str:
    """FEWSHOT_CANDIDATES drafts, keep the one with the fewest unsupported claims."""
    user = render_user_message(outline.format, outline.notes, outline.keep_phrases)
    if words:
        user += f"\n\n{SPEAKER_WORDS_NOTE}\n<speaker_words>\n{words}\n</speaker_words>\n"
    system = fewshot_system(profile, examples)
    texts = await asyncio.gather(*(complete(system, user, effort=effort) for _ in range(FEWSHOT_CANDIDATES)))
    candidates = []
    for i, text in enumerate(texts, start=1):
        name = f"fewshot.{i}.md"
        write_text(cand_dir / name, text)
        candidates.append((name, text))
    _, best = await pick_best(candidates, outline, words)
    return best


FINETUNE_CANDIDATES = 4
FINETUNE_TEMPERATURE = "0.6"  # 0.8 roughly tripled unsupported claims on the same outline
FEWSHOT_CANDIDATES = 2


async def count_unsupported(name: str, draft: str, outline: Outline, words: str | None) -> int:
    """How many factual claims in the draft the sources do not support (same checker as draft.judge)."""
    user_turns = words.splitlines() if words else []
    claims = await parse(CLAIMS_SYSTEM, claims_user(outline, user_turns, draft), Claims, effort="low")
    check_claims(name, draft, claims)
    return len(claims.unsupported)


async def pick_best(
    candidates: list[tuple[str, str]], outline: Outline, words: str | None
) -> tuple[str, str]:
    """Fewest unsupported claims wins; ties go to more keep_phrases kept, then more words."""
    counts = await asyncio.gather(*(count_unsupported(n, t, outline, words) for n, t in candidates))
    scored = []
    for (name, text), claims in zip(candidates, counts):
        kept = survived(text, outline.keep_phrases)
        scored.append((claims, kept, len(text.split()), name, text))
        print(f"  candidate {name}: {claims} unsupported claims, kept {kept}/{len(outline.keep_phrases)}, {len(text.split())} words")
    best = min(scored, key=lambda t: (t[0], -t[1], -t[2]))
    print(f"  chose {best[3]}")
    return best[3], best[4]


def sample_finetune(run_dir: Path, cand_dir: Path, sampler: str) -> list[tuple[str, str]]:
    first = cand_dir / "finetune.md"
    subprocess.run(
        [
            "uv", "run", "python", "-m", "train.sample",
            "--outline", str(run_dir / "outline.json"),
            "--out", str(first),
            "--n", str(FINETUNE_CANDIDATES),
            "--sampler", sampler,
            "--temperature", FINETUNE_TEMPERATURE,
        ],
        check=True,
    )
    paths = [first] + [cand_dir / f"finetune.{i}.md" for i in range(2, FINETUNE_CANDIDATES + 1)]
    out: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"train.sample did not write {path}")
        text = path.read_text()
        if not text.strip():
            raise RuntimeError(f"train.sample wrote an empty {path}")
        out.append((path.name, text))
    return out


async def draft_finetune(run_dir: Path, out: Path, outline: Outline, sampler: str, words: str | None) -> str:
    """Sample FINETUNE_CANDIDATES drafts, keep the one with the fewest unsupported claims."""
    cand_dir = out.parent / "finetune_candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    candidates = await asyncio.to_thread(sample_finetune, run_dir, cand_dir, sampler)
    _, text = await pick_best(candidates, outline, words)
    write_text(out, text)
    return text


async def run_variant(
    name: str, outline: Outline, run_dir: Path, profile: str, examples: list[CorpusRecord], effort: str, sampler: str
) -> str:
    out = run_dir / "drafts" / f"{name}.md"
    if name == "vanilla":
        text = await draft_vanilla(outline, effort)
        write_text(out, text)
    elif name == "fewshot":
        text = await draft_fewshot(outline, profile, examples, effort, speaker_words(run_dir), out.parent / "fewshot_candidates")
        write_text(out, text)
    elif name == "finetune":
        text = await draft_finetune(run_dir, out, outline, sampler, speaker_words(run_dir))
    else:
        raise ValueError(f"unknown variant {name!r}")
    print(f"{out}: keep phrases survived {survived(text, outline.keep_phrases)}/{len(outline.keep_phrases)}")
    return text


async def run_all(
    variants: tuple[str, ...], outline: Outline, run_dir: Path, profile: str, examples: list[CorpusRecord], effort: str, sampler: str
) -> None:
    await asyncio.gather(*(run_variant(v, outline, run_dir, profile, examples, effort, sampler) for v in variants))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs/<id> directory containing outline.json")
    ap.add_argument("--profile", required=True, help="voice profile markdown, e.g. data/voice_profile.md")
    ap.add_argument("--corpus", required=True, help="corpus jsonl, e.g. data/corpus.jsonl")
    ap.add_argument("--examples", type=int, default=25, help="max few-shot examples (default: 25)")
    ap.add_argument(
        "--only", action="append", choices=VARIANTS, help="variant to generate; repeatable (default: all three)"
    )
    ap.add_argument("--effort", default="medium", help="Claude effort level (default: medium)")
    ap.add_argument("--sampler", default="data/sampler.json", help="sampler descriptor for the finetune column (default: data/sampler.json)")
    args = ap.parse_args()

    variants = tuple(dict.fromkeys(args.only)) if args.only else VARIANTS
    run_dir = Path(args.run)
    outline = Outline.model_validate_json((run_dir / "outline.json").read_text())

    profile_path = Path(args.profile)
    profile = profile_path.read_text()
    if not profile.strip():
        raise ValueError(f"{profile_path} is empty")
    examples = select_examples(read_jsonl(args.corpus, CorpusRecord), args.examples)
    by_format = Counter(r.format for r in examples)
    print(f"{len(examples)} few-shot examples: " + ", ".join(f"{k}={v}" for k, v in by_format.items()))

    asyncio.run(run_all(variants, outline, run_dir, profile, examples, args.effort, args.sampler))


if __name__ == "__main__":
    main()
