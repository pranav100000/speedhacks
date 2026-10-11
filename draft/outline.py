"""transcript.json -> outline.json. See SCOPE.md §3 and §4E.

Claude produces a title, terse first-person notes, and keep_phrases lifted
character-for-character from the speaker's (user) turns only. After parsing,
every keep_phrase is checked to be a substring of the user turns; a miss raises.

    uv run python -m draft.outline --run runs/<id>
"""

import argparse
import asyncio
import difflib
from pathlib import Path

from pydantic import BaseModel, Field

from common.claude import parse
from common.schema import Outline, Transcript, TranscriptTurn, write_json

FORMAT = "blog"
MIN_PHRASES = 0
MAX_PHRASES = 2
TURN_SIMILARITY_MIN = 0.80
MIN_WORDS_FOR_TURN_CHECK = 8
OVERALL_SIMILARITY_MIN = 0.92

SYSTEM = f"""You turn an interview transcript into the raw material for a post the speaker will write in their own voice.

The transcript has two roles. <interviewer> turns are the person asking questions. <speaker> turns are the person whose post this is. Only the speaker's words count. Never attribute anything the interviewer said to the speaker.

Produce:
- title: a short working title for the post, in the speaker's framing. No clickbait, no colon-subtitle.
- notes: 5 to 12 terse first-person bullets, one per line, each starting with "- ". Fragments, not prose. No topic sentences. Capture the substance: the story, the numbers, the opinion, the mistake, the advice. Keep the speaker's own framing and words; do not tidy their language into something more polished.
- keep_phrases: {MIN_PHRASES} to {MAX_PHRASES} spans of 4 to 12 words, copied character-for-character from the SPEAKER TURNS ONLY section. Only a line a reader would quote qualifies: an opinion, a joke, a blunt verdict. Never a fact, a number, a description of how something works, or a sentence that starts with "I would say" or "I think". If nothing qualifies, return an empty list; that is the normal outcome for most interviews. Skip any span with spoken disfluencies: repeated words, fillers ("um", "like,", "kind of", "you know"), false starts, or a comma that only exists because of the transcription.
- format: always "{FORMAT}".
"""


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


CLEAN_SYSTEM = """You fix speech-to-text errors in an interview transcript. You are given the speaker's turns, numbered. Return the same turns, same count, same order, with ONLY transcription errors fixed: words the recognizer dropped (a missing "I" or "the"), words it misheard (a wrong word that makes no sense in context, like "and" for "in"), and run-together or split words.

Keep everything else exactly as spoken: fillers, slang, "gonna", repeated words, false starts, odd commas, lowercase, grammar quirks. Do not merge, split, reorder, shorten, or polish turns. Do not add anything the speaker did not say. If a turn needs no fix, return it unchanged."""


class CleanedTurns(BaseModel):
    turns: list[str] = Field(description="The speaker turns with transcription errors fixed, same count and order as given.")


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_ws(a).lower().split(), normalize_ws(b).lower().split()).ratio()


async def clean_transcript(transcript: Transcript, effort: str) -> Transcript:
    """Fix ASR errors in user turns only, guarded so the pass cannot rewrite the speaker's voice."""
    user_turns = [t for t in transcript.turns if t.role == "user"]
    numbered = "\n".join(f"{i + 1}. {t.text}" for i, t in enumerate(user_turns))
    cleaned = await parse(CLEAN_SYSTEM, f"Speaker turns:\n{numbered}", CleanedTurns, effort=effort)
    if len(cleaned.turns) != len(user_turns):
        raise ValueError(f"cleanup returned {len(cleaned.turns)} turns for {len(user_turns)} user turns")
    for i, (before, after) in enumerate(zip(user_turns, cleaned.turns)):
        if len(before.text.split()) < MIN_WORDS_FOR_TURN_CHECK:
            continue  # a one-word fix dominates a short turn; the overall ratio still bounds it
        sim = _similarity(before.text, after)
        if sim < TURN_SIMILARITY_MIN:
            raise ValueError(f"cleanup rewrote user turn {i + 1} too much (similarity {sim:.2f}):\n  before: {before.text}\n  after:  {after}")
    overall = _similarity(" ".join(t.text for t in user_turns), " ".join(cleaned.turns))
    if overall < OVERALL_SIMILARITY_MIN:
        raise ValueError(f"cleanup changed the transcript too much overall (similarity {overall:.2f})")
    fixed = iter(cleaned.turns)
    turns = [TranscriptTurn(role=t.role, text=next(fixed)) if t.role == "user" else t for t in transcript.turns]
    changed = sum(1 for a, b in zip(user_turns, cleaned.turns) if a.text != b)
    print(f"cleanup: {changed}/{len(user_turns)} user turns changed, overall similarity {overall:.3f}")
    return Transcript(id=transcript.id, started_at=transcript.started_at, turns=turns)


def user_turns_text(transcript: Transcript) -> str:
    return "\n".join(t.text for t in transcript.turns if t.role == "user")


def render_transcript(transcript: Transcript) -> str:
    tagged = "\n".join(
        f"<{'speaker' if t.role == 'user' else 'interviewer'}>{t.text}</{'speaker' if t.role == 'user' else 'interviewer'}>"
        for t in transcript.turns
    )
    speaker_only = "\n".join(f"<speaker>{t.text}</speaker>" for t in transcript.turns if t.role == "user")
    return (
        f"<transcript>\n{tagged}\n</transcript>\n\n"
        "SPEAKER TURNS ONLY (copy keep_phrases from here, character-for-character):\n"
        f"<speaker_turns>\n{speaker_only}\n</speaker_turns>\n"
    )


def validate_keep_phrases(outline: Outline, transcript: Transcript) -> None:
    """Every keep_phrase must be a verbatim span of the user turns (whitespace-normalized)."""
    haystack = normalize_ws(user_turns_text(transcript))
    n = len(outline.keep_phrases)
    if not MIN_PHRASES <= n <= MAX_PHRASES:
        raise ValueError(f"expected {MIN_PHRASES} to {MAX_PHRASES} keep_phrases, got {n}: {outline.keep_phrases}")
    for phrase in outline.keep_phrases:
        if normalize_ws(phrase) not in haystack:
            raise ValueError(f"keep_phrase is not a verbatim span of the user turns: {phrase!r}")


async def make_outline(transcript: Transcript, effort: str) -> Outline:
    drafted = await parse(SYSTEM, render_transcript(transcript), Outline, effort=effort)
    outline = Outline(
        title=drafted.title,
        format=FORMAT,
        notes=drafted.notes,
        keep_phrases=drafted.keep_phrases,
    )
    validate_keep_phrases(outline, transcript)
    return outline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs/<id> directory containing transcript.json")
    ap.add_argument("--effort", default="medium", help="Claude effort level (default: medium)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    transcript_path = run_dir / "transcript.json"
    transcript = Transcript.model_validate_json(transcript_path.read_text())
    if not any(t.role == "user" for t in transcript.turns):
        raise ValueError(f"{transcript_path} has no user turns; nothing to outline")

    cleaned = asyncio.run(clean_transcript(transcript, args.effort))
    write_json(run_dir / "transcript_clean.json", cleaned)
    outline = asyncio.run(make_outline(cleaned, args.effort))
    out = run_dir / "outline.json"
    write_json(out, outline)
    print(
        f"wrote {out}: title={outline.title!r}, "
        f"{len(outline.notes.strip().splitlines())} note lines, {len(outline.keep_phrases)} keep_phrases"
    )


if __name__ == "__main__":
    main()
