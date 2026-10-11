"""Run the interview over a real phone call with Dial (https://getdial.ai).

Dial hosts the voice agent: we hand it the same interviewer system prompt the browser bot uses,
it calls the founder's phone, holds the conversation, and returns timed transcript turns.
This script writes runs/run_<ts>/transcript.json in the shape the rest of the pipeline reads.

  uv run python -m interview.dial_call --to +14155550123 --corpus data/corpus.jsonl

Needs DIAL_API_KEY in .env and at least one number on the Dial account (or --from-number-id).
"""

import argparse
import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dial_sdk import DialClient, DialConfig
from dial_sdk.types import MakeCallParams
from dotenv import load_dotenv

from common.schema import Transcript, TranscriptTurn, author_name, write_json
from interview.bot import build_system_prompt, recent_corpus_lines

load_dotenv()

POLL_SECONDS = 3
PHONE_RULES = (
    "\n\nPhone rules:\n"
    "- This is a phone call. Keep every turn under two sentences.\n"
    "- The call is capped at {minutes} minutes. Watch the time: if you are near the cap and still missing "
    "items from the checklist, ask for the most important missing one, then wrap up.\n"
    "- Never combine a question with the closing line. Ask, wait for the answer, and only then say exactly: "
    "\"I have enough. Thanks, you can hang up.\" Then stop talking.\n"
)


def speaker_role(speaker: str) -> str | None:
    if speaker == "agent":
        return "assistant"
    if speaker == "user":
        return "user"
    if speaker == "transfer_target":
        return None
    raise ValueError(f"unknown Dial speaker {speaker!r}")


async def run(to: str, from_number_id: str | None, corpus: Path | None, max_seconds: int, timeout: int) -> Path:
    api_key = os.environ.get("DIAL_API_KEY")
    if not api_key:
        raise RuntimeError("DIAL_API_KEY is not set in the environment or .env")
    dial = DialClient(DialConfig(api_key=api_key))

    if from_number_id is None:
        numbers = await dial.list_numbers()
        if not numbers:
            raise RuntimeError("the Dial account has no phone numbers; provision one in the dashboard or pass --from-number-id")
        from_number_id = numbers[0].id
        print(f"[dial] calling from {getattr(numbers[0], 'number', numbers[0].id)}")

    recent = recent_corpus_lines(corpus) if corpus else []
    instruction = build_system_prompt(author_name(), recent) + PHONE_RULES.format(minutes=max_seconds // 60)

    started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    call = await dial.make_call(
        MakeCallParams(
            to=to,
            from_number_id=from_number_id,
            outbound_instruction=instruction,
            language="en-US",
            max_call_duration_seconds=max_seconds,
            idempotency_key=run_id,
        )
    )
    print(f"[dial] call {call.id} placed to {to}. Answer your phone.")

    t0 = time.time()
    last_status = None
    while True:
        await asyncio.sleep(POLL_SECONDS)
        call = await dial.get_call(call.id)
        state = call.status.get("label") or call.status.get("state") if isinstance(call.status, dict) else str(call.status)
        if state != last_status:
            print(f"[dial] {state}")
            last_status = state
        if call.terminated_at and call.transcript_turns is not None:
            break
        if call.terminated_at and call.termination_type not in (None, "completed"):
            raise RuntimeError(f"call ended without completing: {call.termination_type} {call.failure_reason or ''}".strip())
        if time.time() - t0 > timeout:
            raise TimeoutError(f"call {call.id} did not finish with a transcript within {timeout}s (status={call.status})")

    turns: list[TranscriptTurn] = []
    for t in call.transcript_turns:
        role = speaker_role(t.speaker)
        if role is None or not t.text.strip():
            continue
        turns.append(TranscriptTurn(role=role, text=t.text.strip()))
    if not any(t.role == "user" for t in turns):
        raise RuntimeError(f"call {call.id} ended with no user turns; transcript: {call.transcript!r}")

    run_dir = Path("runs") / run_id
    out = run_dir / "transcript.json"
    write_json(out, Transcript(id=run_id, started_at=started_at, turns=turns))
    print(f"[dial] {len(turns)} turns, {call.duration}s on the phone")
    print(f"TRANSCRIPT WRITTEN: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", required=True, help="the founder's phone number, E.164, e.g. +14155550123")
    ap.add_argument("--from-number-id", default=None, help="Dial number id to call from (default: first number on the account)")
    ap.add_argument("--corpus", type=Path, default=None, help="corpus to inject recent posts from, e.g. data/corpus.jsonl")
    ap.add_argument("--max-seconds", type=int, default=280, help="hard cap on the call (free tier allows 300)")
    ap.add_argument("--timeout", type=int, default=480, help="give up waiting for the transcript after this many seconds")
    args = ap.parse_args()
    asyncio.run(run(args.to, args.from_number_id, args.corpus, args.max_seconds, args.timeout))


if __name__ == "__main__":
    main()
