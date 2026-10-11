# interview — voice interviewer (Pipecat)

Talks to the author for a few minutes and writes the transcript that `draft/outline.py` consumes.

## Env keys (in `.env` at the repo root)

```
AUTHOR_NAME=...          # who is being interviewed (already required by common/schema.py)
DEEPGRAM_API_KEY=...     # speech to text
CARTESIA_API_KEY=...     # text to speech
OPENAI_API_KEY=...       # the interviewer LLM (quickstart default, gpt-4.1)
```

The bot raises at connect time if any of these is missing. Nothing is read at import.

## Run

From the repo root:

```
uv run python -m interview.bot
uv run python -m interview.bot --corpus data/corpus.jsonl      # interviewer sees your 10 most recent posts
INTERVIEW_CORPUS=data/corpus.jsonl uv run python -m interview.bot  # same thing via env
```

Optional: `--runs-dir <path>` (default `runs/`).

Then open **http://localhost:7860/client** in a browser, allow the microphone, and press Connect. The interviewer speaks first and asks what you want to write about.

## How the interview runs

The interviewer asks what you want to write about, then who the post is for, then digs through a story, a number, a belief, a mistake and a piece of advice, with at least one follow-up on each. It mirrors only a 3-6 word fragment of what you said, never a whole sentence.

It stops on a content checklist, not a count: the audience, a story with a when and a who, two concrete numbers, one belief most people disagree with, one mistake and what replaced it, and one piece of advice. Hard cap is 16 questions (`HARD_CAP_EXCHANGES` in `bot.py`).

## Speech to text

Deepgram `nova-3-general` with keyterm prompting. The keyterm list is a fixed set of product and tool names (`FIXED_KEYTERMS`) plus every capitalized word and `@handle` (without the `@`) that appears at least twice in the `--corpus` file, capped at 50. The list is logged at startup.

## Ending the call

When the checklist is complete (or at the cap) the interviewer says "I have enough" and that you can hang up. Press **Disconnect** in the client page (or close the tab). On disconnect the bot writes

```
runs/run_<YYYYmmdd_HHMMSS>/transcript.json
```

and prints the path in a banner. It raises if no user turn was captured. Stop the server with Ctrl+C.

## Notes

- Transport: SmallWebRTC via the pipecat dev runner (`-t webrtc` is the only transport wired up; the `daily` extra is not installed).
- Pipecat 1.12 has no `TranscriptProcessor`; turns come from the context aggregators' `on_user_turn_message_added` and `on_assistant_turn_stopped` events.
