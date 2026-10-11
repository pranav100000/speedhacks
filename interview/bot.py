"""Pipecat voice interviewer. Workstream D in SCOPE.md.

Interviews the author to collect raw material for one blog post, then writes
runs/<id>/transcript.json (common.schema.Transcript) when the client hangs up.

Run from the repo root::

    uv run python -m interview.bot [--corpus data/corpus.jsonl] [--runs-dir runs]

Client: http://localhost:7860/client
"""

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.runner.run import main as runner_main
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from common.schema import (
    CorpusRecord,
    Transcript,
    TranscriptTurn,
    author_name,
    read_jsonl,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
RECENT_CORPUS_COUNT = 10
RECENT_SNIPPET_CHARS = 80
CARTESIA_VOICE = "71a7ad14-091c-4e8e-a314-022ece01c121"  # quickstart's British Reading Lady
HARD_CAP_EXCHANGES = 16

# Newest general model in the installed Deepgram SDK (7.12.0). Keyterm prompting is a
# Nova-3 feature; `keywords` is the older Nova-2 equivalent and is not used here.
DEEPGRAM_MODEL = "nova-3-general"
FIXED_KEYTERMS = [
    "Aether",
    "Firecracker",
    "Codex",
    "Claude Code",
    "worktree",
    "worktrees",
    "MCP",
    "Sentry",
    "Linear",
    "devbox",
    "Convex",
    "Tinker",
    "LoRA",
    "HeyGen",
    "BeatGen",
    "YC",
]
MAX_KEYTERMS = 50
MIN_KEYTERM_COUNT = 2  # a corpus word must appear this many times to become a keyterm
CAPITALIZED_WORD = re.compile(r"\b[A-Z][A-Za-z0-9]+\b")
HANDLE = re.compile(r"@([A-Za-z0-9_]+)")


# ---------- config ----------


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set in the environment or .env")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voice interviewer (Pipecat)")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=os.environ.get("INTERVIEW_CORPUS") or None,
        help="data/corpus.jsonl; when given, the 10 most recent records are shown to the interviewer. "
        "Defaults to $INTERVIEW_CORPUS.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Where runs/<id>/transcript.json is written (default: {DEFAULT_RUNS_DIR})",
    )
    return parser


# ---------- system prompt ----------


def recent_corpus_lines(corpus_path: Path, count: int = RECENT_CORPUS_COUNT) -> list[str]:
    """Titles, or the first 80 characters, of the `count` most recent corpus records."""
    records = read_jsonl(corpus_path, CorpusRecord)
    if not records:
        raise ValueError(f"{corpus_path} has no records")
    # created_at is ISO 8601 per the contract; parse so a bad timestamp fails here, not silently.
    records.sort(key=lambda r: datetime.fromisoformat(r.created_at), reverse=True)
    lines: list[str] = []
    for record in records[:count]:
        snippet = record.title if record.title else record.text[:RECENT_SNIPPET_CHARS]
        snippet = " ".join(snippet.split())
        lines.append(f"- ({record.format}) {snippet}")
    return lines


def corpus_keyterms(corpus_path: Path, minimum_count: int = MIN_KEYTERM_COUNT) -> list[str]:
    """Capitalized words and @handles (without the @) that appear at least `minimum_count`
    times across the corpus, most frequent first."""
    records = read_jsonl(corpus_path, CorpusRecord)
    if not records:
        raise ValueError(f"{corpus_path} has no records")
    counts: Counter[str] = Counter()
    for record in records:
        text = record.text if record.title is None else f"{record.title}\n{record.text}"
        counts.update(HANDLE.findall(text))
        counts.update(CAPITALIZED_WORD.findall(HANDLE.sub(" ", text)))
    frequent = [(term, count) for term, count in counts.items() if count >= minimum_count]
    frequent.sort(key=lambda item: (-item[1], item[0]))
    return [term for term, _ in frequent]


def build_keyterms(corpus_path: Path | None) -> list[str]:
    """Fixed product and tool names first, then corpus-derived terms, deduplicated, capped."""
    terms: list[str] = list(FIXED_KEYTERMS)
    if corpus_path:
        for term in corpus_keyterms(corpus_path):
            if term not in terms:
                terms.append(term)
    return terms[:MAX_KEYTERMS]


def build_system_prompt(name: str, recent: list[str]) -> str:
    prompt = f"""You are interviewing {name} to get the raw material for ONE blog post. You are not writing the post. You are getting the stories, numbers and opinions out of them.

How to run the interview:
- Ask one short question at a time. Never two questions in one turn.
- Open by asking what they want to write about.
- Your second question, right after they name the topic, is "Who is this post for?" The audience shapes everything that follows.
- Then dig, in roughly this order: a specific story (what happened, who was involved, when it was); a number; what they believe that most people do not; what they got wrong and what replaced it; what they would tell someone two years behind them.
- For every one of those topics, ask at least one follow-up before moving on. Pick the follow-up from: "give me an example", "when was that", "who was involved", "how many, how long, or how much", "what did that look like on your screen", "what did you do the next day".
- Mirror only a short fragment of their exact words, three to six words, then ask the next question. Never repeat a whole sentence back. Never rephrase what they said into tidier language. If they say "it blew up", you say "it blew up", not "it was successful". Example: they say "my computer kept running out of RAM and crashing every day" and you say "Crashing every day. How many agents were you running when that happened?"
- Keep asking until you have all of this: who the post is for; a story with a when and a who; two concrete numbers; one belief most people disagree with; one mistake and what replaced it; one piece of advice. Before you stop, check that list. If anything is missing, ask for it.
- Hard cap: {HARD_CAP_EXCHANGES} questions from you. When the list is complete, or at the cap, say "I have enough" and that they can hang up. Do not summarize what they said when you stop.

Voice rules:
- This is a voice call. Speak in short sentences. No lists, no markdown, no headings.
- Do not summarize. Do not give advice. Do not compliment their answers.
- Do not say "great question" or "that's interesting". Just ask the next question."""
    if recent:
        joined = "\n".join(recent)
        prompt += f"""

Things {name} has written recently (most recent first). If what they want to write about is related to one of these, say so in their words, for example "you posted about X, is this related?":
{joined}"""
    return prompt


# ---------- transcript ----------


@dataclass
class Session:
    run_id: str | None = None
    started_at: str | None = None
    turns: list[TranscriptTurn] = field(default_factory=list)


def new_run_id(now: datetime) -> str:
    return now.strftime("run_%Y%m%d_%H%M%S")


def write_transcript(
    runs_dir: Path, run_id: str, started_at: str, turns: list[TranscriptTurn]
) -> Path:
    """Write runs/<id>/transcript.json. Raises if the user never said anything."""
    if not any(turn.role == "user" for turn in turns):
        raise RuntimeError(
            f"{run_id}: no user turns were captured ({len(turns)} turns total); refusing to write transcript"
        )
    transcript = Transcript(id=run_id, started_at=started_at, turns=turns)
    path = runs_dir / run_id / "transcript.json"
    write_json(path, transcript)
    banner = "=" * 72
    print(f"\n{banner}\nTRANSCRIPT WRITTEN: {path}\n{len(turns)} turns\n{banner}\n", flush=True)
    logger.info(f"Transcript written: {path}")
    return path


# ---------- bot ----------


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    *,
    corpus_path: Path | None,
    runs_dir: Path,
) -> None:
    name = author_name()
    recent = recent_corpus_lines(corpus_path) if corpus_path else []
    if corpus_path:
        logger.info(f"Loaded {len(recent)} recent records from {corpus_path}")
    else:
        logger.info("No corpus given; interviewer has no recent-post context")

    keyterms = build_keyterms(corpus_path)
    logger.info(f"Deepgram {DEEPGRAM_MODEL} with {len(keyterms)} keyterms: {keyterms}")
    stt = DeepgramSTTService(
        api_key=required_env("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(model=DEEPGRAM_MODEL, keyterm=keyterms),
    )
    tts = CartesiaTTSService(
        api_key=required_env("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(voice=CARTESIA_VOICE),
    )
    llm = OpenAILLMService(
        api_key=required_env("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(system_instruction=build_system_prompt(name, recent)),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1.0))),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=False, enable_metrics=True, enable_usage_metrics=True),
    )

    session = Session()

    # Pipecat 1.12 has no standalone TranscriptProcessor; the context aggregators
    # emit the finalized user and assistant turns as events instead.
    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn(aggregator, message: UserTurnMessageAddedMessage):
        text = message.content.strip()
        if not text:
            logger.warning("Empty user turn; not recording")
            return
        session.turns.append(TranscriptTurn(role="user", text=text))
        logger.info(f"[user] {text}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn(aggregator, message: AssistantTurnStoppedMessage):
        text = message.content.strip()
        if not text:
            logger.warning("Empty assistant turn (interrupted before any tokens); not recording")
            return
        session.turns.append(TranscriptTurn(role="assistant", text=text))
        logger.info(f"[assistant{' (interrupted)' if message.interrupted else ''}] {text}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        now = datetime.now().astimezone()
        session.run_id = new_run_id(now)
        session.started_at = now.isoformat()
        session.turns.clear()
        logger.info(f"Client connected; run id {session.run_id}")
        context.add_message(
            {
                "role": "developer",
                "content": f"The call just started. Greet {name} in one short sentence and ask what they want to write about.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if session.run_id is None or session.started_at is None:
            raise RuntimeError("Client disconnected before on_client_connected set a run id")
        try:
            await task.cancel()
        finally:
            write_transcript(runs_dir, session.run_id, session.started_at, session.turns)

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point discovered by the pipecat dev runner."""
    if runner_args.cli_args is None:
        raise RuntimeError("runner_args.cli_args is None; run via `uv run python -m interview.bot`")
    cli = runner_args.cli_args
    corpus_path: Path | None = cli.corpus
    runs_dir: Path = cli.runs_dir

    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args, corpus_path=corpus_path, runs_dir=runs_dir)


def main() -> None:
    load_dotenv(override=True)
    runner_main(build_parser())


if __name__ == "__main__":
    main()
