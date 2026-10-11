# Voice-to-post, in your own voice — 4-hour build scope

**Demo:** talk to an interviewer for ~3 minutes about a topic → see a blog post that reads like you, side by side with vanilla Claude and few-shot Claude. Judges guess which column is "really you".

**Constraints:** 4 hours, one builder plus Aether agents, Python everywhere, one user (you), corpus built from export files already on disk. No live integrations, no OAuth.

**Hard gate:** the real Tinker training run is launched by T+1:00. Everything before that gate serves that launch.

---

## 1. Pipeline

```
exports on disk ──▶ ingest ──▶ data/corpus.jsonl
                                     │
                                     ▼
                              synth (Claude) ──▶ data/pairs.jsonl ──▶ data/train.jsonl + data/val.jsonl
                                     │                                        │
                                     ▼                                        ▼
                              data/voice_profile.md                    train (Tinker LoRA) ──▶ sampler checkpoint

mic ──▶ interview bot (Pipecat) ──▶ runs/<id>/transcript.json
                                          │
                                          ▼
                                   outline (Claude) ──▶ runs/<id>/outline.json
                                          │
                 ┌────────────────────────┼─────────────────────────┐
                 ▼                        ▼                         ▼
        vanilla Claude          few-shot Claude + profile     fine-tuned model
                 └────────────────────────┼─────────────────────────┘
                                          ▼
                                 compare UI (Streamlit)
```

**Key design decision:** the fine-tune is trained on `(notes → real post)` pairs synthesized backwards from the corpus, and the drafter produces *notes*, never prose. The small model is the only thing that ever writes sentences in the "you" column. Spoken phrases from the interview are carried through verbatim as "phrases to keep".

---

## 2. Repo layout

```
speedhacks/
  pyproject.toml            # uv; deps: anthropic, tinker, tinker-cookbook, pipecat-ai, streamlit, trafilatura
  .env                      # keys, gitignored
  common/
    schema.py               # pydantic models for every contract below
    claude.py               # one Anthropic client factory (claude-opus-5-5, streaming helper)
  ingest/
    x.py  linkedin.py  gmail.py  blog.py
    run.py                  # runs all parsers → data/corpus.jsonl
  synth/
    make_pairs.py           # corpus → pairs.jsonl
    make_train.py           # pairs → train.jsonl / val.jsonl
    voice_profile.py        # corpus → voice_profile.md
  train/
    train.py                # Tinker SFT run
    sample.py               # load sampler checkpoint, generate from an outline
  interview/
    bot.py                  # Pipecat interviewer; writes transcript.json on disconnect
  draft/
    outline.py              # transcript → outline.json
    generate.py             # outline → three drafts
    compare.py              # Streamlit three-column view
  data/                     # gitignored
  runs/                     # gitignored
```

**Repo rules (apply to every agent):** no fallbacks, fail loudly; imports at the top of the file; every script is a CLI with explicit input and output paths; write outputs atomically.

---

## 3. Contracts

All JSONL. Every agent codes against these, not against each other.

### `data/corpus.jsonl` — one record per piece of writing
```json
{"id": "x_1234", "source": "x", "format": "tweet", "text": "...", "created_at": "2025-03-01T12:00:00Z", "title": null}
```
- `source`: `x | linkedin | gmail | blog`
- `format`: `tweet | thread | reply | linkedin_post | email | blog`
- `text`: cleaned plain text. No URLs, no quoted replies, no signatures.
- Filters are explicit and logged with counts: retweets dropped, replies tagged `reply`, emails under 40 words dropped.

### `data/pairs.jsonl` — one record per training example
```json
{"id": "x_1234", "format": "blog", "notes": "- bullet\n- bullet", "keep_phrases": ["exact phrase from text"], "text": "...the real post..."}
```
- `notes`: 2 to 8 terse first-person bullets, the way someone talks in an interview. Not prose. No topic sentences.
- `keep_phrases`: 1 to 4 short verbatim spans copied from `text`.

### `data/train.jsonl`, `data/val.jsonl` — Tinker conversation format
```json
{"messages": [
  {"role": "system", "content": "You are <NAME>. Write in your own voice."},
  {"role": "user", "content": "Format: blog\n\nNotes:\n- ...\n\nPhrases to keep verbatim:\n- ...\n"},
  {"role": "assistant", "content": "...the real post..."}
]}
```
The user message is built by **one function** in `common/schema.py` (`render_user_message(format, notes, keep_phrases)`) and that same function is used at inference. Train/serve skew is the easiest way to lose the demo.

### `runs/<id>/transcript.json`
```json
{"id": "run_001", "started_at": "...", "turns": [{"role": "assistant", "text": "..."}, {"role": "user", "text": "..."}]}
```

### `runs/<id>/outline.json`
```json
{"title": "...", "format": "blog", "notes": "- ...", "keep_phrases": ["..."]}
```
Same shape as a `pairs.jsonl` record minus `text`. `keep_phrases` are the user's own spoken words, lifted from `user` turns only.

### `runs/<id>/drafts/{vanilla,fewshot,finetune}.md`

---

## 4. Workstreams and Aether briefs

Launch A, C, D, E at T+0:20. B starts on a 20-record fixture and switches to the real corpus the moment A lands. Each brief is self-contained. Each has an acceptance test; the agent is not done until it passes.

### A. Ingest (time box 40 min; Gmail is the first thing cut)

> Build `ingest/` producing `data/corpus.jsonl` per the contract in SCOPE.md §3.
> - `x.py`: parse the X archive. Input is `data/tweets.js` (or `tweets-part*.js`); strip the `window.YTD.tweets.part0 =` prefix then JSON-parse. Use `tweet.full_text`, `tweet.created_at`, `tweet.in_reply_to_status_id`, `tweet.id_str`. Drop retweets (text starts with `RT @`). Tag replies as `reply`. Strip `t.co` URLs. Reconstruct threads (consecutive self-replies) into one `thread` record **only if it takes under 15 minutes**, else tag them `reply`.
> - `linkedin.py`: parse `Shares.csv` from the LinkedIn export. `ShareCommentary` is the text, `Date` the timestamp. Format `linkedin_post`.
> - `gmail.py`: stream the Takeout `.mbox` with `mailbox.mbox`. Keep messages where `From` contains the user's address and `X-Gmail-Labels` contains `Sent`. Take the `text/plain` part. Cut quoted replies (from the first line matching `On .* wrote:` or starting with `>`) and signatures (from `-- ` or the name line). Drop under 40 words. Format `email`.
> - `blog.py`: accept a directory of `.md` files or a text file of URLs. For URLs use `trafilatura` to extract the body. Format `blog`, keep `title`.
> - `run.py`: run every parser, write `data/corpus.jsonl`, print a count table by source and format.
> Parse errors raise. Documented filters log counts. No silent skips.
> **Acceptance:** `uv run ingest/run.py` prints the count table and every record validates against `common/schema.py`.

### B. Pairs, train set, voice profile (time box 45 min after corpus lands)

> Build `synth/`. Use `common/claude.py` (Claude Opus 5.5, structured outputs, concurrency 10 via asyncio).
> - `make_pairs.py`: for every corpus record with format in `blog | linkedin_post | thread | tweet`, ask Claude for `notes` and `keep_phrases` per the `pairs.jsonl` contract. Prompt: "Reverse-engineer the terse spoken notes someone would have given an interviewer before writing this. First person, fragments, no polish, 2 to 8 bullets. Then copy 1 to 4 short exact phrases from the text." Validate that each `keep_phrases` entry is a substring of `text`; raise if not.
> - `make_train.py`: cap tweets at 300 (most recent), keep all blogs, linkedin posts and threads. Stratified 90/10 split by format into `train.jsonl` / `val.jsonl` in Tinker conversation format, using `render_user_message` from `common/schema.py`. Print counts per format per split.
> - `voice_profile.py`: give Claude 40 posts (all blogs plus a mix) and ask for `data/voice_profile.md`: sentence length habits, how posts open and close, recurring words and punctuation, things this writer never does, and a blacklist of AI tells to avoid. Under 400 words.
> **Acceptance:** `train.jsonl` has at least 150 records, every record round-trips through `common/schema.py`, and `val.jsonl` has at least 15.

### C. Tinker training and sampling (develop on fixtures from T+0:20, real launch at T+1:00)

> Build `train/train.py` and `train/sample.py` on the Tinker API with `tinker-cookbook`.
> - `train.py`: start from `tinker_cookbook/recipes/sl_basic.py`. Dataset via `FromConversationFileBuilder(file_path="data/train.jsonl")`. Base model: the cookbook default (`Qwen/Qwen3.5-9B-Base`), rank 32. Reason for a base model: an instruct model carries the exact assistant voice we are removing. Settings: `batch_size=16`, `num_epochs=3`, `learning_rate=2e-4`, `train_on_what=ALL_ASSISTANT_MESSAGES`, renderer from `get_recommended_renderer_name()`. Log path under `runs/tinker/`. At the end call `save_weights_and_get_sampling_client()` and write the sampler identifier to `data/sampler.txt`.
> - `sample.py`: CLI taking `--outline runs/<id>/outline.json`, builds the same messages as training (system + `render_user_message`), renders the generation prompt with the same renderer, samples with `max_tokens=1500`, `temperature=0.8`, decodes, writes markdown to stdout or `--out`. Must stop at the renderer's end-of-turn token.
> - Verify on `tinker_cookbook/example_data/conversations.jsonl` first, so the real launch at T+1:00 is a one-liner.
> **Acceptance:** a training run completes end to end on the fixture, and `sample.py` prints coherent text from the resulting sampler. Then: real run launched, run ID written down, and `sample.py` on 5 `val.jsonl` outlines looks like the author, not like an assistant.
> If base-model outputs ramble or never stop, switch to the instruct variant of the same model and rerun once. That is the only retry budgeted.

### D. Voice interviewer (time box 45 min)

> Build `interview/bot.py` on Pipecat. Start from `pipecat init quickstart` (Deepgram STT, Cartesia TTS, the quickstart's LLM service; swapping the LLM to Claude is a later 5-minute change, not on the critical path). Web client at `http://localhost:7860/client`.
> - System prompt: you are interviewing <NAME> to get the raw material for one blog post. One question at a time, short. Open by asking what they want to write about. Then dig: a specific story, a number, what they believe that most people don't, what they got wrong, what they would tell someone two years behind them. Mirror their words back, never rephrase them into tidier language. After about 8 exchanges say you have enough and that they can hang up.
> - Inject the titles or first lines of the 10 most recent corpus posts into the system prompt so the interviewer can say "you posted about X, is this related?"
> - Accumulate the transcript with Pipecat's transcript processor and write `runs/<id>/transcript.json` on client disconnect. Print the path.
> **Acceptance:** a 2-minute conversation produces a valid `transcript.json` with both roles.

### E. Drafter and compare UI (time box 60 min, start on a hand-written fixture transcript)

> Build `draft/`. Claude Opus 5.5 via `common/claude.py`, adaptive thinking, streaming.
> - `outline.py`: transcript → `outline.json`. Structured output. `notes` are terse first-person bullets. `keep_phrases` are 5 to 10 exact spans from the user's turns only; validate substring membership against the transcript and raise if not.
> - `generate.py`: writes three drafts to `runs/<id>/drafts/`:
>   - `vanilla.md`: Claude, "write a blog post from these notes".
>   - `fewshot.md`: Claude with `data/voice_profile.md` plus 25 corpus examples (all blogs first, then linkedin and threads) plus the notes, instructed to keep every `keep_phrase` verbatim.
>   - `finetune.md`: shell out to `train/sample.py`.
> - `compare.py`: Streamlit page. Pick a run, show the three drafts in columns with labels hidden and order shuffled, a "reveal" button, and a count of `keep_phrases` that survived verbatim in each draft.
> **Acceptance:** `uv run draft/outline.py --run fixture && uv run draft/generate.py --run fixture` produces three files and `streamlit run draft/compare.py` shows them.

### F. Demo prep (you, T+3:15)

Pick a topic you actually have opinions on. Rehearse the interview twice. Write down the two sentences you will say before pressing play.

---

## 5. Timeline

| Clock | What is happening |
|---|---|
| 0:00 to 0:20 | uv project, keys in `.env`, `common/schema.py` and `common/claude.py` written, this file committed. Launch Aether agents A, C, D, E. |
| 0:20 to 1:00 | A delivers corpus. B runs pairs synthesis. C passes the fixture run. D has a bot that talks. |
| **1:00** | **Launch real training. Hard gate.** |
| 1:00 to 2:00 | E builds outline, generate, compare on the fixture. B finishes voice profile. D tunes the interviewer prompt. |
| 2:00 to 2:30 | Training done. Sample 5 held-out outlines. One retry allowed. |
| 2:30 to 3:15 | End to end: interview → outline → three drafts → compare. Fix glue. |
| 3:15 to 3:45 | Rehearse twice. Freeze the code. |
| 3:45 to 4:00 | Buffer. |

**Cut order if behind:** Gmail parser, thread reconstruction, corpus-aware interviewer, blind shuffle in the UI.

---

## 6. Keys needed before T+0:20

`TINKER_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, plus whatever key the Pipecat quickstart's LLM service needs. If one is missing, D is the only workstream that changes: swap the missing service in Pipecat's service list.

## 7. Risks worth naming

- **Tinker queue time.** Unknown. That is why the gate is at 1:00 and not later.
- **Train/serve skew.** One `render_user_message` function, used in both places. Non-negotiable.
- **Gmail mbox size.** Multi-GB files are common. Stream, never load. First cut.
- **Keep-phrases not surviving.** The compare UI counts them so the demo shows the difference explicitly.
