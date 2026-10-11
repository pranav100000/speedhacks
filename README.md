# LoRAkeet: it calls you, then it writes like you

Talk to an interviewer for three minutes. It turns the conversation into terse notes, then three writers draft the post from the same notes: vanilla Claude, Claude with your voice profile and real posts in context, and a LoRA fine-tuned on your writing with [Tinker](https://thinkingmachines.ai/tinker). The compare page shows the three side by side with labels hidden and counts how many of your spoken phrases each one kept verbatim.

Design notes and the build plan are in `SCOPE.md`.

## Run the demo

Keys live in `.env` (see `.env.example`). Everything runs from the repo root with `uv run`.

```bash
# 1. interviewer (http://localhost:7860/client), hang up when it says it has enough
uv run python -m interview.bot --corpus data/corpus.jsonl

# 2. notes + three drafts for the run it just wrote
#    outline: fixes speech-to-text errors (diff-guarded), picks 0-2 quotable lines (opinions, jokes, verdicts)
#    generate: samples 4 fine-tune candidates at temperature 0.6 and 2 few-shot candidates,
#              keeps the one with the fewest claims the speaker never made
uv run python -m draft.outline  --run runs/run_<timestamp>
uv run python -m draft.generate --run runs/run_<timestamp> --profile data/voice_profile.md --corpus data/corpus.jsonl

# 2b. score the drafts: voice likeness vs your real posts, unsupported claims, repeats, AI tells
uv run python -m draft.judge --run runs/run_<timestamp> --corpus data/corpus.jsonl --profile data/voice_profile.md --pairwise

# 2c. approve the post you edited and teach the model from it (warm start, ~1 minute)
uv run python -m synth.approve --run runs/run_<timestamp> --text runs/run_<timestamp>/final.md

# 3. compare page (http://localhost:8501)
PYTHONPATH=. uv run streamlit run draft/compare.py
```

## Rebuild the voice model from your exports

```bash
uv run python -m ingest.run --x data/x --blog data/blog --out data/corpus.jsonl      # add --linkedin / --gmail when you have them
uv run python -m synth.make_pairs   --corpus data/corpus.jsonl --out data/pairs.jsonl      # also makes note variants + section pairs for long posts
uv run python -m synth.make_train   --pairs data/pairs.jsonl --corpus data/corpus.jsonl --train data/train.jsonl --val data/val.jsonl \
    --repeat blog=3,section=2 --keep-in-train blog,section                                 # long-form upweighted, never held out
uv run python -m synth.voice_profile --corpus data/corpus.jsonl --out data/voice_profile.md
uv run python -m synth.screen_emails --corpus data/corpus.jsonl --out data/corpus_screened.jsonl --decisions data/email_screen.jsonl   # if you ingested mail
uv run python -m train.train --train data/train.jsonl --val data/val.jsonl --model Qwen/Qwen3.5-35B-A3B-Base --epochs 2 --batch-size 4 --lr 1e-4   # writes data/sampler.json
```

Training 75 records for 2 epochs takes about two minutes on Tinker. On a corpus this size, 3 epochs at 2e-4 overfits
(validation loss rises after step ~15) and roughly triples invented claims; 2 epochs at 1e-4 held flat. Each directory
has its own README with flags and caveats.

## What moved the needle (measured with draft/judge.py on the same interview)

| Change | Effect |
|---|---|
| Cap keep-phrases at 5 and skip spoken stutters | fine-tune stopped repeating itself; few-shot stopped echoing transcript grammar |
| Make echoed lines soft (0-2, opinions only, grammar may be fixed; model trained with a third of records carrying none) | drafts stopped reading as stitched transcript; few-shot voice 7 -> 8, invented claims 0-1 on both columns |
| Few-shot rule: facts only from notes, phrases, speaker's words | unsupported claims 14 -> 2 at the same voice score |
| Relabel scratch notes out of "blog", add section pairs, upweight long-form x3 | fine-tune writes sentence-case long-form instead of lowercase tweet fragments |
| Sample at 0.6 instead of 0.8 | fine-tune unsupported claims 6 -> 0 to 2 per sample |
| Pick candidates by fewest unsupported claims | turns the judge into a filter; costs one Claude call per candidate |
| Four recent substantive emails added (screened for privacy and substance; 700 of 711 sent emails were logistics, private, or school essays) | 9B pairwise wins 2 -> 6 on the same outline |
| Qwen3.5-35B-A3B-Base instead of 9B, same data | fine-tune voice 6 -> 7, first time it matched few-shot; samples about as fast |
| Soft echoed lines (0-2, opinions only) instead of 3-5 verbatim phrases | drafts stopped reading as stitched transcript; pairwise wins for soft 8 of 8 vs hard 3 once the judge was told not to reward spoken phrasing |
| 45 synthetic posts distilled from the few-shot writer (synth/distill.py), grounded in real records, claims-filtered | no measurable change on the judge (voice 6, 0 claims, same pairwise); kept as an experiment, not the demo model |

## How it fits together

`common/schema.py` holds every contract and the one function that renders the model's input. Training and sampling both call it, so the fine-tune never sees a prompt shape it wasn't trained on. The drafter emits notes and at most two quotable lines, never prose, so the only thing that writes sentences in the "you" column is the model trained on you. The lines are soft: the writer keeps their meaning and key words and may fix spoken grammar, and survival is scored by fuzzy match.
