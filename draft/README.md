# draft/

Transcript to outline to three drafts to a blind compare UI. Run everything from the repo root.

```sh
# 1. transcript.json -> outline.json (title, notes, keep_phrases)
uv run python -m draft.outline --run runs/fixture

# 2. outline.json -> drafts/{vanilla,fewshot,finetune}.md (prints keep-phrase survival per draft)
uv run python -m draft.generate --run runs/fixture --profile data/voice_profile.md --corpus data/corpus.jsonl --examples 25
#    subset: add --only vanilla --only fewshot (finetune shells out to train.sample)

# 3. judge the drafts -> runs/<id>/judge.json (voice 1-10, unsupported claims, 6-gram repeats, AI tells, kept phrases, words)
uv run python -m draft.judge --run runs/<id> --corpus data/corpus.jsonl --profile data/voice_profile.md --pairwise
#    --drafts vanilla fewshot finetune_e3   judge these names if present under drafts/
#    --include-real                         also judge runs/<id>/real.md as "real" (held-out posts)
#    --pairwise                             every pair of drafts, both orders: which one did the author write?
#    references: 15 corpus posts (blog, then tweet, then reply); a reference identical to a draft is dropped

# 4. compare UI: pick a run, three shuffled columns, Reveal button; shows judge.json captions when present
PYTHONPATH=. uv run streamlit run draft/compare.py
```

## Fixture

`runs/fixture/` holds a hand-written interview (`transcript.json`), a 200-word `voice_profile.md`, and a 10-record `corpus.jsonl`, so the pipeline runs before real data lands:

```sh
uv run python -m draft.outline --run runs/fixture
uv run python -m draft.generate --run runs/fixture --profile runs/fixture/voice_profile.md --corpus runs/fixture/corpus.jsonl --only vanilla --only fewshot
```

**Placeholder:** `runs/fixture/outline.json`, `vanilla.md` and `fewshot.md` are real Claude output from the commands above. `runs/fixture/drafts/finetune.md` is a hand-written placeholder (its first line says so) so the compare UI has three columns until `train.train` writes a sampler; running `draft.generate` without `--only` (or `--only finetune`) overwrites it via `train.sample`.
