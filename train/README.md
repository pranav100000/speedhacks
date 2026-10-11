# train/ — Tinker LoRA fine-tune and sampling

Both commands run from the repo root. They read `TINKER_API_KEY` and
`AUTHOR_NAME` from `.env` (loaded by `common.schema`).

## 1. Train

```
uv run python -m train.train --train data/train.jsonl --val data/val.jsonl
```

Defaults: `--model Qwen/Qwen3.5-9B-Base --rank 32 --epochs 3 --batch-size 16
--lr 2e-4 --max-length 4096 --eval-every 10 --log-path runs/tinker/<YYYYmmdd-HHMMSS>
--sampler-out data/sampler.json`.

What it does: validates every record against `common.schema.TrainRecord`,
builds the cookbook `supervised.train.Config` the way `recipes/sl_basic.py`
does (renderer from `get_recommended_renderer_name`, `ALL_ASSISTANT_MESSAGES`),
runs `train.main`, then reads the final checkpoint's `sampler_path` from
`<log-path>/checkpoints.jsonl` and writes `data/sampler.json`:

```
{"model": ..., "renderer": ..., "sampler_path": "tinker://...", "state_path": "tinker://...", "log_path": ..., ...}
```

Logs land in `<log-path>/`: `metrics.jsonl` (train NLL, `test/nll`, `test/bpb`
on val every `--eval-every` steps), `checkpoints.jsonl`, `config.json`, and
`logs.log`. The sampler path is also printed in a banner at the end.

The log path must not already exist. If a run dies before training starts
(for example on auth), the cookbook has already created the dir; `rm -rf` it or
pass a fresh `--log-path` before rerunning.

Expected duration (not yet measured, the API key was rejected during the build):
the 128-record fixture is 16 steps and should be a few minutes; the real run
(~200 records x 3 epochs at batch 16, ~36 steps) is on the order of 5-15
minutes plus Tinker queue time.

## 2. Sample

```
uv run python -m train.sample --outline runs/<id>/outline.json --out runs/<id>/drafts/finetune.md
```

Defaults: `--sampler data/sampler.json --max-tokens 1500 --temperature 0.8 --n 1`.

Builds the system + user turns with `common.schema.build_messages` (the same
function `synth/make_train.py` uses), renders the generation prompt with the
same cookbook renderer, samples from `sampler_path`, stops at the renderer's
end-of-turn sequence, decodes with `renderer.parse_response`, writes the
stripped text to `--out` (creating parent dirs) and prints it to stdout.
Diagnostics (prompt tokens, stop reason, termination kind) go to stderr.
With `--n 3` the extra samples land in `finetune.2.md`, `finetune.3.md`.

Expected duration: tens of seconds for a 1500-token draft (not yet measured).

## Fixture smoke test (no real data needed)

```
uv run python -m train.train \
  --train .venv/lib/python3.12/site-packages/tinker_cookbook/example_data/conversations.jsonl \
  --val   .venv/lib/python3.12/site-packages/tinker_cookbook/example_data/conversations.jsonl \
  --epochs 1 --batch-size 8 --max-length 512 --skip-schema-check \
  --log-path runs/tinker/fixture_c --sampler-out data/sampler_fixture.json
uv run python -m train.sample --outline runs/fixture_c/outline.json \
  --out runs/fixture_c/drafts/finetune.md --sampler data/sampler_fixture.json
```

`--skip-schema-check` exists only because the cookbook fixture has no system
turn; never pass it for `data/train.jsonl`. `runs/fixture_c/outline.json` is
checked in as a sample Outline.

## API notes (installed tinker 0.33.1 / tinker-cookbook 0.5.7)

- `Qwen/Qwen3.5-9B-Base` maps to the `role_colon` renderer. The prompt is
  `System: ...\n\nUser: ...\n\nAssistant:` and the stop sequence is `"\n\nUser:"`.
  Instruct variants map to `qwen3_5`.
- `save_weights_and_get_sampling_client()` is now ephemeral (its `name` arg is
  deprecated). `train.main` already saves a persistent final checkpoint
  (`save_weights_for_sampler` + `save_state`, no TTL) and records it in
  `checkpoints.jsonl`; `train.py` uses that instead of saving twice.
- `SamplingClient.sample(prompt, num_samples, sampling_params).result()`;
  `SamplingParams(max_tokens, temperature, stop, seed, top_k, top_p)`.
- `FromConversationFileBuilder` has only `test_size`; the explicit val file
  is wired through `TrainValConversationFileBuilder` in `train.py`, and
  `train.main` runs an `NLLEvaluator` on it every `--eval-every` steps.
