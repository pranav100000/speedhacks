#!/bin/zsh
# Stage version of "sample from the model we just trained": one short paragraph, no logs.
cd "$(dirname "$0")"
RUN=$1
uv run python -m train.sample --outline "$RUN/outline.json" --out "$RUN/drafts/finetune_after_update.md" --max-tokens 160 --temperature 0.6 2>&1 | grep -v -E "httpx|DEBUG|^\[sample"
