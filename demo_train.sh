#!/bin/zsh
# Stage version of "approve this post and train on it": quiet output, key lines only.
# usage: ./demo_train.sh runs/<id> [finetune|fewshot]
cd "$(dirname "$0")"
RUN=$1; FROM=${2:-finetune}
if [ "$FROM" = "text" ]; then SRC=(--text "$RUN/final.md"); else SRC=(--from "$FROM"); fi
uv run python -u -m synth.approve --run "$RUN" "${SRC[@]}" 2>&1 \
  | grep --line-buffered -E "^\[approve\]|train_mean_nll|test/nll|TRAINING DONE|SAMPLER PATH|Traceback|Error" \
  | sed -u -E 's/^│ *train_mean_nll *│ *([0-9.]+) *│/   step loss \1/; s/.*"test\/nll": ([0-9.]+).*/   val loss \1/'
