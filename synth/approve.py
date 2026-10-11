"""Approve a post and teach the model from it.

Takes the run's outline (notes + echoed lines) and the post you approved, appends that pair to
data/pairs.jsonl, builds a small update set (the new pair upweighted plus a replay sample of
existing pairs), and warm-starts a short training run from the current sampler's checkpoint.
The new sampler replaces data/sampler.json; the previous one is kept as data/sampler_prev.json.

  uv run python -m synth.approve --run runs/<id> --from finetune            # or --from fewshot / --text path.md
  uv run python -m synth.approve --run runs/<id> --from finetune --no-train # just record the pair
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common.schema import Outline, PairRecord, TrainRecord, build_messages, read_jsonl, write_jsonl
from common.text import survived

NEW_PAIR_COPIES = 4
REPLAY = 28
LONG_FORM = ("blog", "section", "email")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs/<id> with outline.json and drafts/")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="from_draft", choices=("finetune", "fewshot", "vanilla"), help="approve this draft as-is")
    src.add_argument("--text", help="path to the final, edited post")
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--sampler", default="data/sampler.json")
    ap.add_argument("--steps", type=int, default=8, help="warm-start steps at batch 4")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-train", action="store_true", help="record the pair, skip training")
    args = ap.parse_args()

    run_dir = Path(args.run)
    outline = Outline.model_validate_json((run_dir / "outline.json").read_text())
    text_path = Path(args.text) if args.text else run_dir / "drafts" / f"{args.from_draft}.md"
    text = text_path.read_text().strip()
    if not text:
        raise ValueError(f"{text_path} is empty")

    echoed = [p for p in outline.keep_phrases if survived(text, [p])]
    pair = PairRecord(id=f"approved_{run_dir.name}", format=outline.format, notes=outline.notes, keep_phrases=echoed, text=text)
    pairs = read_jsonl(args.pairs, PairRecord)
    pairs = [p for p in pairs if p.id != pair.id] + [pair]
    write_jsonl(args.pairs, pairs)
    print(f"[approve] recorded {pair.id}: {len(text.split())} words, {len(echoed)} echoed lines kept; {len(pairs)} pairs total")
    if args.no_train:
        return

    rng = random.Random(args.seed)
    pool = [p for p in pairs if p.id != pair.id]
    long_form = [p for p in pool if p.format in LONG_FORM]
    short_form = [p for p in pool if p.format not in LONG_FORM]
    replay = rng.sample(long_form, min(len(long_form), REPLAY // 2)) + rng.sample(short_form, min(len(short_form), REPLAY - REPLAY // 2))
    update = [pair] * NEW_PAIR_COPIES + replay
    rng.shuffle(update)
    records = [TrainRecord(messages=build_messages(p.format, p.notes, p.keep_phrases[:2], p.text)) for p in update]
    update_path = Path("data/train_update.jsonl")
    write_jsonl(update_path, records)
    print(f"[approve] update set: {NEW_PAIR_COPIES} copies of the new post + {len(replay)} replayed pairs = {len(records)} records")

    sampler = json.loads(Path(args.sampler).read_text())
    stamp = datetime.now().strftime("%H%M%S")
    shutil.copy(args.sampler, "data/sampler_prev.json")
    cmd = [
        sys.executable, "-m", "train.train",
        "--train", str(update_path), "--val", args.val,
        "--model", sampler["model"], "--load-checkpoint", sampler["state_path"],
        "--epochs", "1", "--batch-size", "4", "--lr", str(args.lr), "--max-length", "2048", "--eval-every", "4",
        "--log-path", f"runs/tinker/update_{stamp}", "--sampler-out", args.sampler,
    ]
    print(f"[approve] warm-starting from {sampler['state_path']}")
    subprocess.run(cmd, check=True)
    print(f"[approve] data/sampler.json now points at the updated model; previous kept at data/sampler_prev.json")


if __name__ == "__main__":
    main()
