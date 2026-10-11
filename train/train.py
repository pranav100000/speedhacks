"""Tinker LoRA SFT on data/train.jsonl, evaluated on data/val.jsonl.

Mirrors tinker_cookbook/recipes/sl_basic.py, with two differences:
  * the dataset builder reads an explicit held-out val file (the cookbook's
    FromConversationFileBuilder only knows `test_size`, a split of one file);
  * after training, the final checkpoint's sampler path is written to
    data/sampler.json so train/sample.py can recreate the SamplingClient.

Run from the repo root:
  uv run python -m train.train --train data/train.jsonl --val data/val.jsonl
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import chz
import datasets
from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

from common.schema import TrainRecord, read_jsonl

DEFAULT_MODEL = "Qwen/Qwen3.5-9B-Base"
RECIPE_NAME = "speedhacks_voice_sft"


def load_conversations(path: str) -> list[dict]:
    """Read a Tinker-format JSONL file: every line is {"messages": [...]}. Raises on anything else."""
    rows: list[dict] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "messages" not in data:
                raise ValueError(f"{path}:{line_no} has no 'messages' key; got keys {list(data)}")
            rows.append({"messages": data["messages"]})
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


@chz.chz
class TrainValConversationFileBuilder(ChatDatasetBuilder):
    """Like FromConversationFileBuilder, but train and val come from two explicit files."""

    train_path: str
    val_path: str

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_rows = load_conversations(self.train_path)
        val_rows = load_conversations(self.val_path)
        batch_size = self.common_config.batch_size
        if len(train_rows) < batch_size:
            raise ValueError(
                f"{self.train_path} has {len(train_rows)} records, fewer than batch_size={batch_size}; "
                "the dataset would yield zero batches"
            )
        if self.common_config.train_on_what is None:
            raise ValueError("train_on_what must be set explicitly")
        train_on_what = TrainOnWhat(self.common_config.train_on_what)
        renderer = self.renderer
        max_length = self.common_config.max_length

        def map_fn(row: dict):
            return conversation_to_datum(row["messages"], renderer, max_length, train_on_what)

        train_ds = SupervisedDatasetFromHFDataset(
            datasets.Dataset.from_list(train_rows), batch_size=batch_size, map_fn=map_fn
        )
        # One batch holding the whole val set: NLLEvaluator scores it in a single forward call.
        val_ds = SupervisedDatasetFromHFDataset(
            datasets.Dataset.from_list(val_rows), batch_size=len(val_rows), map_fn=map_fn
        )
        return train_ds, val_ds


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="Tinker conversation JSONL, e.g. data/train.jsonl")
    p.add_argument("--val", required=True, help="Held-out JSONL in the same format, e.g. data/val.jsonl")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--rank", type=int, default=32, help="LoRA rank")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=4096, help="Token cap per example after rendering")
    p.add_argument("--eval-every", type=int, default=10, help="Run the val NLL evaluator every N steps (0 disables)")
    p.add_argument(
        "--log-path",
        default=None,
        help="Checkpoints + metrics dir. Default runs/tinker/<YYYYmmdd-HHMMSS>. Must not already exist.",
    )
    p.add_argument("--sampler-out", default="data/sampler.json", help="Where to write the sampler descriptor")
    p.add_argument("--load-checkpoint", default=None, help="tinker:// state_path to warm-start from (default: base model)")
    p.add_argument(
        "--skip-schema-check",
        action="store_true",
        help="Do not validate records against common.schema.TrainRecord. Only for the cookbook fixture, "
        "whose records lack a system turn.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log_path or f"runs/tinker/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_path = str(Path(log_path).resolve())
    train_path = str(Path(args.train).resolve())
    val_path = str(Path(args.val).resolve())

    if args.skip_schema_check:
        print(f"[train] schema check skipped for {train_path} and {val_path}")
    else:
        n_train = len(read_jsonl(train_path, TrainRecord))
        n_val = len(read_jsonl(val_path, TrainRecord))
        print(f"[train] schema ok: {n_train} train records, {n_val} val records (TrainRecord)")

    renderer_name = model_info.get_recommended_renderer_name(args.model)
    print(f"[train] model={args.model} renderer={renderer_name} rank={args.rank} "
          f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} max_length={args.max_length}")
    print(f"[train] log_path={log_path}")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset_builder = TrainValConversationFileBuilder(
        common_config=common_config, train_path=train_path, val_path=val_path
    )
    config = train.Config(
        log_path=log_path,
        model_name=args.model,
        recipe_name=RECIPE_NAME,
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=args.epochs,
        lora_rank=args.rank,
        eval_every=args.eval_every,
        load_checkpoint_path=args.load_checkpoint,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists="raise")
    t0 = time.time()
    asyncio.run(train.main(config))
    elapsed = time.time() - t0

    # train.main ends with CheckpointManager.save_final_async(kind="both"), which appends a
    # record with both state_path and sampler_path to <log_path>/checkpoints.jsonl.
    # That sampler_path (a tinker:// URI) is what create_sampling_client(model_path=...) takes.
    record = checkpoint_utils.get_last_checkpoint(log_path, required_key="sampler_path")
    if record is None or not record.sampler_path:
        raise RuntimeError(f"no checkpoint with a sampler_path in {log_path}/checkpoints.jsonl")

    sampler = {
        "model": args.model,
        "renderer": renderer_name,
        "sampler_path": record.sampler_path,
        "state_path": record.state_path,
        "log_path": log_path,
        "train": train_path,
        "val": val_path,
        "rank": args.rank,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warm_start_from": args.load_checkpoint,
        "max_length": args.max_length,
        "elapsed_seconds": round(elapsed, 1),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    sampler_out = Path(args.sampler_out).resolve()
    write_json_atomic(sampler_out, sampler)

    banner = "=" * 72
    print(banner)
    print(f"TRAINING DONE in {elapsed/60:.1f} min")
    print(f"SAMPLER PATH : {record.sampler_path}")
    print(f"STATE PATH   : {record.state_path}")
    print(f"LOG PATH     : {log_path}")
    print(f"WROTE        : {sampler_out}")
    print(banner)


if __name__ == "__main__":
    main()
