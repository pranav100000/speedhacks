"""pairs.jsonl -> train.jsonl + val.jsonl in Tinker conversation format.

Caps tweets at the most recent --max-tweets (by corpus created_at), keeps every blog,
linkedin_post and thread, stratified split by format, shuffles train with --seed.

Usage:
  uv run python -m synth.make_train --pairs data/pairs.jsonl --corpus data/corpus.jsonl \
      --train data/train.jsonl --val data/val.jsonl --max-tweets 300 --val-frac 0.1 --seed 0
"""

import argparse
import random
from collections import defaultdict

from common.schema import (
    TRAINABLE_FORMATS,
    CorpusRecord,
    PairRecord,
    TrainRecord,
    build_messages,
    read_jsonl,
    write_jsonl,
)
from synth.util import created_at_key


def corpus_id_for(pair_id: str) -> str:
    """Variant and section pairs are named <corpus_id>__sparse / __dense / __s<N>."""
    return pair_id.split("__", 1)[0]


def parse_repeat(spec: str) -> dict[str, int]:
    """'blog=3,section=2' -> {'blog': 3, 'section': 2}; a value is the total number of copies."""
    out: dict[str, int] = {}
    for part in filter(None, spec.split(",")):
        fmt, _, n = part.partition("=")
        if fmt not in TRAINABLE_FORMATS or not n.isdigit() or int(n) < 1:
            raise ValueError(f"bad --repeat entry {part!r}; expected <format>=<copies>")
        out[fmt] = int(n)
    return out


def cap_tweets(pairs: list[PairRecord], corpus_by_id: dict[str, CorpusRecord], max_tweets: int) -> list[PairRecord]:
    tweets = [p for p in pairs if p.format == "tweet"]
    others = [p for p in pairs if p.format != "tweet"]
    tweets.sort(key=lambda p: created_at_key(corpus_by_id[corpus_id_for(p.id)]), reverse=True)
    kept = tweets[:max_tweets]
    print(f"tweets: {len(tweets)} total, kept most recent {len(kept)}, dropped {len(tweets) - len(kept)}")
    return others + kept


def cap_emails(pairs: list[PairRecord], max_emails: int) -> list[PairRecord]:
    """Keep the longest emails: those are the ones with actual writing in them."""
    emails = [p for p in pairs if p.format == "email"]
    others = [p for p in pairs if p.format != "email"]
    emails.sort(key=lambda p: len(p.text.split()), reverse=True)
    kept = emails[:max_emails]
    print(f"emails: {len(emails)} total, kept longest {len(kept)}, dropped {len(emails) - len(kept)}")
    return others + kept


def stratified_split(
    pairs: list[PairRecord], val_frac: float, rng: random.Random, keep_in_train: set[str]
) -> tuple[list[PairRecord], list[PairRecord]]:
    by_format: dict[str, list[PairRecord]] = defaultdict(list)
    for p in pairs:
        by_format[p.format].append(p)
    train: list[PairRecord] = []
    val: list[PairRecord] = []
    for fmt in TRAINABLE_FORMATS:
        group = by_format.get(fmt, [])
        n = len(group)
        n_val = round(n * val_frac)
        if n >= 5:
            n_val = max(n_val, 1)
        if fmt in keep_in_train:
            n_val = 0
        idx = list(range(n))
        rng.shuffle(idx)
        val_idx = set(idx[:n_val])
        for i, p in enumerate(group):
            (val if i in val_idx else train).append(p)
    return train, val


def to_train_record(p: PairRecord, rng: random.Random, max_phrases: int, phrase_dropout: float) -> TrainRecord:
    """Phrases are soft and optional at inference, so training sees 0 to max_phrases of them."""
    if rng.random() < phrase_dropout:
        phrases: list[str] = []
    else:
        k = min(len(p.keep_phrases), rng.randint(1, max_phrases))
        phrases = rng.sample(p.keep_phrases, k)
    return TrainRecord(messages=build_messages(p.format, p.notes, phrases, p.text))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="input data/pairs.jsonl")
    ap.add_argument("--corpus", required=True, help="data/corpus.jsonl, joined on id for created_at")
    ap.add_argument("--train", required=True, help="output data/train.jsonl")
    ap.add_argument("--val", required=True, help="output data/val.jsonl")
    ap.add_argument("--max-tweets", type=int, default=300)
    ap.add_argument("--max-emails", type=int, default=80)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", default="", help="upweight formats in train, e.g. blog=3,section=2 (total copies)")
    ap.add_argument("--keep-in-train", default="", help="comma-separated formats that are never held out, e.g. blog,section")
    ap.add_argument("--max-phrases", type=int, default=2, help="max lines to echo per training record (default: 2)")
    ap.add_argument("--phrase-dropout", type=float, default=0.33, help="share of records trained with no lines to echo (default: 0.33)")
    args = ap.parse_args()
    repeat = parse_repeat(args.repeat)
    keep_in_train = {f for f in args.keep_in_train.split(",") if f}
    unknown = keep_in_train - set(TRAINABLE_FORMATS)
    if unknown:
        raise ValueError(f"--keep-in-train has non-trainable formats: {unknown}")

    pairs = read_jsonl(args.pairs, PairRecord)
    corpus_by_id = {r.id: r for r in read_jsonl(args.corpus, CorpusRecord)}
    missing = [p.id for p in pairs if corpus_id_for(p.id) not in corpus_by_id]
    if missing:
        raise ValueError(f"{len(missing)} pair ids are not in the corpus, e.g. {missing[:5]}")
    bad = [p.id for p in pairs if p.format not in TRAINABLE_FORMATS]
    if bad:
        raise ValueError(f"{len(bad)} pairs have a non-trainable format, e.g. {bad[:5]}")

    rng = random.Random(args.seed)
    pairs = cap_tweets(pairs, corpus_by_id, args.max_tweets)
    pairs = cap_emails(pairs, args.max_emails)
    train_pairs, val_pairs = stratified_split(pairs, args.val_frac, rng, keep_in_train)
    base_counts = {fmt: sum(1 for p in train_pairs if p.format == fmt) for fmt in repeat}
    train_pairs = [p for p in train_pairs for _ in range(repeat.get(p.format, 1))]
    for fmt, n in repeat.items():
        print(f"repeat: {fmt} x{n} ({base_counts[fmt]} -> {base_counts[fmt] * n})")
    rng.shuffle(train_pairs)

    n_train = write_jsonl(args.train, (to_train_record(p, rng, args.max_phrases, args.phrase_dropout) for p in train_pairs))
    n_val = write_jsonl(args.val, (to_train_record(p, rng, args.max_phrases, args.phrase_dropout) for p in val_pairs))

    print(f"{'format':14s} {'train':>6s} {'val':>6s}")
    for fmt in TRAINABLE_FORMATS:
        t = sum(1 for p in train_pairs if p.format == fmt)
        v = sum(1 for p in val_pairs if p.format == fmt)
        print(f"{fmt:14s} {t:6d} {v:6d}")
    print(f"{'total':14s} {n_train:6d} {n_val:6d}")
    print(f"wrote {args.train} and {args.val}")


if __name__ == "__main__":
    main()
