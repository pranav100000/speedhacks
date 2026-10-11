"""Parse an X/Twitter archive into CorpusRecords.

Input: a directory containing tweets.js / data/tweets.js / tweets-part*.js.
Filters (logged with counts): retweets dropped, empty-after-cleaning dropped.
Consecutive self-replies are chained into one `thread` record; replies to
other people are tagged `reply`.
"""

import argparse
import html
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common.schema import CorpusRecord, write_jsonl

TCO_RE = re.compile(r"https?://t\.co/\S+")
WS_RE = re.compile(r"[ \t]+")
CREATED_AT_FMT = "%a %b %d %H:%M:%S %z %Y"


def find_tweet_files(archive_dir: Path) -> list[Path]:
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"X archive directory not found: {archive_dir}")
    files = sorted(p for p in archive_dir.rglob("tweets*.js") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no tweets*.js under {archive_dir}")
    return files


def load_tweet_file(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    eq = raw.find("=")
    if eq < 0:
        raise ValueError(f"{path}: expected a 'window.YTD... = [...]' prefix, found no '='")
    payload = json.loads(raw[eq + 1 :])
    if not isinstance(payload, list):
        raise ValueError(f"{path}: top-level JSON is not a list")
    tweets = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict) or "tweet" not in item:
            raise ValueError(f"{path}[{i}]: element has no 'tweet' key")
        tweets.append(item["tweet"])
    return tweets


def to_iso(created_at: str) -> str:
    dt = datetime.strptime(created_at, CREATED_AT_FMT)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = TCO_RE.sub("", text)
    lines = [WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def parse_x_archive(archive_dir: Path) -> tuple[list[CorpusRecord], dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    tweets: dict[str, dict] = {}
    for path in find_tweet_files(archive_dir):
        for t in load_tweet_file(path):
            for key in ("id_str", "full_text", "created_at"):
                if key not in t:
                    raise ValueError(f"{path}: tweet missing '{key}': {t}")
            counts["x_total"] += 1
            if t["full_text"].startswith("RT @"):
                counts["x_retweets_dropped"] += 1
                continue
            if t["id_str"] in tweets:
                raise ValueError(f"{path}: duplicate tweet id {t['id_str']}")
            tweets[t["id_str"]] = t

    # Self-reply chains: parent is one of our own tweets. Walk up to the root.
    def parent_of(t: dict) -> str | None:
        pid = t.get("in_reply_to_status_id_str")
        return pid if pid in tweets else None

    def root_of(tid: str) -> str:
        seen = set()
        while True:
            if tid in seen:
                raise ValueError(f"reply cycle at tweet {tid}")
            seen.add(tid)
            pid = parent_of(tweets[tid])
            if pid is None:
                return tid
            tid = pid

    groups: dict[str, list[dict]] = defaultdict(list)
    for tid, t in tweets.items():
        groups[root_of(tid)].append(t)

    records: list[CorpusRecord] = []
    for root_id, group in groups.items():
        group.sort(key=lambda t: (to_iso(t["created_at"]), int(t["id_str"])))
        root = tweets[root_id]
        root_is_reply = bool(root.get("in_reply_to_status_id_str"))
        if root_is_reply or len(group) == 1:
            for t in group:
                text = clean_text(t["full_text"])
                if not text:
                    counts["x_empty_after_cleaning_dropped"] += 1
                    continue
                fmt = "reply" if t.get("in_reply_to_status_id_str") else "tweet"
                counts[f"x_{fmt}"] += 1
                records.append(
                    CorpusRecord(
                        id=f"x_{t['id_str']}",
                        source="x",
                        format=fmt,
                        text=text,
                        created_at=to_iso(t["created_at"]),
                    )
                )
            continue
        parts = [clean_text(t["full_text"]) for t in group]
        dropped = sum(1 for p in parts if not p)
        counts["x_empty_after_cleaning_dropped"] += dropped
        parts = [p for p in parts if p]
        if not parts:
            continue
        counts["x_thread"] += 1
        counts["x_thread_tweets_merged"] += len(parts)
        records.append(
            CorpusRecord(
                id=f"x_{root_id}",
                source="x",
                format="thread",
                text="\n\n".join(parts),
                created_at=to_iso(root["created_at"]),
            )
        )
    records.sort(key=lambda r: r.created_at)
    return records, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="X archive directory")
    ap.add_argument("--out", required=True, type=Path, help="output jsonl")
    args = ap.parse_args()
    records, counts = parse_x_archive(args.input)
    n = write_jsonl(args.out, records)
    print(json.dumps(counts, indent=2))
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
