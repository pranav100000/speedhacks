"""Parse Shares.csv from a LinkedIn data export into CorpusRecords.

Columns: Date, ShareLink, ShareCommentary, SharedUrl, MediaUrl, Visibility.
Filters (logged with counts): empty commentary dropped.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common.schema import CorpusRecord, write_jsonl

REQUIRED_COLUMNS = ("Date", "ShareLink", "ShareCommentary")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"[ \t]+")
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def to_iso(date: str) -> str:
    dt = datetime.strptime(date.strip(), DATE_FMT).replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    lines = [WS_RE.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_linkedin_shares(path: Path) -> tuple[list[CorpusRecord], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"LinkedIn Shares.csv not found: {path}")
    counts: dict[str, int] = defaultdict(int)
    records: list[CorpusRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; found {reader.fieldnames}")
        for row_no, row in enumerate(reader, start=1):
            counts["linkedin_total"] += 1
            commentary = row["ShareCommentary"] or ""
            if not commentary.strip():
                counts["linkedin_empty_commentary_dropped"] += 1
                continue
            text = clean_text(commentary)
            if not text:
                counts["linkedin_empty_after_cleaning_dropped"] += 1
                continue
            counts["linkedin_post"] += 1
            records.append(
                CorpusRecord(
                    id=f"linkedin_{row_no}",
                    source="linkedin",
                    format="linkedin_post",
                    text=text,
                    created_at=to_iso(row["Date"]),
                )
            )
    records.sort(key=lambda r: r.created_at)
    return records, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="path to Shares.csv")
    ap.add_argument("--out", required=True, type=Path, help="output jsonl")
    args = ap.parse_args()
    records, counts = parse_linkedin_shares(args.input)
    n = write_jsonl(args.out, records)
    print(json.dumps(counts, indent=2))
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
