"""Run every ingest parser and write data/corpus.jsonl (SCOPE.md §3).

Usage:
  uv run python -m ingest.run --x data/x --linkedin data/linkedin/Shares.csv \
      --gmail data/gmail/sent.mbox --blog data/blog --out data/corpus.jsonl
"""

import argparse
from collections import Counter
from pathlib import Path

from common.schema import CorpusRecord, write_jsonl
from ingest.blog import parse_blog_dir
from ingest.gmail import parse_mbox, resolve_author_email
from ingest.linkedin import parse_linkedin_shares
from ingest.x import parse_x_archive


def print_table(title: str, rows: dict[str, int]) -> None:
    print(f"\n{title}")
    width = max((len(k) for k in rows), default=10)
    for k, v in rows.items():
        print(f"  {k:<{width}}  {v:>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=Path, help="X archive directory (contains tweets*.js)")
    ap.add_argument("--linkedin", type=Path, help="LinkedIn Shares.csv")
    ap.add_argument("--gmail", type=Path, action="append", default=[], help="Takeout .mbox (repeatable)")
    ap.add_argument("--blog", type=Path, help="directory of .md files and/or urls.txt")
    ap.add_argument("--author-email", default=None, help="for --gmail; defaults to AUTHOR_EMAIL in .env")
    ap.add_argument("--gmail-since", default=None, help="for --gmail; drop emails dated before YYYY-MM-DD")
    ap.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    args = ap.parse_args()

    if not (args.x or args.linkedin or args.gmail or args.blog):
        ap.error("give at least one of --x, --linkedin, --gmail, --blog")

    records: list[CorpusRecord] = []
    filters: dict[str, int] = {}
    if args.x:
        recs, counts = parse_x_archive(args.x)
        records += recs
        filters.update(counts)
    if args.linkedin:
        recs, counts = parse_linkedin_shares(args.linkedin)
        records += recs
        filters.update(counts)
    if args.gmail:
        author = resolve_author_email(args.author_email)
        for mbox_path in args.gmail:
            recs, counts = parse_mbox(mbox_path, author, args.gmail_since)
            records += recs
            for k, v in counts.items():
                filters[k] = filters.get(k, 0) + v
    if args.blog:
        recs, counts = parse_blog_dir(args.blog)
        records += recs
        filters.update(counts)

    ids = Counter(r.id for r in records)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate corpus ids: {dupes[:10]}")

    n = write_jsonl(args.out, records)
    print_table("by source", dict(Counter(r.source for r in records)))
    print_table("by format", dict(Counter(r.format for r in records)))
    print_table("parser counts and filters", filters)
    print(f"\nwrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
