"""Stream a Google Takeout .mbox and keep the author's sent mail as CorpusRecords.

Keeps messages whose From contains the author email AND whose X-Gmail-Labels
contains "Sent". Takes the text/plain part, cuts quoted replies and signatures.
Filters (logged with counts): not-from-author, not-Sent, no text/plain part,
under 40 words after cleaning.
"""

import argparse
import json
import mailbox
import os
import re
from collections import defaultdict
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
from datetime import timezone
from pathlib import Path

from common.schema import CorpusRecord, write_jsonl

MIN_WORDS = 40
ON_WROTE_START_RE = re.compile(r"^On .+")
QUOTE_RE = re.compile(r"^>")
SIG_RE = re.compile(r"^-- ?$")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"[ \t]+")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•>]|\d+[.)])\s")
GREETING_RE = re.compile(r"^(hi|hey|hello|dear|yo|hiya|good (morning|afternoon|evening))\b[^\n]{0,80}$", re.I)
SIGNOFF_RE = re.compile(
    r"^(best|thanks|thank you|thanks again|thanks so much|cheers|regards|best regards|kind regards|warm regards|"
    r"talk soon|sincerely|appreciate it|much appreciated|take care|later)[,!.]*(\s*\n\s*)?(-\s*)?(pranav(\s+sharan)?)?[.!]?$",
    re.I,
)
NAME_ONLY_RE = re.compile(r"^-?\s*pranav(\s+sharan)?\s*$", re.I)
WRAP_MIN_FILL = 0.6  # a hard-wrapped line is at least this fraction of the paragraph's longest line
WRAP_LOOKAHEAD = 3  # "On ... wrote:" may wrap onto the next couple of lines


def decode_header_value(value: str | None) -> str | None:
    if value is None:
        return None
    return str(make_header(decode_header(value)))


def resolve_author_email(cli_value: str | None) -> str:
    author = cli_value or os.environ.get("AUTHOR_EMAIL")
    if not author:
        raise RuntimeError("author email not given: pass --author-email or set AUTHOR_EMAIL in .env")
    return author.strip().lower()


def text_plain_part(msg: Message) -> Message | None:
    for part in msg.walk():
        if part.get_content_type() != "text/plain":
            continue
        if part.get_content_disposition() == "attachment":
            continue
        return part
    return None


def decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raise ValueError("text/plain part has no decodable payload")
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset)


def cut_index(lines: list[str]) -> int:
    """Index of the first quoted-reply or signature line, or len(lines)."""
    n = len(lines)
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if QUOTE_RE.match(stripped) or SIG_RE.match(stripped):
            return i
        if ON_WROTE_START_RE.match(stripped):
            for j in range(i, min(i + WRAP_LOOKAHEAD, n)):
                joined = " ".join(l.strip() for l in lines[i : j + 1])
                if joined.rstrip().endswith("wrote:"):
                    return i
    return n


def unwrap_paragraph(lines: list[str]) -> str:
    """Join a hard-wrapped (~76 col) paragraph into one line; leave lists and short-line text alone."""
    if len(lines) < 2 or any(LIST_ITEM_RE.match(l) for l in lines):
        return "\n".join(lines)
    longest = max(len(l) for l in lines)
    if longest < 50 or any(len(l) < WRAP_MIN_FILL * longest for l in lines[:-1]):
        return "\n".join(lines)
    return " ".join(l.strip() for l in lines)


def clean_body(body: str) -> str:
    lines = body.replace("\r\n", "\n").split("\n")
    lines = lines[: cut_index(lines)]
    text = "\n".join(WS_RE.sub(" ", URL_RE.sub("", line)).rstrip() for line in lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = [unwrap_paragraph(p.split("\n")) for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p.strip()]
    if paragraphs and GREETING_RE.match(paragraphs[0].strip()):
        paragraphs = paragraphs[1:]
    while paragraphs and (SIGNOFF_RE.match(paragraphs[-1].strip()) or NAME_ONLY_RE.match(paragraphs[-1].strip())):
        paragraphs = paragraphs[:-1]
    return "\n\n".join(paragraphs)


def to_iso(msg: Message) -> str:
    date = msg.get("Date")
    if not date:
        raise ValueError("message has no Date header")
    dt = parsedate_to_datetime(date)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_mbox(path: Path, author_email: str, since: str | None = None) -> tuple[list[CorpusRecord], dict[str, int]]:
    """since: ISO date (YYYY-MM-DD); emails dated before it are dropped and counted."""
    if not path.is_file():
        raise FileNotFoundError(f"mbox not found: {path}")
    counts: dict[str, int] = defaultdict(int)
    records: list[CorpusRecord] = []
    box = mailbox.mbox(str(path), create=False)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_")
    for key, msg in box.iteritems():
        counts["gmail_total"] += 1
        from_header = decode_header_value(msg.get("From")) or ""
        _, from_addr = parseaddr(from_header)
        if author_email not in from_addr.lower() and author_email not in from_header.lower():
            counts["gmail_not_from_author_dropped"] += 1
            continue
        labels = decode_header_value(msg.get("X-Gmail-Labels")) or ""
        if "sent" not in [l.strip().lower() for l in labels.split(",")]:
            counts["gmail_not_sent_label_dropped"] += 1
            continue
        part = text_plain_part(msg)
        if part is None:
            counts["gmail_no_text_plain_dropped"] += 1
            continue
        text = clean_body(decode_part(part))
        if len(text.split()) < MIN_WORDS:
            counts["gmail_under_40_words_dropped"] += 1
            continue
        created_at = to_iso(msg)
        if since and created_at < since:
            counts["gmail_before_cutoff_dropped"] += 1
            continue
        counts["gmail_email"] += 1
        subject = decode_header_value(msg.get("Subject"))
        records.append(
            CorpusRecord(
                id=f"gmail_{stem}_{key}",
                source="gmail",
                format="email",
                text=text,
                created_at=created_at,
                title=subject.strip() if subject and subject.strip() else None,
            )
        )
    records.sort(key=lambda r: r.created_at)
    return records, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="path to a Takeout .mbox")
    ap.add_argument("--out", required=True, type=Path, help="output jsonl")
    ap.add_argument("--author-email", default=None, help="defaults to AUTHOR_EMAIL from .env")
    ap.add_argument("--since", default=None, help="drop emails dated before YYYY-MM-DD")
    args = ap.parse_args()
    records, counts = parse_mbox(args.input, resolve_author_email(args.author_email), args.since)
    n = write_jsonl(args.out, records)
    print(json.dumps(counts, indent=2))
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
