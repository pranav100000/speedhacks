"""Screen emails in a corpus: keep only the ones that are substantive writing and safe to learn from.

Every record with source gmail goes through one Claude call that labels it. Non-email records pass
through untouched. Decisions are written next to the output so they can be audited.

  uv run python -m synth.screen_emails --corpus data/corpus.jsonl --out data/corpus_screened.jsonl \
      --decisions data/email_screen.jsonl
"""

import argparse
import asyncio
from collections import Counter

from pydantic import BaseModel, Field

from common.claude import gather_limited, get_client, parse
from common.schema import CorpusRecord, read_jsonl, write_jsonl

SYSTEM = (
    "You screen a person's sent emails to decide which ones may be used as examples of their writing voice "
    "for public posts. Be strict on privacy and strict on substance."
)


class Decision(BaseModel):
    substantive: bool = Field(description="True if the email contains real writing: it explains, argues, pitches, teaches, or tells something. False for scheduling, confirmations, thank-yous, forms, forwarding, and one-line logistics.")
    public_safe: bool = Field(description="True only if nothing in it concerns personal relationships, family, health, money (salary, offers, negotiation, invoices, debts), legal matters, other people's private situations, or credentials. Business outreach, product explanations, technical discussion, and professional opinions are safe.")
    topic: str = Field(description="Three to six words naming what the email is about.")
    reason: str = Field(description="One short sentence justifying the two booleans.")


class Screened(BaseModel):
    id: str
    title: str | None
    words: int
    substantive: bool
    public_safe: bool
    topic: str
    reason: str


async def screen(records: list[CorpusRecord], concurrency: int) -> list[Screened]:
    client = get_client()

    async def one(r: CorpusRecord) -> Screened:
        d = await parse(SYSTEM, f"Subject: {r.title or ''}\n\n<email>\n{r.text}\n</email>", Decision, effort="low", client=client)
        return Screened(id=r.id, title=r.title, words=len(r.text.split()), **d.model_dump())

    return await gather_limited(records, one, concurrency=concurrency)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="corpus with only the kept emails")
    ap.add_argument("--decisions", required=True, help="one Screened record per email, kept or not")
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    corpus = read_jsonl(args.corpus, CorpusRecord)
    emails = [r for r in corpus if r.source == "gmail"]
    others = [r for r in corpus if r.source != "gmail"]
    print(f"screening {len(emails)} emails")
    decisions = asyncio.run(screen(emails, args.concurrency))
    by_id = {d.id: d for d in decisions}
    kept = [r for r in emails if by_id[r.id].substantive and by_id[r.id].public_safe]
    write_jsonl(args.decisions, decisions)
    n = write_jsonl(args.out, others + kept)

    reasons = Counter(
        "kept" if d.substantive and d.public_safe else "not substantive" if not d.substantive else "not public-safe"
        for d in decisions
    )
    print(f"wrote {n} records to {args.out}; emails: {dict(reasons)}")
    print(f"kept email words: {sum(len(r.text.split()) for r in kept)}")
    print("\nKEPT (longest first):")
    for r in sorted(kept, key=lambda r: len(r.text.split()), reverse=True):
        d = by_id[r.id]
        print(f"  {len(r.text.split()):5d}w  {d.topic:40s}  {(r.title or '')[:50]}")


if __name__ == "__main__":
    main()
