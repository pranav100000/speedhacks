# ingest

Drop your real exports here (all gitignored under `data/`):

- `data/x/` — unzipped X/Twitter archive (any folder containing `tweets.js` or `tweets-part*.js`)
- `data/linkedin/Shares.csv` — from the LinkedIn data export
- `data/gmail/*.mbox` — Google Takeout Mail export; set `AUTHOR_EMAIL` in `.env` (or pass `--author-email`)
- `data/blog/` — `.md` files and/or a `urls.txt` with one URL per line

Run from the repo root (every source is optional, at least one is required; `--gmail` is repeatable):

    uv run python -m ingest.run --x data/x --linkedin data/linkedin/Shares.csv --gmail data/gmail/sent.mbox --blog data/blog --out data/corpus.jsonl

Fixture smoke test: `uv run python -m ingest.run --x ingest/fixtures/x --linkedin ingest/fixtures/linkedin/Shares.csv --gmail ingest/fixtures/gmail/sent.mbox --author-email founder@example.com --blog ingest/fixtures/blog --out data/corpus_fixture.jsonl`
