"""Parse a directory of blog posts (.md files and/or a urls.txt) into CorpusRecords.

.md: title = YAML frontmatter `title:` or first `# ` heading or filename; text = body.
urls.txt: one URL per line; body and title via trafilatura. Extraction that
returns nothing raises.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import trafilatura

from common.schema import CorpusRecord, write_jsonl

HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"[ \t]+")


def split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unterminated YAML frontmatter")
    meta: dict[str, str] = {}
    for line in raw[4:end].split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("\"'")
    return meta, raw[end + 5 :]


def clean_markdown(body: str) -> str:
    body = MD_IMAGE_RE.sub("", body)
    body = MD_LINK_RE.sub(r"\1", body)
    body = URL_RE.sub("", body)
    lines = [WS_RE.sub(" ", line).rstrip() for line in body.replace("\r\n", "\n").split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return text.strip()


def parse_md_file(path: Path) -> CorpusRecord:
    meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
    title = meta.get("title")
    m = HEADING_RE.search(body)
    if m:
        if not title:
            title = m.group(1).strip()
        body = body[: m.start()] + body[m.end() :]
    if not title:
        title = path.stem
    text = clean_markdown(body)
    if not text:
        raise ValueError(f"{path}: empty body after cleaning")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").lower()
    return CorpusRecord(
        id=f"blog_{slug}",
        source="blog",
        format="blog",
        text=text,
        created_at=mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        title=title,
    )


def parse_url(url: str, today: str) -> CorpusRecord:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError(f"fetch failed for {url}")
    doc = trafilatura.bare_extraction(downloaded, url=url, include_comments=False, with_metadata=True)
    if doc is None or not doc.text or not doc.text.strip():
        raise RuntimeError(f"trafilatura extracted no text from {url}")
    if not doc.title or not doc.title.strip():
        raise RuntimeError(f"trafilatura extracted no title from {url}")
    text = clean_markdown(doc.text)
    if not text:
        raise RuntimeError(f"{url}: empty body after cleaning")
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return CorpusRecord(
        id=f"blog_url_{digest}",
        source="blog",
        format="blog",
        text=text,
        created_at=today,
        title=doc.title.strip(),
    )


def parse_blog_dir(blog_dir: Path) -> tuple[list[CorpusRecord], dict[str, int]]:
    if not blog_dir.is_dir():
        raise FileNotFoundError(f"blog directory not found: {blog_dir}")
    counts: dict[str, int] = defaultdict(int)
    records: list[CorpusRecord] = []
    md_files = sorted(p for p in blog_dir.rglob("*.md") if p.is_file() and p.name.lower() != "readme.md")
    for path in md_files:
        counts["blog_md_files"] += 1
        records.append(parse_md_file(path))
    urls_file = blog_dir / "urls.txt"
    if urls_file.is_file():
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        urls = [line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines()]
        urls = [u for u in urls if u and not u.startswith("#")]
        for url in urls:
            counts["blog_urls"] += 1
            records.append(parse_url(url, today))
    if not md_files and not urls_file.is_file():
        raise FileNotFoundError(f"no .md files and no urls.txt under {blog_dir}")
    counts["blog"] = len(records)
    records.sort(key=lambda r: r.created_at)
    return records, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="directory of .md files and/or urls.txt")
    ap.add_argument("--out", required=True, type=Path, help="output jsonl")
    args = ap.parse_args()
    records, counts = parse_blog_dir(args.input)
    n = write_jsonl(args.out, records)
    print(json.dumps(counts, indent=2))
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
