"""Streamlit: the three drafts side by side, labels hidden, order shuffled per run.

Run from the repo root:  PYTHONPATH=. uv run streamlit run draft/compare.py
"""

import json
import random
from pathlib import Path

import streamlit as st

from common.schema import Outline

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
VARIANTS = ("vanilla", "fewshot", "finetune")
LETTERS = ("A", "B", "C")


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def survived(draft: str, keep_phrases: list[str]) -> int:
    body = normalize_ws(draft)
    return sum(normalize_ws(p) in body for p in keep_phrases)


def load_judge(run: Path) -> dict[str, dict]:
    """Per-draft scores from draft.judge, or {} if judge.json has not been written for this run."""
    path = run / "judge.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())["drafts"]


def runs_with_drafts() -> list[Path]:
    if not RUNS.is_dir():
        raise FileNotFoundError(f"{RUNS} does not exist")
    return sorted(p for p in RUNS.iterdir() if p.is_dir() and any((p / "drafts").glob("*.md")))


st.set_page_config(page_title="Which one is you?", layout="wide")

runs = runs_with_drafts()
if not runs:
    st.error(f"No run under {RUNS} has a drafts/ folder with .md files. Run draft.generate first.")
    st.stop()
run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
outline = Outline.model_validate_json((run / "outline.json").read_text())
phrases = outline.keep_phrases
judged = load_judge(run)

st.title(outline.title)

order = list(VARIANTS)
random.Random(run.name).shuffle(order)
reveal_key = f"revealed:{run.name}"
if st.button("Reveal", key=f"reveal-btn:{run.name}"):
    st.session_state[reveal_key] = True
revealed = st.session_state.get(reveal_key, False)

for col, letter, name in zip(st.columns(3), LETTERS, order):
    with col:
        st.subheader(f"Draft {letter}" + (f": {name}" if revealed else ""))
        path = run / "drafts" / f"{name}.md"
        if not path.exists():
            st.error(f"missing {path.relative_to(ROOT)}")
            continue
        draft = path.read_text()
        st.caption(f"keep phrases survived: {survived(draft, phrases)}/{len(phrases)}")
        if name in judged:
            j = judged[name]
            st.caption(
                f"voice {j['voice_score']}/10 · {len(j['unsupported_claims'])} unsupported claims · "
                f"{len(j['repeated_ngrams'])} repeats"
            )
        st.markdown(draft)

with st.expander("Outline: notes and keep phrases", expanded=False):
    st.markdown("**Notes**")
    st.markdown(outline.notes)
    st.markdown("**Keep phrases**")
    st.markdown("\n".join(f"- {p}" for p in phrases))
    for name in order:
        if name in judged:
            st.markdown(f"**Unsupported claims: {name}**")
            claims = judged[name]["unsupported_claims"]
            st.markdown("\n".join(f"- \"{c['span']}\" ({c['why']})" for c in claims) if claims else "_none_")
