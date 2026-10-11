"""Generate a draft from runs/<id>/outline.json with the fine-tuned sampler.

Builds exactly the messages training saw (common.schema.build_messages, no
assistant turn), renders them with the same cookbook renderer the training
run used, samples from the saved weights, and stops at the renderer's
end-of-turn sequence.

Run from the repo root:
  uv run python -m train.sample --outline runs/<id>/outline.json --out runs/<id>/drafts/finetune.md
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import tinker
from tinker_cookbook.renderers import Message as RendererMessage
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from common.schema import Outline, build_messages

REQUIRED_SAMPLER_KEYS = ("model", "renderer", "sampler_path")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outline", required=True, help="runs/<id>/outline.json")
    p.add_argument("--out", required=True, help="runs/<id>/drafts/finetune.md (parent dirs are created)")
    p.add_argument("--sampler", default="data/sampler.json", help="Descriptor written by train.train")
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--n", type=int, default=1, help="Samples to draw; the 2nd+ go to <out>.2.md, <out>.3.md, ...")
    p.add_argument("--style-context", default=None, help="Optional text file (profile + example posts) appended to the system turn. Off by default: training never saw it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise ValueError("--n must be >= 1")

    sampler = json.loads(Path(args.sampler).read_text())
    missing = [k for k in REQUIRED_SAMPLER_KEYS if not sampler.get(k)]
    if missing:
        raise KeyError(f"{args.sampler} is missing {missing}")

    outline = Outline.model_validate_json(Path(args.outline).read_text())

    # Same system + user turns as training; no assistant turn, the model writes that.
    messages = build_messages(outline.format, outline.notes, outline.keep_phrases)
    if args.style_context:
        context = Path(args.style_context).read_text().strip()
        if not context:
            raise ValueError(f"{args.style_context} is empty")
        messages[0].content = f"{messages[0].content}\n\n{context}"
        print(f"[sample] style context: {len(context.split())} words appended to the system turn", file=sys.stderr)
    renderer_messages = [RendererMessage(role=m.role, content=m.content) for m in messages]

    tokenizer = get_tokenizer(sampler["model"])
    renderer = get_renderer(sampler["renderer"], tokenizer)
    prompt = renderer.build_generation_prompt(renderer_messages)
    stop = renderer.get_stop_sequences()
    print(
        f"[sample] model={sampler['model']} renderer={sampler['renderer']} "
        f"prompt_tokens={prompt.length} stop={stop!r} sampler_path={sampler['sampler_path']}",
        file=sys.stderr,
    )

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=sampler["sampler_path"])
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, stop=stop)
    response = sampling_client.sample(prompt, num_samples=args.n, sampling_params=params).result()
    if len(response.sequences) != args.n:
        raise RuntimeError(f"asked for {args.n} samples, got {len(response.sequences)}")

    out = Path(args.out)
    for i, seq in enumerate(response.sequences):
        tokens = seq.tokens
        message, termination = renderer.parse_response(tokens)
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError(f"renderer returned non-text content: {type(content).__name__}")
        text = content.strip()
        print(
            f"[sample {i + 1}/{args.n}] tokens={len(tokens)} stop_reason={seq.stop_reason} "
            f"termination={termination}",
            file=sys.stderr,
        )
        if not text:
            raise RuntimeError(f"sample {i + 1} decoded to empty text")
        target = out if i == 0 else out.with_name(f"{out.stem}.{i + 1}{out.suffix}")
        write_text_atomic(target, text)
        print(f"[sample {i + 1}/{args.n}] wrote {target}", file=sys.stderr)
        if i > 0:
            print("\n" + "-" * 72 + "\n")
        print(text)


if __name__ == "__main__":
    main()
