#!/bin/zsh
# Ten minutes before the demo. Every line must print OK.
cd "$(dirname "$0")"
set -a; source .env 2>/dev/null; set +a
ok() { echo "OK   $1"; }; bad() { echo "FAIL $1"; FAILED=1; }
curl -s -o /dev/null -m 5 -w "%{http_code}" http://localhost:7860/client/ | grep -q 200 && ok "interviewer serving on 7860" || bad "interviewer not serving: uv run python -m interview.bot --corpus data/corpus.jsonl"
curl -s -o /dev/null -m 5 -w "%{http_code}" http://localhost:8501/ | grep -q 200 && ok "compare page serving on 8501" || bad "compare page not serving: PYTHONPATH=. uv run streamlit run draft/compare.py"
[ "$(curl -s -o /dev/null -m 15 -w '%{http_code}' -H "Authorization: Token $DEEPGRAM_API_KEY" https://api.deepgram.com/v1/projects)" = 200 ] && ok "deepgram key" || bad "deepgram key"
[ "$(curl -s -o /dev/null -m 15 -w '%{http_code}' -H "X-API-Key: $CARTESIA_API_KEY" -H "Cartesia-Version: 2025-04-16" https://api.cartesia.ai/voices)" = 200 ] && ok "cartesia key" || bad "cartesia key"
curl -s -m 20 https://api.openai.com/v1/chat/completions -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" -d '{"model":"gpt-4.1","messages":[{"role":"user","content":"ok"}],"max_tokens":2}' | grep -q '"choices"' && ok "openai key + credits (bot brain)" || bad "openai key or credits: https://platform.openai.com/settings/organization/billing/"
uv run python -c "
import asyncio, sys
from common.claude import complete
print(asyncio.run(complete('Reply with the single word OK.', 'ping', max_tokens=5, effort='low')).strip())
" 2>/dev/null | grep -q OK && ok "anthropic key + credits" || bad "anthropic key or credits"
uv run python - <<'PY' 2>/dev/null && ok "tinker sampler answers (data/sampler.json)" || bad "tinker sampler: check billing https://tinker.thinkingmachines.ai/billing/balance"
import json, signal, sys
signal.signal(signal.SIGALRM, lambda *_: sys.exit(1)); signal.alarm(90)
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
d = json.load(open("data/sampler.json"))
sc = tinker.ServiceClient().create_sampling_client(model_path=d["sampler_path"])
tok = get_tokenizer(d["model"]); r = renderers.get_renderer(d["renderer"], tok)
prompt = r.build_generation_prompt([{"role": "user", "content": "Say hi."}])
out = sc.sample(prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=5, temperature=0.0)).result()
sys.exit(0 if out.sequences else 1)
PY
uv run python -c "
import asyncio, os
from dotenv import load_dotenv; load_dotenv()
from dial_sdk import DialClient, DialConfig
nums = asyncio.run(DialClient(DialConfig(api_key=os.environ['DIAL_API_KEY'])).list_numbers())
assert nums and nums[0].setup_status == 'ready' and 'call' in nums[0].capabilities
" 2>/dev/null && ok "dial key + number ready" || bad "dial: key missing or number not ready"
ls runs/run_*/drafts/finetune.md >/dev/null 2>&1 && ok "at least one run has drafts" || bad "no run with drafts: record the interview and run outline + generate"
[ -z "$FAILED" ] && echo "--- all clear" || { echo "--- fix the FAIL lines"; exit 1; }
