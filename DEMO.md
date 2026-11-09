# Demo recording script: problem, stack, live demo + code (about 5:00)

Their recorder captures screen and camera in one take. Set up before you press record:

- Phone on speaker, in the camera frame.
- Terminal 1: `uv run python -m interview.dial_inbound --corpus data/corpus.jsonl --fast --deliver imessage` already running (the post lands in Messages on your phone; from your own account, since the Dial number needs A2P/10DLC for SMS and its iMessage line costs $250/month), showing `+15722283645 now answers as the interviewer`.
- Terminal 2: this command typed, not yet run: `./demo_train.sh runs/run_20261011_141849 text`
- Chrome: http://localhost:8501 on run `run_20261011_141849`, reloaded so labels are hidden. Two columns now: see the letter mapping printed by the setup check.
- Editor with these six files open, so you can tab through them:
  `common/schema.py` (line 99), `synth/make_pairs.py` (line 27), `train/train.py` (lines 157 and 168),
  `draft/judge.py` (line 91), `draft/generate.py` (line 110), `interview/dial_inbound.py` (line 105).
- `./demo_preflight.sh` green. Reset state: `cp data/sampler_demo.json data/sampler.json` and no `approved_*` line in `data/pairs.jsonl`.

| # | Clock | Section | On screen | Say |
|---|---|---|---|---|
| 1 | 0:00 | The problem | Camera | "I'm a solo founder. Writing is my bottleneck, and anything I write with an LLM sounds like the LLM. The fix isn't a better prompt. It's getting the substance out of me by talking, and putting the voice into a model that has only ever read me. LoRAkeet: a LoRA that mimics you." |
| 2 | 0:30 | Tech stack | README top | "Dial gives it a phone number and runs the voice agent, so the interviewer is a system prompt. Claude does the reasoning: speech to notes, training pairs synthesized backwards from my posts, and a judge for claims I never made. Tinker trains a LoRA on Qwen 35B in two minutes per person and warm-starts it in one. Without Tinker this is a prompt. Without Dial it's a web form." |
| 3 | 1:00 | LIVE 0:00 | Terminal 2 | Run `./demo_train.sh runs/run_20261011_141849 text`. "The post I approved this morning. Training on it now, warm start, about a minute. Leave it." |
| 4 | 1:10 | LIVE 0:10 | Phone on speaker, Terminal 1 behind it | Dial +1 572 228 3645. It greets you by name and asks the topic. One sentence. It mirrors a fragment, asks a follow-up. One sentence. Hang up at LIVE 0:40. "A real one is four minutes. It's writing from that now; about forty seconds." |
| 5 | 1:45 | LIVE 0:45 | Compare tab, labels hidden | "This morning's full call. One is Claude. One is the model that has only ever read me. Which is me?" Four-second hold. Reveal. "Claude: 631 words, headers, ten claims I never made. The fine-tune: 204 words, zero." |
| 6 | 2:25 | LIVE 1:25 | Terminal 1, then phone | `transcript at runs/...` then `imessage: sent`. Pick up the phone: the post is in Messages. "That is the post from the call I just made." Scroll it for two seconds. |
| 7 | 2:45 | LIVE 1:45 | Terminal 2 | `TRAINING DONE in 1.0 min` is on screen. Run `./demo_sample.sh runs/run_20261011_141849`. "And that's the model with this morning's post in it, writing the next one." First lines stream. |
| 8 | 3:00 | Code | Editor | 60 seconds, three stops: `common/schema.py` line 99 (one prompt for train and serve; notes, never prose), `synth/make_pairs.py` line 27 (pairs synthesized backwards), `train/train.py` lines 157 and 168 (the Tinker config and the warm-start line). One sentence each. Optional fourth: `draft/judge.py` line 91. |
| 9 | 4:00 | Close | Camera | "Call a number, talk, hang up, get a post that's yours, and every post you approve makes the next one more you." |

Live segment is beats 3 to 7: two minutes, three things happening at once. Training starts at LIVE 0:00 and is needed at LIVE 1:45. The call's pipeline starts at LIVE 0:40 in fast mode (no cleanup pass, fine-tune only, two candidates, about 35 s) and delivers at LIVE 1:20 to 1:30, during the reveal.

Three-minute cut: drop beats 6 and 8, shorten beat 10 to the reveal.

## If something breaks on the take

- The number doesn't answer: run `uv run python -m interview.dial_call --to +15038287565 --corpus data/corpus.jsonl` in Terminal 1 instead; it calls you.
- Delivery is iMessage from this Mac (`--deliver imessage`), because US carriers reject SMS from a number without A2P/10DLC registration, which Dial's new number doesn't have yet. On the take, say "it texts me the post" and hold up the phone. If nothing arrives by beat 9: `cat runs/<newest>/drafts/finetune.md` and keep going.
- Training not done by beat 11: show `runs/tinker/update_*/metrics.jsonl` from the rehearsal and close.
- Reset between takes: `cp data/sampler_demo.json data/sampler.json`; delete the `approved_*` line from `data/pairs.jsonl`; reload the compare tab.

---

## Verbatim script (dress rehearsal, 16:07)

### Before you press record (2 minutes)
1. `./demo_preflight.sh` → all OK. `cp data/sampler_demo.json data/sampler.json`. No `approved_*` line in `data/pairs.jsonl`.
2. Windows, left to right: Terminal 1 (the daemon, its last line `+15722283645 now answers as the interviewer`), Terminal 2 (prompt at repo root, this typed and NOT run: `./demo_train.sh runs/run_20261011_141849 text`), Chrome on http://localhost:8501 with `run_20261011_141849` selected and freshly reloaded (no labels showing). Font 18+ in both terminals.
3. Editor tabs open in this order: `common/schema.py` at line 99, `synth/make_pairs.py` at line 27, `train/train.py` at line 157, `draft/judge.py` at line 91.
4. README open in a tab, scrolled to the top.
5. Phone: unlocked, Messages app closed, volume up, on speaker, in frame. Do Not Disturb off.

### 0:00 The problem (camera)
"I'm a solo founder. Writing is my bottleneck. And everything I write with an LLM sounds like the LLM: headers, bold, 'night and day', a thousand words I would never post. I stopped using them for content. The fix isn't a better prompt. It's two things: get the substance out of me by talking, and put the voice into a model that has only ever read me. I called it LoRAkeet. A LoRA that mimics you."

### 0:30 The stack (README, top)
"Dial gives it a phone number and runs the voice agent, so the interviewer is just a system prompt. Claude does the three jobs that need reasoning: turning speech into notes, synthesizing training pairs backwards from my own posts, and judging drafts for claims I never made. And Tinker trains a LoRA on Qwen 35B in two minutes per person, and warm-starts it in one. Without Tinker this is a prompt. Without Dial it's a web form."

### 1:00 LIVE 0:00 (Terminal 2)
Press Enter on the training command. As the first lines print:
"This is the post I approved from this morning's interview. I'm training the model on it right now, eight steps from the last checkpoint, about a minute. Leave it."
What you'll see: `[approve] recorded ...`, `[approve] warm-starting from tinker://...`, then `step loss` lines.

### 1:10 LIVE 0:10 (phone on speaker, Terminal 1 behind it)
"It's a phone number. I call it."
Dial +1 572 228 3645. It rings within a few seconds. It greets you by name and asks what you want to write about.
You: "Why I built a voice agent that writes in my voice, today, at the hackathon."
It mirrors a fragment and asks who the post is for.
You: "Founders on Twitter."
Hang up. (LIVE 0:40.)
"A real one runs four minutes. It's writing from that now. About forty seconds."
What you'll see in Terminal 1 over the next 40 s: `In-Progress`, `Completed`, `transcript at runs/run_...`, `wrote ... outline.json`, candidate lines, `imessage: sent`.

### 1:45 LIVE 0:45 (Chrome, compare page)
"This is this morning's full call, about why I started the company. Two drafts. One is Claude. One is the model that has only ever read me. Which is me?"
Hold for four seconds. Click Reveal.
Point at A: "Claude. 631 words, headers, ten claims I never made." Read the caption under it.
Point at B: "The fine-tune. 204 words. Zero. And every sentence is mine."
If Terminal 1 has printed `imessage: sent` by now, say: "And the post from the call I just made is on my phone."

### 2:25 LIVE 1:25 (phone)
Pick it up. Open Messages. The message starts "LoRAkeet draft:". Scroll it for two seconds.
"That's from the call I made a minute ago. I can copy it straight into a post."

### 2:45 LIVE 1:45 (Terminal 2)
`TRAINING DONE in 1.0 min` and `SAMPLER PATH` are on screen. Point at them.
"Training's done. That's a new checkpoint with this morning's post in it."
Run `./demo_sample.sh runs/run_20261011_141849`.
"And that's the model writing the next one." Let two or three lines stream, then move on.

### 3:00 Code (editor, four tabs, one line each)
Tab 1, `common/schema.py` line 99: "One function renders the prompt, for training and for sampling. Notes and at most two lines I said. Never prose. Claude's structure can't leak onto the page."
Tab 2, `synth/make_pairs.py` line 27: "Training data is synthesized backwards. For every real tweet or post, Claude writes the spoken notes I'd have given before writing it. Notes in, my text out. Ninety-five pairs from my own corpus."
Tab 3, `train/train.py` line 157: "The Tinker config: base model, rank 32, two epochs. And `load_checkpoint_path`, which is how every approved post warm-starts the model instead of training from scratch."
Tab 4, `draft/judge.py` line 91: "The judge: likeness to my real posts, and every claim not in the notes. Every change today was measured against it, including the ones that made it worse. Those are in the README too."

### 4:00 Close (camera)
"Call a number. Talk. Hang up. Get a post that's yours, with nothing in it you didn't say. And every post you approve makes the next one more you. Two minutes per person on Tinker."

### If
- No ring in 15 s: hang up, redial once. Still nothing: "carrier gremlins", skip to the compare page, and at LIVE 1:25 show the Messages thread from the earlier call instead.
- Nothing in Messages by LIVE 1:25: say "it's on disk", keep going; show it after the sample with `cat runs/<newest>/drafts/finetune.md`.
- Training not done by LIVE 1:45: "still training", skip the sample, go to code; come back to Terminal 2 before the close if it's done.
