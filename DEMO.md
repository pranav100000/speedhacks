# Demo recording script (target 3:30 in the final cut)

The demo is a screen recording, not a live stage, so: record every segment for real, cut the dead time, speed up the long parts. Record at 1080p with system audio on; the phone call needs the phone on speaker next to the mic. Three windows, arranged before you hit record: Chrome tab 1 = http://localhost:8501 (compare page, demo run selected, labels hidden), Terminal 1 and Terminal 2 at the repo root, font 18+.

Run `./demo_preflight.sh` first. Every line must say OK.

Record in this order; it is also the edit order.

| Beat | Clock | On screen | What you say | What they see |
|---|---|---|---|---|
| 1. Hook | 0:00 - 0:20 | Compare tab, three columns, labels hidden | "Everything I write with an LLM sounds like the LLM. I built something that interviews me by voice and writes in my voice. One of these three columns is me. Hold that thought." | Three posts, no labels |
| 2. The call | 0:20 - 1:05 | Terminal 1: `uv run python -m interview.dial_call --to +15038287565 --corpus data/corpus.jsonl`, phone on speaker in frame | "It calls me." Phone rings, answer, do the whole interview (4 min). In the edit: keep the ring, the first two exchanges, then speed-ramp the rest with the transcript turns scrolling, land on `TRANSCRIPT WRITTEN`. | The phone, the call, the transcript arriving |
| 3. Notes, then train | 1:05 - 1:45 | Terminal 1: `cat runs/<demo>/outline.json`. Terminal 2: `./demo_train.sh runs/<demo> text` (approves `runs/<demo>/final.md`, the post you edited this morning) | "The drafter never writes sentences. It writes notes plus at most two lines I said. Claude's skeleton never reaches the page. The thing that writes my column is a LoRA on Qwen 35B, trained on Tinker on pairs synthesized backwards from my posts. And here's the loop: this is the post I edited and approved this morning. I'm training on it right now. Eight steps, warm start, about a minute. We'll come back to it." | Notes, then `[approve] recorded ... 77 pairs`, step losses ticking |
| 4. Which is me | 1:45 - 2:35 | Compare tab | "Vanilla Claude. Claude with my posts in context. The fine-tune. Which is me?" Hold the three columns for a 5-second beat so the viewer can read. Click Reveal. Point at vanilla: 631 words, headers, ten claims I never made. Point at the fine-tune: 204 words, zero. | Labels, voice score and invented-claim count under each |
| 5. We measured it | 2:40 - 3:10 | README table "What moved the needle" | "We built a judge: likeness to my real posts, and claims I never made. Every change today was measured on the same interview. The fine-tune went from a 5 with six invented claims to a 7 with one. Honest part: Claude with my posts still wins head to head. The gap is data. 711 sent emails yielded four usable ones." | The numbers |
| 6. It learned | 3:05 - 3:30 | Terminal 2 | Cut back to the training terminal as `TRAINING DONE in 1.0 min` prints (in the edit, this is real time; the minute of training ran under beats 4 and 5). Run `./demo_sample.sh runs/<demo>`. "That's the model that now includes this morning's post, writing the next one." | New checkpoint, a fresh paragraph streaming in |
| 7. Close | 3:30 - 3:45 | Compare tab | "Talk for three minutes, get a post that's yours, know which parts you actually said, and every post you approve makes the next one more you. Two minutes per user on Tinker." | |

Three-minute version: drop beat 5 and keep beat 2 to 25 seconds.

## Recording notes

- Beat 2 is the only segment with a hard dependency on timing: start the terminal command, then pick up the phone. If the call drops, re-record the segment; the earlier transcript is not reused in the video.
- Beats 3 to 6 are one continuous take if you can manage it: the training runs under beats 4 and 5 and finishes on camera. If it finishes early, pause on the compare page a little longer.
- Reset between takes: `cp data/sampler_demo.json data/sampler.json`, remove the `approved_*` line from `data/pairs.jsonl`, reload the compare tab.

## If something breaks

- Phone doesn't ring within 20 seconds or the audio is bad: hang up, say "carrier gremlins", skip to beat 3. The recording from this afternoon is already there. Second fallback: the browser bot on 7860.
- Training stalls in beat 3 or isn't done by beat 6: say "Tinker queue", show `runs/tinker/update_*/metrics.jsonl` from the rehearsal, go to close. Nothing else depends on it.
- Sampling in beat 6 slow: Ctrl-C after 15 seconds and close on the checkpoint line.
- Compare page not loading: `PYTHONPATH=. uv run streamlit run draft/compare.py` in a second terminal, 10 seconds.

## Before the demo

1. The demo run is `runs/run_20261011_141849`, the phone interview recorded at 14:18 (build vs GTM as a solo founder). Re-record only if you want a different topic.
2. Run outline, generate (all three), judge on it. Read all three drafts yourself. If the fine-tune draft is bad, run `generate --only finetune` again; sampling is cheap.
2b. Edit the draft you like into `runs/<demo>/final.md`. Real edits: that file is what trains live on stage. Then `cp data/sampler.json data/sampler_demo.json` so you can reset with `cp data/sampler_demo.json data/sampler.json` after each rehearsal.
3. Reload the compare tab so Reveal is hidden. Note which letter is which (the shuffle is seeded by run id, so it's fixed).
4. `./demo_preflight.sh`
