# Duckets integration — tinker-ai-skill as the dataset front-end of the house training stack

Fork of drost-ai/tinker-ai-skill (Migel Tissera), integrated 2026-11-07 by Juno.
Upstream stays the pipeline; this file is the house layer. Where this file and
upstream SKILL.md disagree, THIS file wins for Duckets' runs.

## Verdict (why this shape)

Not a replacement for the local stack — a front-end graft:

- **Adopt from upstream:** transcript collection, per-session anonymization
  (fail-closed), renderer-true slicing with compaction bridges, tool
  definitions on every row, thinking preserved in history, seeded whole-session
  holdout, base-vs-checkpoint behavioral sampling.
- **Keep ours:** local training stays the default (MLX-LM-LoRA for the text
  loop, Unsloth for vision — the RSI dual-engine rule). Nothing trains outside
  the RSI program rules or without Duckets' direct order. ≤35B, ever.
- **Tinker (cloud) is an opt-in lane**, per run, with Duckets' explicit yes:
  it costs money per token and session data leaves the house (anonymized).
  Its value: LoRA on models too big/slow for a Mac night window, without
  occupying the Mac. The pipeline stops before any spend; the smoke test
  prints token count and cost first. A TINKER_API_KEY lives only in `.env`
  (git-ignored), set by Duckets through secure entry — never in chat, code,
  or this repo.

## Data sources (ours)

Upstream expects Claude Code JSONL under `~/.claude/projects/`. Ours, in
priority order:

1. **DuckBot session corpora** — the merged dataset family's sources
   (Mac `~/Documents/datasets/merged/`, archive on Sabrent
   `DuckBot-Datasets-Archive-2026-10-11/`). Existing cleaned sets can enter
   at the normalize step via the message-schema bridge
   (`references/claude-code-transcripts.md`: emit the schema, continue from
   step 2).
2. **Hermes session exports** from the Mac (`~/.hermes/state.db` sessions) —
   export to the same message schema first; never train on raw DB dumps.
3. **Claude Code transcripts**, if/when they exist on the Mac — raise
   `cleanupPeriodDays` to 3650 in `~/.claude/settings.json` first (upstream
   step 0).

Hard privacy rule (RSI): **no private/personal info in any set.** The
anonymizer is mandatory, not a step you can skip for local runs either —
run it, read its CLEAN verdict, and eyeball sample rows before ANY training,
local or cloud. `identity.duckets.json` in this repo is built from public
profile facts only; extend it from the Mac before a real run.

## House gates after any training run (local or Tinker)

Upstream's step 7 (behavioral sampling) is necessary, not sufficient. A
candidate is not done until ALL of these pass:

1. **Naming:** `<model>-DuckBot-<code>` — never invent another scheme.
2. **Live-harness gate:** the candidate is judged in the real harness it
   will serve in — transcript-verified model identity, live tool-call smoke,
   head-to-head against the incumbent serving artifact. A score for the base
   or the adapter is never a score for the shipped thing: gate the fused,
   loaded artifact under its own name.
3. **LM Studio rule:** we only use LM Studio. The exported model must load
   and serve in LM Studio (GGUF from step 9, or MLX where the engine
   supports it) before it can replace anything. A failed test never costs
   the old working model.
4. **Lineage:** a version earns its number only by beating its parent on
   held-out eval by the pre-set margin — the upstream seeded whole-session
   holdout is the eval source for transcript-trained runs.
5. **Data custody:** datasets are backed up to the Sabrent RSI tree
   (`/Volumes/Sabrent/datasets/rsi/`) before training; they outlive the
   models. Never trim or edit a dataset to solve a memory problem — adjust
   the training config instead.

## Serving note

Upstream serves with llama.cpp. Ours serves with **LM Studio**: use step 9's
GGUF + quantize, then import into LM Studio and serve with the
training-matched template and sampling settings from `serve_gguf.sh`
translated to the LM Studio preset. Template drift between training and
serving is the exact failure upstream warns about — carry the flags over.

## Status

- 2026-11-07: fork cloned, venv built, synthetic dry-run phase 1 verified
  (sample → collect → normalize → anonymize, anonymizer verdict CLEAN).
  build_dataset phase runs once `tinker-cookbook` finishes installing
  (renderer/tokenizer dependency). No real transcripts processed yet; no
  Tinker key set; no spend.
