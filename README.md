# tinker-ai-skill

An [Agent Skill](https://agentskills.io) that teaches a coding agent (Claude Code,
Codex, OpenCode, Cursor and any other agentskills.io-compatible agent) how to turn
your own agent session transcripts into a supervised fine-tuning dataset and train a
LoRA on [Tinker](https://thinkingmachines.ai/tinker) with the
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

It packages the pipeline we used to ship the Tess family of agentic coding models:

1. collect Claude Code session transcripts into a dataset location;
2. clean them: drop harness noise, replace credentials by value, de-identify people
   and paths with a per-session randomized identity;
3. slice sessions to the model's context length at safe tool boundaries;
4. prepend a deterministic compaction context to every slice after the first;
5. split train/val by whole session;
6. fine-tune on Tinker, with Weights & Biases connected, then sample checkpoints.

Everything is plain Python on top of tinker-cookbook. The dataset steps run offline;
only training and sampling call the Tinker service.

## Install the skill

Clone (or copy) this repository into your agent's skills location. The directory
name must be `tinker-ai-skill`, which `git clone` gives you by default:

| Agent | Personal (all projects) | Project (this repo) |
|---|---|---|
| Claude Code | `~/.claude/skills/tinker-ai-skill/` | `.claude/skills/tinker-ai-skill/` |
| OpenAI Codex | `~/.agents/skills/tinker-ai-skill/` | `.agents/skills/tinker-ai-skill/` |
| OpenCode | `~/.config/opencode/skills/tinker-ai-skill/` | `.opencode/skills/tinker-ai-skill/` |

OpenCode also reads `.claude/skills` and `.agents/skills`, so one project copy
serves all three. Other agents: see your agent's documentation for its skills
directory; the format is the same `SKILL.md` + `scripts/` + `references/` layout.

```
# Claude Code
git clone https://github.com/drost-ai/tinker-ai-skill.git ~/.claude/skills/tinker-ai-skill
# Codex
git clone https://github.com/drost-ai/tinker-ai-skill.git ~/.agents/skills/tinker-ai-skill
# OpenCode
git clone https://github.com/drost-ai/tinker-ai-skill.git ~/.config/opencode/skills/tinker-ai-skill
```

Inside a repository, use the project paths from the table instead so the skill
travels with the code.

Then ask your agent, for example: "fine-tune Qwen3.6-35B-A3B on my Claude Code
sessions from the last three months", or invoke it directly with `/tinker-ai-skill`
in Claude Code (Codex and OpenCode pick it up from its description).

## What you need

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip).
- A Tinker account and API key: https://tinker.thinkingmachines.ai (training and
  sampling are billed per token; prices at
  https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/).
- Optional but recommended: a Weights & Biases API key, https://wandb.ai/authorize.
- Your transcripts. Claude Code stores them under `~/.claude/projects/` and deletes
  them after 30 days by default; set `"cleanupPeriodDays": 3650` in
  `~/.claude/settings.json` first.

```
cd tinker-ai-skill
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env          # fill in TINKER_API_KEY and WANDB_API_KEY
```

## Try it in five minutes on synthetic data

```
python scripts/make_sample_transcript.py --out ./demo-claude-projects --turns 14 --long-results
python scripts/collect_sessions.py   --claude-dir ./demo-claude-projects --data-dir ./demo-data --min-bytes 1000
python scripts/normalize_sessions.py --data-dir ./demo-data
python scripts/anonymize_sessions.py --data-dir ./demo-data --name "Taylor Quintrell"
python scripts/build_dataset.py      --data-dir ./demo-data --model Qwen/Qwen3.6-35B-A3B --max-len 8192 \
                                     --system-prompt-file assets/system-prompt.md
python scripts/train_tinker.py       --data-dir ./demo-data --name qwen3.6-35b-a3b-8k        # offline smoke test
```

The last command prints a decoded training row and the token count. Add `--train`
to launch a real run (this one is tiny and cheap; a real dataset of a few hundred
64K rows costs on the order of $50-$150 per epoch on a mid-size MoE).

## Then on your own sessions

Follow `SKILL.md` step by step, or let your agent drive. The important human
decisions are: which projects to exclude, what goes in the identity file
(`identity.example.json`), which model and context length, the system prompt and
tool definitions of the harness you will serve the model in, and the learning rate
when Tinker has no calibrated default for the model.

## Layout

```
SKILL.md                       instructions the agent follows
README.md                      this file
scripts/collect_sessions.py    step 1
scripts/normalize_sessions.py  step 2a
scripts/anonymize_sessions.py  step 2b
scripts/build_dataset.py       steps 3-5
scripts/train_tinker.py        step 6
scripts/sample_checkpoint.py   step 7
scripts/make_sample_transcript.py   synthetic transcript for a dry run
scripts/common.py              shared rendering helpers (builder and trainer render identically)
references/                    format, anonymization, chunking, training, lessons learned
assets/system-prompt.md        a generic agent system prompt to start from
```

## Data safety

`data/`, `demo-*/`, `runs/`, `wandb/` and `.env` are git-ignored. Review anonymized
rows before training, and never publish a dataset built from transcripts that
belong to someone else without their consent.

## Author and license

Written by Migel Tissera. MIT license. Tinker and tinker-cookbook are products of
Thinking Machines Lab; this skill is an independent guide.
