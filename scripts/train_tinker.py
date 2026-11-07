#!/usr/bin/env python3
"""Step 6: LoRA supervised fine-tuning on Tinker with the tinker-cookbook trainer.

Default action is an OFFLINE smoke test: it checks the dataset against its build
report (same model, renderer, tokenizer and effort; unchanged files), renders every
row, prints one decoded row so you can read what the model will see, and reports
steps, tokens and the resolved configuration. Nothing talks to Tinker until you
pass --train.

Learning rate: `hyperparam_utils.get_lr(model, is_lora=True)` is used when Tinker has
calibrated it (Qwen and Llama families). For other models (Inkling, Nemotron,
DeepSeek, Kimi, GLM) the helper raises on purpose: pass --learning-rate explicitly.
See references/tinker-training.md for the evidence behind the suggested ranges.

Weights & Biases: on by default; put WANDB_API_KEY in .env (wandb.ai/authorize) or
pass --no-wandb. The full configuration, including the dataset builder, is logged
with the run.

Usage:
    python scripts/train_tinker.py --data-dir ./data --name qwen3.6-35b-a3b-64k                 # smoke test
    python scripts/train_tinker.py --data-dir ./data --name qwen3.6-35b-a3b-64k --train         # launch
    python scripts/train_tinker.py --data-dir ./data --name inkling-2-small-64k --learning-rate 1e-4 --lora-rank 8 --train
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import chz
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_dotenv, make_renderer, sha256_file, supervised_example  # noqa: E402

load_dotenv()
STABLE_PROVENANCE_KEYS = ("model", "renderer", "renderer_class", "tokenizer", "preserve_thinking", "accepts_effort")


def load_report(data_dir, name):
    report_path = data_dir / "datasets" / f"{name}.report.json"
    if not report_path.exists():
        sys.exit(f"{report_path} not found; run build_dataset.py first")
    return json.loads(report_path.read_text())


def verify_build(data_dir, name, report, prov):
    """Fail before any service call if the data or the rendering changed since the build."""
    built = report["rendering"]
    for key in STABLE_PROVENANCE_KEYS:
        if built.get(key) != prov.get(key):
            raise SystemExit(f"rendering mismatch on {key!r}: dataset built with {built.get(key)!r}, now {prov.get(key)!r}")
    if built.get("tinker_cookbook") != prov.get("tinker_cookbook"):
        print(f"note: tinker-cookbook {built.get('tinker_cookbook')} at build time, {prov.get('tinker_cookbook')} now",
              file=sys.stderr)
    for split in ("train", "val"):
        got = sha256_file(data_dir / "datasets" / f"{name}.{split}.jsonl")
        if got != report["files_sha256"][split]:
            raise SystemExit(f"{split} split changed since the build; rebuild or restore it")


class JsonlDataset(SupervisedDataset):
    """Lazy, randomly addressable JSONL split: only the requested batch is rendered."""

    def __init__(self, path, batch_size, renderer, prov, effort, max_len, shuffle, tokenizer=None):
        self.path, self.batch_size, self.renderer, self.prov = Path(path), batch_size, renderer, prov
        self.effort, self.max_len, self.shuffle, self.tokenizer = effort, max_len, shuffle, tokenizer
        self.offsets = []
        with self.path.open("rb") as f:
            while True:
                offset = f.tell()
                if not f.readline():
                    break
                self.offsets.append(offset)
        self.order = list(range(len(self.offsets)))

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, seed=0):
        self.order = list(range(len(self.offsets)))
        if self.shuffle:                       # rows are written session by session
            random.Random(seed).shuffle(self.order)

    def rows(self, index):
        with self.path.open("rb") as f:
            for row in self.order[index * self.batch_size:(index + 1) * self.batch_size]:
                f.seek(self.offsets[row])
                line = json.loads(f.readline())
                yield line["messages"], line.get("tools") or []

    def datum(self, messages, tools):
        from tinker_cookbook.supervised.common import datum_from_model_input_weights
        model_input, weights = supervised_example(self.renderer, messages, tools, self.effort, self.prov)
        if model_input.length > self.max_len:
            raise ValueError(f"row renders to {model_input.length} tokens > {self.max_len}")
        if not any(w > 0 for w in weights[1:].tolist()):
            raise ValueError("row has no loss tokens")
        return datum_from_model_input_weights(model_input, weights, max_length=None, reduction="mean")

    def get_batch(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return [self.datum(m, t) for m, t in self.rows(index)]


@chz.chz
class TranscriptDatasetBuilder(SupervisedDatasetBuilder):
    data_dir: str
    name: str
    batch_size: int = 8
    effort: float | None = None

    def __call__(self):
        data_dir = Path(self.data_dir)
        report = load_report(data_dir, self.name)
        rendering = report["rendering"]
        tokenizer, renderer, prov = make_renderer(report["model"], rendering["renderer"], rendering["preserve_thinking"])
        verify_build(data_dir, self.name, report, prov)
        effort = self.effort if self.effort is not None else rendering.get("effort")
        if prov["accepts_effort"] and rendering.get("effort") is not None and effort != rendering["effort"]:
            raise SystemExit(f"dataset was rendered at effort {rendering['effort']}; train at the same value")
        common = dict(renderer=renderer, prov=prov, effort=effort, max_len=report["max_len"], tokenizer=tokenizer)
        train_ds = JsonlDataset(data_dir / "datasets" / f"{self.name}.train.jsonl", self.batch_size, shuffle=True, **common)
        val_ds = JsonlDataset(data_dir / "datasets" / f"{self.name}.val.jsonl", self.batch_size, shuffle=False, **common)
        return train_ds, (val_ds if len(val_ds) else None)   # the cookbook skips evaluation when val is None


def pick_learning_rate(model, explicit):
    if explicit is not None:
        return explicit, "explicit"
    from tinker_cookbook.hyperparam_utils import get_lr
    try:
        return get_lr(model, is_lora=True), "hyperparam_utils.get_lr"
    except NotImplementedError:
        sys.exit(f"Tinker has no calibrated learning rate for {model}; pass --learning-rate "
                 "(see references/tinker-training.md for ranges that worked)")


def smoke_test(builder, report, hp):
    train_ds, val_ds = builder()
    renderer, prov = train_ds.renderer, train_ds.prov
    stats, sample = {}, None
    for ds in [d for d in (train_ds, val_ds) if d is not None]:
        ds.set_epoch(0)
        lengths, loss = [], 0
        for b in range(len(ds)):
            for messages, tools in ds.rows(b):
                model_input, weights = supervised_example(renderer, messages, tools, ds.effort, prov)
                w = weights.tolist()
                assert model_input.length <= ds.max_len and any(x > 0 for x in w[1:]), (ds.path.name, b)
                lengths.append(model_input.length)
                loss += sum(1 for x in w[1:] if x > 0)
                if sample is None:
                    sample = ds.tokenizer.decode(model_input.to_ints()[:400])
        first = [d.model_input.length for d in ds.get_batch(0)] if len(ds) else []
        stats[ds.path.name] = {"rows": len(lengths), "batches": len(ds), "tokens": sum(lengths), "loss_tokens": loss,
                               "max_len": max(lengths) if lengths else 0, "first_batch_lengths": first}
    train_key = f"{builder.name}.train.jsonl"
    if f"{builder.name}.val.jsonl" not in stats:
        print("note: the validation split is empty; the trainer will skip held-out evaluation", file=sys.stderr)
    steps = stats[train_key]["batches"] * hp["epochs"]
    print("---- decoded head of the first training row (read it: system prompt, tools, user turn) ----")
    print(sample)
    print("---- resolved configuration ----")
    print(json.dumps({"model": report["model"], "rendering": prov, "dataset": builder.name,
                      **hp, "steps_per_epoch": stats[train_key]["batches"], "total_steps": steps,
                      "train_tokens_all_epochs": stats[train_key]["tokens"] * hp["epochs"],
                      "splits": stats, "offline": True}, indent=1))
    if steps < 100:
        print("note: fewer than 100 optimizer steps; Tinker's guidance is at least 100 (add epochs or lower batch size)")
    print("SMOKE TEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--name", required=True, help="dataset name from build_dataset.py (report file stem)")
    ap.add_argument("--learning-rate", type=float, help="peak LR; default: Tinker's calibrated get_lr when available")
    ap.add_argument("--lr-schedule", choices=["constant", "linear", "cosine"], default="linear")
    ap.add_argument("--lora-rank", type=int, default=32, help="32 is Tinker's default; 8 was enough for very large MoEs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8, help="rows per optimizer step")
    ap.add_argument("--effort", type=float, help="Inkling thinking effort; must equal the dataset build's")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--wandb-project", default="agent-transcript-sft")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--log-dir", type=Path, help="run directory (default: ./runs/<name>-<tag>)")
    ap.add_argument("--train", action="store_true", help="start the Tinker job (default: offline smoke test)")
    args = ap.parse_args()
    if args.lora_rank < 1 or args.epochs < 1 or args.batch_size < 1:
        ap.error("invalid hyperparameter")

    report = load_report(args.data_dir, args.name)
    lr, lr_source = pick_learning_rate(report["model"], args.learning_rate)
    hp = {"learning_rate": lr, "learning_rate_source": lr_source, "lr_schedule": args.lr_schedule,
          "lora_rank": args.lora_rank, "epochs": args.epochs, "batch_size": args.batch_size}
    builder = TranscriptDatasetBuilder(data_dir=str(args.data_dir), name=args.name,
                                       batch_size=args.batch_size, effort=args.effort)
    if not args.train:
        smoke_test(builder, report, hp)
        return

    tag = f"b{args.batch_size}-r{args.lora_rank}-lr{lr:g}-{args.lr_schedule}-e{args.epochs}"
    log_dir = args.log_dir or Path("./runs") / f"{args.name}-{tag}"
    if log_dir.exists():
        ap.error(f"{log_dir} exists; choose a new --log-dir")
    if not os.environ.get("TINKER_API_KEY"):
        ap.error("TINKER_API_KEY is not set (put it in .env)")
    wandb_project = None if args.no_wandb else args.wandb_project
    if wandb_project and not os.environ.get("WANDB_API_KEY"):
        ap.error("WANDB_API_KEY is not set (put it in .env or pass --no-wandb)")
    train_ds, _ = builder()                     # provenance + checksum checks before any service call

    from tinker_cookbook.supervised import train
    config = train.Config(
        model_name=report["model"], log_path=str(log_dir), dataset_builder=builder,
        recipe_name="agent_transcript_sft", renderer_name=train_ds.prov["renderer"],
        learning_rate=lr, lr_schedule=args.lr_schedule, num_epochs=args.epochs, lora_rank=args.lora_rank,
        eval_every=args.eval_every, save_every=args.save_every,
        wandb_project=wandb_project, wandb_name=f"{args.name}-{tag}-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    print(json.dumps({"model": config.model_name, "renderer_name": config.renderer_name, "rendering": train_ds.prov,
                      "log_path": config.log_path, "learning_rate": config.learning_rate,
                      "learning_rate_source": lr_source, "lr_schedule": config.lr_schedule,
                      "num_epochs": config.num_epochs, "lora_rank": config.lora_rank, "batch_size": args.batch_size,
                      "steps_per_epoch": len(train_ds), "eval_every": config.eval_every,
                      "save_every": config.save_every, "wandb_project": config.wandb_project,
                      "wandb_name": config.wandb_name}, indent=1))
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
