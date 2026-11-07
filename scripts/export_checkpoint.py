#!/usr/bin/env python3
"""Step 8: download the trained LoRA from Tinker and merge it into the base model.

Produces a standalone HuggingFace model directory (bf16 safetensors) that
transformers, vLLM, SGLang and llama.cpp's converter all understand, and
optionally a PEFT adapter for hot-swapping on a served base model.

Which checkpoint path to use (this matters with tinker>=0.38 / cookbook>=0.6):
  * runs/<run>/checkpoints.jsonl has, per saved step, a `state_path`
    (tinker://<run>/weights/<name>) and a `sampler_path` (.../sampler_weights/<name>).
  * Sampler checkpoints cannot be downloaded. The training checkpoint can: Tinker
    exports it through a temporary, encrypted external-weights checkpoint (this
    briefly starts a training session on the base model) and deletes the export
    afterwards. `weights.download` does all of that.
  * Periodic checkpoints expire (7 days by default); `final` is kept. Export the
    checkpoint you chose by evaluation before it expires.

Memory and disk: the merge streams shard by shard, so peak RAM is roughly the
largest base shard plus the adapter; expect a 27B merge to take minutes and the
output to be the size of the base model in bf16 (about 2 bytes per parameter).
The base model is fetched from the HuggingFace Hub (or use a local directory).

Usage:
    python scripts/export_checkpoint.py --run-dir ./runs/<run> --base-model Qwen/Qwen3.6-35B-A3B --out ./export/my-model
    python scripts/export_checkpoint.py --tinker-path tinker://<run>/weights/final --base-model ... --out ... --peft
    python scripts/export_checkpoint.py --run-dir ./runs/<run> --checkpoint 000100 --base-model ... --out ... --dry-run
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_dotenv  # noqa: E402

load_dotenv()


def resolve_tinker_path(args):
    if args.tinker_path:
        return args.tinker_path, None
    records = [json.loads(l) for l in (args.run_dir / "checkpoints.jsonl").read_text().splitlines() if l.strip()]
    if not records:
        sys.exit("checkpoints.jsonl is empty")
    names = [r["name"] for r in records]
    match = [r for r in records if r["name"] == args.checkpoint]
    if not match:
        sys.exit(f"no checkpoint named {args.checkpoint!r}; available: {names}")
    rec = match[-1]
    path = rec.get("state_path")
    if not path:
        sys.exit(f"checkpoint {args.checkpoint!r} has no training state path (only {list(rec)}); "
                 "sampler-only checkpoints cannot be downloaded")
    return path, rec


def long_timeouts():
    """The export endpoint can hold a request open while Tinker builds the archive."""
    try:
        import httpx
        import tinker._base_client as b
        import tinker._constants as c
        c.DEFAULT_TIMEOUT = b.DEFAULT_TIMEOUT = httpx.Timeout(timeout=1800.0, connect=10.0)
    except Exception:
        pass


def base_file(base_model, name):
    """A file of the base model: from a local directory or from the Hub cache."""
    local = Path(base_model)
    if local.is_dir():
        p = local / name
        return p if p.exists() else None
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(base_model, name))
    except Exception:
        return None


def template_hint(merged_dir):
    """Hybrid-thinking chat templates strip earlier reasoning unless told not to; our
    training kept it. Tell the user which kwarg restores the training conditioning."""
    tpl = merged_dir / "chat_template.jinja"
    if not tpl.exists():
        cfg = merged_dir / "tokenizer_config.json"
        text = json.loads(cfg.read_text()).get("chat_template", "") if cfg.exists() else ""
    else:
        text = tpl.read_text()
    if not isinstance(text, str):
        return None
    for key, kwargs in (("preserve_thinking", '{"preserve_thinking": true}'),
                        ("truncate_history_thinking", '{"truncate_history_thinking": false}')):
        if key in text:
            return (f"chat template has `{key}`; serve with --chat-template-kwargs '{kwargs}' "
                    f"(and --reasoning-preserve in llama-server) so earlier-turn thinking is rendered as in training")
    return "chat template has no history-thinking switch; check how it renders reasoning in earlier assistant turns"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="training run directory (reads checkpoints.jsonl)")
    src.add_argument("--tinker-path", help="tinker://<run>/weights/<name> or .../external_weights/<name>")
    ap.add_argument("--checkpoint", default="final", help="checkpoint name in checkpoints.jsonl (default: final)")
    ap.add_argument("--base-model", required=True, help="HF model id or local directory the LoRA was trained on")
    ap.add_argument("--out", type=Path, required=True, help="output directory (created; must be new)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--merge-strategy", default="auto", choices=["auto", "shard", "full"])
    ap.add_argument("--trust-remote-code", action="store_true", help="needed for some architectures (Qwen3.5 family)")
    ap.add_argument("--peft", action="store_true", help="also build a PEFT adapter (vLLM --lora-modules, SGLang)")
    ap.add_argument("--no-merge", action="store_true", help="download (and --peft) only; skip the merged model")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without contacting Tinker")
    args = ap.parse_args()

    tinker_path, rec = resolve_tinker_path(args)
    adapter_dir, merged_dir, peft_dir = args.out / "adapter-tinker", args.out / "hf-merged", args.out / "peft"
    plan = {"tinker_path": tinker_path, "checkpoint_record": rec, "base_model": args.base_model,
            "adapter_dir": str(adapter_dir), "merged_dir": None if args.no_merge else str(merged_dir),
            "peft_dir": str(peft_dir) if args.peft else None, "dtype": args.dtype, "merge_strategy": args.merge_strategy}
    print(json.dumps(plan, indent=1))
    if args.dry_run:
        print("dry run: nothing downloaded")
        return
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} exists and is not empty; choose a new --out")
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY is not set (put it in .env)")
    args.out.mkdir(parents=True, exist_ok=True)

    long_timeouts()
    from tinker_cookbook import weights

    print(f"STEP 1/3 download {tinker_path} -> {adapter_dir}", flush=True)
    got = Path(weights.download(tinker_path=tinker_path, output_dir=str(adapter_dir)))
    for p in sorted(got.rglob("*")):
        if p.is_file():
            print(f"  {p.stat().st_size:>14,}  {p.relative_to(got)}")
    if not any(got.rglob("adapter_config.json")):
        sys.exit("download did not produce adapter_config.json; if files end in .age, decrypt them with `age -d` first")
    adapter_root = next(p.parent for p in got.rglob("adapter_config.json"))

    if args.peft:
        print(f"STEP 2/3 PEFT adapter -> {peft_dir}", flush=True)
        weights.build_lora_adapter(base_model=args.base_model, adapter_path=str(adapter_root),
                                   output_path=str(peft_dir), trust_remote_code=args.trust_remote_code or None)
    if args.no_merge:
        print("DONE (no merge requested)")
        return

    print(f"STEP 3/3 merge into {args.base_model} -> {merged_dir} ({args.dtype}, {args.merge_strategy})", flush=True)
    weights.build_hf_model(base_model=args.base_model, adapter_path=str(adapter_root), output_path=str(merged_dir),
                           dtype=args.dtype, merge_strategy=args.merge_strategy,
                           trust_remote_code=args.trust_remote_code or None)
    for name in ("generation_config.json", "chat_template.jinja"):
        if not (merged_dir / name).exists():
            src_file = base_file(args.base_model, name)
            if src_file:
                shutil.copy2(src_file, merged_dir / name)
                print(f"  copied {name} from the base model")
    shards = sorted(p.name for p in merged_dir.glob("*.safetensors"))
    print(f"  merged model: {len(shards)} safetensors shard(s), "
          f"{sum(p.stat().st_size for p in merged_dir.glob('*.safetensors')) / 1e9:.1f} GB")
    hint = template_hint(merged_dir)
    if hint:
        print(f"  note: {hint}")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", args.out.name)
    print("DONE. Next: scripts/make_gguf.sh", str(merged_dir), str(args.out / "gguf"), name, "Q4_K_M")


if __name__ == "__main__":
    main()
