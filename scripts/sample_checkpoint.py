#!/usr/bin/env python3
"""Sample from the base model or a fine-tuned checkpoint through Tinker, rendering
the prompt exactly as the training rows were rendered (same renderer, same system
prompt, same tool declaration, same effort). Use it to compare base vs checkpoint on
a few prompts before you trust the loss curve: a falling NLL does not prove the
model still behaves well.

Checkpoint paths come from the run's checkpoints.jsonl (tinker://.../sampler_weights/...).
Without --checkpoint the base model is sampled.

You can also point any OpenAI client at Tinker's OpenAI-compatible endpoint with the
checkpoint path as the model name; see references/tinker-training.md.

Usage:
    python scripts/sample_checkpoint.py --name qwen3.6-35b-a3b-64k --data-dir ./data \
        --prompt "List the files in the current directory and tell me what this project does."
    python scripts/sample_checkpoint.py --name ... --checkpoint tinker://<run>/sampler_weights/000100 --prompt "..."
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import conversation_with_tools, load_dotenv, make_renderer  # noqa: E402

load_dotenv()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--name", required=True, help="dataset name; its report supplies model, renderer, tools and effort")
    ap.add_argument("--checkpoint", help="tinker://... sampler weights path (default: base model)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system-prompt-file", type=Path, help="defaults to the first training row's system prompt")
    ap.add_argument("--cwd", default="/Users/dev/example-project", help="working directory to put in the system prompt")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-samples", type=int, default=1)
    args = ap.parse_args()

    report = json.loads((args.data_dir / "datasets" / f"{args.name}.report.json").read_text())
    rendering = report["rendering"]
    _, renderer, prov = make_renderer(report["model"], rendering["renderer"], rendering["preserve_thinking"])
    first = json.loads(next(open(args.data_dir / "datasets" / f"{args.name}.train.jsonl", encoding="utf-8")))
    tools = first.get("tools") or []
    if args.system_prompt_file:
        system = args.system_prompt_file.read_text().strip() + f"\n\n# Environment\n\nWorking directory: {args.cwd}"
    elif first["messages"][0]["role"] == "system":
        system = first["messages"][0]["content"].split("\n\n# Environment")[0]
        system += f"\n\n# Environment\n\nWorking directory: {args.cwd}"
    else:
        system = ""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": args.prompt}]
    conv = conversation_with_tools(renderer, messages, tools)
    kwargs = {"effort": rendering.get("effort")} if prov["accepts_effort"] and rendering.get("effort") is not None else {}
    prompt = renderer.build_generation_prompt(conv, **kwargs)

    import tinker
    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=report["model"], model_path=args.checkpoint).result()
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                                   stop=renderer.get_stop_sequences())
    result = client.sample(prompt=prompt, num_samples=args.num_samples, sampling_params=params).result()
    print(f"model={report['model']} checkpoint={args.checkpoint or '(base)'} prompt_tokens={prompt.length}")
    for i, seq in enumerate(result.sequences):
        message, termination = renderer.parse_response(seq.tokens)
        print(f"\n===== sample {i + 1}: {len(seq.tokens)} tokens, termination={termination} =====")
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "thinking":
                    print("[thinking]", part["thinking"][:2000])
                elif part.get("type") == "text":
                    print(part["text"])
        elif content:
            print(content)
        for tc in message.get("tool_calls") or []:
            print(f"[tool call] {tc.function.name} {tc.function.arguments[:500]}")


if __name__ == "__main__":
    main()
