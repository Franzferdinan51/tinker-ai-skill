# Export and serve: LoRA download, merge, GGUF, llama.cpp

Training leaves you with checkpoints in Tinker storage. Getting a model you can run
locally is three steps: download the adapter, merge it into the base model, convert
and quantize to GGUF. `scripts/export_checkpoint.py`, `scripts/make_gguf.sh` and
`scripts/serve_gguf.sh` do them; this note is the why and the gotchas.

## 1. Pick the checkpoint and download it

`runs/<run>/checkpoints.jsonl` has one record per saved step:

```
{"name":"000100","batch":99,"epoch":0,
 "state_path":"tinker://<run>/weights/000100",
 "sampler_path":"tinker://<run>/sampler_weights/000100", ...}
```

- Choose by evaluation (`sample_checkpoint.py`, your own benchmark), not by the
  last step. Our best checkpoint was usually the end of epoch 1.
- Download the **training** path (`weights/<name>`). With tinker 0.38 and
  tinker-cookbook 0.6 sampler checkpoints cannot be downloaded; the training
  checkpoint is exported through a temporary encrypted external-weights
  checkpoint, which briefly starts a training session on the base model and is
  deleted afterwards. `weights.download` and `tinker checkpoint download` do this
  for you. If a download ever stalls on an archive build, save external weights
  explicitly from a training client (`save_weights_external`) and download that
  path; it took us 37 seconds where the archive route stalled for 40 minutes.
- Periodic checkpoints expire after 7 days by default; `final` is kept. Export
  before the one you want disappears.

```
python scripts/export_checkpoint.py --run-dir ./runs/<run> --checkpoint 000100 \
    --base-model Qwen/Qwen3.6-35B-A3B --out ./export/my-coder --trust-remote-code
```

## 2. Merge into the base model

`tinker_cookbook.weights.build_hf_model` merges `base + (alpha / r) * B @ A` for
every adapted matrix and writes a normal HuggingFace directory. Shard-by-shard
merging keeps peak memory near the largest shard; a 27B merge takes minutes on a
laptop, a 120B MoE took about 20 minutes and 220 GB of RAM. Output size equals the
base model in bf16.

What the cookbook handles for you: split and fused projections (Qwen3.5-family
linear attention trains separate `in_proj_q/k/v` adapters that the merge fuses
back), expert layouts for MoE families, the `backbone.*` prefix of Nemotron, and
quantized bases (INT4 Kimi, FP8 DeepSeek). The support matrix lives in the
cookbook's `tinker_cookbook/weights/README.md`.

After the merge, the script copies `generation_config.json` from the base when the
merge omitted it, and inspects the chat template (next section). Verify the merge
cheaply before converting: load the config, count shards, run one generation
through transformers or go straight to the GGUF smoke test.

**PEFT adapter instead of a merge** (`--peft`): a few hundred MB of LoRA matrices
for vLLM (`--enable-lora --lora-modules name=./peft`) or SGLang (`--lora-paths`).
Preferred when one base serves many fine-tunes. Not all families are supported by
the serving stacks (DeepSeek V3.1, Kimi-K2.6 at the time of writing).

## 3. The chat template must match training

The dataset was rendered with earlier-turn thinking kept in context. Stock chat
templates of hybrid-thinking models strip it from history by default:

| Family | Template switch | Pass at serving time |
|---|---|---|
| Qwen3.5 / 3.6 / 3.8 | `preserve_thinking` | `--chat-template-kwargs '{"preserve_thinking": true}'` |
| Nemotron 3 / 3.5 | `truncate_history_thinking` | `--chat-template-kwargs '{"truncate_history_thinking": false}'` |

Either pass the kwarg every time (what `serve_gguf.sh` does) or change the default
inside `chat_template.jinja` before converting, so the GGUF carries the training
behaviour and clients need no flags. llama-server additionally needs
`--reasoning-preserve` to send earlier reasoning back through the template, and the
client must echo `reasoning_content` in the history it sends. Also pass the
system prompt and tool definitions the rows were trained with.

## 4. GGUF conversion and quantization

```
LLAMA_CPP=~/src/llama.cpp scripts/make_gguf.sh ./export/my-coder/hf-merged ./export/my-coder/gguf my-coder Q4_K_M Q8_0
```

- Use a llama.cpp checkout that knows the architecture, build it, and use the
  `build/bin/llama-quantize` from the same commit as `convert_hf_to_gguf.py`.
  Package builds lag new architectures by weeks; a stale quantizer fails with
  `unknown model architecture`.
- Convert to BF16 (`OUTTYPE=bf16`), then quantize. Q8_0 and Q6_K are near-lossless;
  Q4_K_M is the usual local choice (about 4.9 bits per weight, 16.5 GB for 27B).
- Qwen3.5-family models carry a multi-token-prediction head. The script passes
  `--no-mtp` for the text model and, with `MTP=1`, exports the head separately for
  speculative decoding (`-md mtp.gguf --spec-type draft-mtp`). Note the MTP head
  is the base model's: after fine-tuning its acceptance rate can drop; measure it.
  An EAGLE-3 draft trained on the fine-tuned model's own outputs recovered 1.8x
  for us, but that is a separate project.
- Vision models: the script exports `mmproj-<name>-F16.gguf` when the config has a
  vision tower; pass it with `--mmproj` to keep image input.
- Files over 50 GB cannot be uploaded to HuggingFace: `SPLIT_MAX_SIZE=45G`
  splits the conversion and `--keep-split` keeps the quantized output split.

## 5. Serve with llama.cpp

```
scripts/serve_gguf.sh ./export/my-coder/gguf/my-coder-Q4_K_M.gguf
CTX=131072 KV=q8_0 MMPROJ=./export/my-coder/gguf/mmproj-my-coder-F16.gguf scripts/serve_gguf.sh ...
```

Expands to `llama-server -m ... -ngl 99 -c 65536 -fa on --jinja --reasoning-preserve
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --chat-template-kwargs '{"preserve_thinking": true}'`.
Then point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`.

Settings that came from failures:

- **Sampling**: temperature 1.0, top-p 0.95, top-k 20, min-p 0. llama.cpp's default
  min-p of 0.05 and a serving proxy's top-p of 1.0 both produced tail-sampling
  garble on a model that was fine on Tinker.
- **Context**: serve at least the training window (64K). Reasoning loops appeared
  only past about 90K tokens, and only with a q4_0 KV cache; q8_0 fixed it. Keep the
  harness's compaction threshold at or below the training window.
- **Smoke test** before you trust it: one prompt with a tool available; check that
  `reasoning_content` is non-empty and well-formed, the tool call parses, and the
  answer follows the system prompt. Then run your benchmark against the base
  model's GGUF built the same way (an imatrix quant from someone else is not a
  same-recipe baseline).
- The same GGUF loads in LM Studio and Ollama; the template kwargs and sampling
  defaults are the parts you must carry over by hand.

## 6. Publish (optional)

`hf upload <you>/<model> ./export/my-coder/hf-merged .` for the merged weights and a
second repository for the GGUFs with the mmproj and draft files. Use the `hf`
CLI from the same environment (`huggingface_hub>=1.3`); an older system CLI hung
on large files for us. Write the model card with the base model, the license it
inherits, the context window, the sampling settings and the template kwarg; leave
your dataset private.
