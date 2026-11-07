# Training on Tinker: settings that worked and why

Tinker runs LoRA fine-tuning as a service: you write the training loop (here, the
cookbook's `supervised.train`), Tinker does the GPU work, bills per token, and
serves checkpoints for sampling. Docs: https://tinker-docs.thinkingmachines.ai

## Prerequisites

- `TINKER_API_KEY` from the Tinker console; `WANDB_API_KEY` from https://wandb.ai/authorize.
  Put both in `.env` (git-ignored). The trainer refuses to launch without them unless
  `--no-wandb`.
- `tinker-cookbook>=0.6.0` (pulls `tinker`, `torch>=2.10`, `tml-renderers`).
- The model list and prices: https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/.
  Base entries are 64K context; `:peft:262144` entries are 256K and cost more to train.

## Renderer and tokenizer: resolve, never hardcode

```python
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
renderer = get_renderer(model_info.get_recommended_renderer_name(model), get_tokenizer(model))
```

Two traps the scripts handle for you:

- Hybrid-thinking renderers (Qwen3.5/3.6, Nemotron) default to
  `strip_thinking_from_history=True`: earlier assistant turns lose their thinking and
  become empty `<think></think>` targets, and the renderer loses the "extension
  property" so the cookbook splits each conversation into one example per assistant
  turn. `common.make_renderer` rebuilds such renderers with
  `strip_thinking_from_history=False`. Some families ship a `*_preserve_thinking`
  renderer name instead (Kimi K2.6, DeepSeek V4.1); pass it with `--renderer`.
- Inkling models (Thinking Machines) use their own `tml_v0` renderer and tokenizer
  (Inkling 2 uses a different vocabulary from Inkling and Inkling-Small) and need an
  explicit thinking effort on every rendered row (`--effort`, default 0.9 = high).
  Render training data, sample and evaluate at the same effort.

## Learning rate

- Qwen and Llama: `hyperparam_utils.get_lr(model, is_lora=True)` is calibrated and
  the trainer uses it by default. It scales a 5e-4 LoRA base by a weak hidden-size
  factor, so it lands around 4e-4 to 5e-4 for most Qwen sizes.
- Other families raise `NotImplementedError` on purpose; pass `--learning-rate`.
- Tinker's own SFT sweeps (tulu3, batch 128, 780 steps, cookbook
  `recipes/chat_sl/results/sft_sweep.md`): the largest MoEs (Qwen3.5-397B-A17B,
  Nemotron-3-Ultra-550B, DeepSeek-V3.1, Kimi-K2.6) are best at 3e-4 with 1e-4 within
  noise; 1e-3 is clearly worse; 3e-3 diverges. Smaller models (Qwen3 8B-30B) tolerate
  1e-3. "LR matters more than rank."
- Our experience on agentic transcripts at batch 8 rows (~400K tokens per step): a
  27B dense model trained at the formula's 4.65e-4 regressed on a behavioural
  benchmark while 1e-4 matched the base model; big MoEs trained well at 1e-4 to 1e-3.
  When unsure, run 1e-4 and 3e-4 for one epoch each and compare behaviour.
- Divergence signature: validation NLL dips then rises in the first 20-40 steps.
  Stop and lower the LR.
- Linear decay to zero is the default and gives built-in annealing.

## LoRA rank, batch, epochs

- Rank 32 is Tinker's default. For very large MoEs rank 8 was enough and keeps the
  adapter small (rank is already a billion parameters on a 600B model); LR is
  independent of rank, so sweep LR, not rank.
- Batch is measured in rows here; with 64K rows that is a large token batch. Aim for
  at least 100 optimizer steps in total (`steps = rows / batch x epochs`); with a
  few hundred rows that means 2 epochs at batch 8.
- Save and evaluate every 20 steps; pick the checkpoint by an eval, not by the final
  step. Our best checkpoint was usually around the end of epoch 1.

## Cost

Training cost = train tokens x train price per million (per model on the pricing
page). The smoke test prints `train_tokens_all_epochs`; periodic evaluation adds
forward passes over the validation split at the training price. Sampling is billed
separately (prefill + sampled tokens). Checkpoint storage is per GB-month.

## Weights & Biases

`train_tinker.py` logs to the project in `--wandb-project` with a run name made of
the dataset name and hyperparameters; the cookbook logs the full config including
the dataset builder. Confirm the first rows arrive, then watch `train_mean_nll`,
`test/nll` and `learning_rate`.

## Evaluating and exporting

- Sample a few prompts from base and from checkpoints with `sample_checkpoint.py`.
  Judge behaviour: tool calls parse, thinking is present and non-empty, the system
  prompt is followed, no loops late in long contexts.
- Tinker's OpenAI-compatible endpoint takes any `tinker://.../sampler_weights/...`
  path as the model name, so existing eval harnesses work:
  base URL `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`,
  API key = your Tinker key. Inkling models take `reasoning_effort` as a preset
  (`high`) or a float.
- Export: `tinker checkpoint download <tinker-path> -o ./adapter`, then merge with
  `tinker_cookbook.weights.build_hf_model(base_model, adapter_path, output_path)`
  for GGUF/vLLM serving. Inkling 2 checkpoints become downloadable when its weights
  are released.

## Operational rules from the cookbook's AGENTS.md

- Submit Tinker calls back-to-back and await futures; never wrap them in your own
  timeouts or retry loops; the SDK retries and detects stuck requests.
- Create a new sampling client after saving weights; a stale client samples old
  weights.
- Print the resolved config before launch, decode a few rows and read them, report
  train/val counts and overlap, commit before each run and record the hash.
