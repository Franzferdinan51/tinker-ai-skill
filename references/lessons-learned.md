# Lessons learned shipping models from agent transcripts

Each of these cost a training run or a release. The scripts encode the fix; this is
the why.

1. **A constant anonymized world is a feature the model learns.** Mapping every
   session to `/home/user/agents/...` produced a model that asserted that world over
   the real one: probing fake paths, calling the user by the fake name, inventing a
   fork. Fix: per-session randomized identity and container names, real repo names.

2. **Healthy loss can hide a dead model.** Two "collapses" turned out not to be
   optimisation problems at all: one was a renderer that stripped thinking from
   history (training empty think blocks), one was a serving-side sampling default.
   Validation NLL fell monotonically in both. Always sample checkpoints and look at
   the output.

3. **Never hardcode the renderer.** Resolve it from the model name. When a renderer
   variant exists for preserving thinking, use it. Check `has_extension_property`;
   without it the cookbook splits conversations per assistant turn and token counts
   explode.

4. **Tool definitions belong in the rows.** A run trained on rows without the
   harness's tool definitions and system prompt regressed on tool use at the
   formula learning rate; the fix was the real system prompt with the working
   directory, the real tool definitions on every row, and a lower LR.

5. **The tokenizer decides the budget.** Rows chunked at 64K with one tokenizer
   rendered 14% longer with another and 10% overflowed. Rebuild per model.

6. **Thinking must be real thinking.** Empty or placeholder think blocks as
   supervision targets degrade reasoning while answers stay plausible. If the
   transcripts carry no plaintext thinking, either train a non-thinking variant or
   generate grounded reasoning traces for every step (a separate pipeline; the
   traces must be first-person, causally grounded in what the step could know, and
   checked for future-only facts such as later commit hashes).

7. **Transcripts disappear.** Claude Code's default retention deleted most of a
   corpus before it was copied. Raise `cleanupPeriodDays` first; copy originals
   verbatim with checksums.

8. **Split by session, by seed.** Chunk-level splits leak conversations; unseeded
   splits make runs incomparable.

9. **Judge checkpoints by an eval you trust.** Our best checkpoint was rarely the
   last one. Score base first so you know what "better" means.

10. **Closed-weight models change the plan.** When a model is Tinker-only, sampling
    and evaluation go through Tinker (native sampler or the OpenAI-compatible
    endpoint) until weights ship; plan the eval harness for that.

11. **Keep the effort fixed.** For effort-conditioned models (Inkling), render
    training rows, sample and evaluate at one effort. Mixed or missing effort means
    out-of-distribution rows and irreproducible scores.

12. **Serve the way you trained.** The stock chat template stripped earlier-turn
    thinking that training had kept, and llama.cpp's default min-p of 0.05 plus a
    proxy's top-p of 1.0 produced garble from a checkpoint that sampled fine on
    Tinker. Match the template kwarg and the base model's sampling defaults, and
    serve at least the training context; a q4_0 KV cache looped past 90K tokens.

13. **Export before it expires, from the right path.** Periodic checkpoints expire
    after 7 days; sampler checkpoints cannot be downloaded at all with current
    tooling. Pick by evaluation, then download the training checkpoint promptly.

14. **Protect the originals.** Never run scrubbing in place; write derived files,
    keep manifests with sha256s, and fail loudly when a scan finds a leak instead of
    writing a partially clean file.
