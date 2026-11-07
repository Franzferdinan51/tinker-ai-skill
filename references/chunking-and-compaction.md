# Chunking to the context length and the compaction context

A coding session is often far longer than the model's context. We slice it into
rows that fit, and we make every row after the first look the way a real harness
makes a compacted session look. The model then learns both to work from a fresh
start and to continue from a summary.

## Measuring rows

Token counts come from the real renderer and tokenizer of the target model
(`common.make_renderer`), including the system prompt and the tool declaration the
trainer will send. Never measure with a different tokenizer: between two models of
similar size our rows differed by 14%, enough to push 10% of them over the budget.

The builder packs windows from a cheap per-message estimate (`tokenizer.encode` of
the message's text plus a small framing constant), then renders every window. If
any window exceeds `--max-len`, the whole session is re-packed with a smaller raw
window and rendered again, down to a floor. Each row is validated: length <=
max_len, at least one loss token, tool calls declared.

## Safe boundaries

A window boundary is allowed only where no tool call is waiting for its result and
the next message is not a tool message. Splitting a call from its result, or
starting a window with an orphan result, teaches the model nonsense.

## What slice N > 0 sees

```
system: <system prompt>

# Environment

Working directory: /Users/<fake>/<container>/<repo>

Archived working context from earlier turns and compacted portions of the current turn is available below.
Use it as authoritative background ...; when it conflicts with the recent raw turns, prefer the recent raw turns.

<working_context>
session=<id>
generation=<n>
cwd=...
active_goals:        the first user request (clipped)
recent_requests:     the last three later user requests
relevant_files:      up to 15 paths touched, (read)/(modified)
recent_commands:     the last 8 shell commands
in_progress / open_todos / completed:   from task tools, if the harness has them
next_steps:          in-progress items, else the first open todo
preserved_trace:     a near-raw trace of the last 4 messages before the window
</working_context>

user: Continuation context: earlier history ... were compacted into the working
      context provided in the system prompt. Continue from the recent raw messages below.
      (only when the window opens on an assistant message)
assistant / tool / user ...  (the window)
```

The working context is deterministic (no model call) and capped (~4K tokens, scaled
with the context length), so it is reproducible and cheap. It is a summary in the
style of real compaction, not a replacement for the model's own compaction at
serving time.

## Zero-loss windows and exact coverage

A window that contains no assistant tokens (for example a run of tool results at a
boundary) is merged into a neighbour or re-split at another safe boundary until
every row has loss. The rows of a session cover its messages exactly once, which
the builder asserts.

## Train/val split

By whole session, so no conversation leaks across splits. Sessions are visited in
seeded-hash order and added to validation while it stays under `--val-frac` of the
corpus by size; a rebuild for another model holds out the same sessions, which
keeps validation NLL comparable across models and runs.

## Tool definitions on every row

Rows are rendered with the tool declaration the serving harness will send. Pass the
real definitions (`--tools-file`, OpenAI function format, optionally with a
`baseline` list of tools always offered); without them the builder infers minimal
schemas from the arguments it observed. A model we trained without definitions
scored well on NLL and badly on tool use.
