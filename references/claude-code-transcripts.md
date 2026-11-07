# Claude Code transcripts: what is in them and what we keep

Claude Code writes one JSONL file per session to `~/.claude/projects/<slug>/<session-uuid>.jsonl`,
where `<slug>` is the working directory with `/` replaced by `-` (for example
`-Users-jdoe-work-my-repo`). Subagent runs are written to `<session-uuid>/subagents/*.jsonl`.
Transcripts are deleted after 30 days unless `cleanupPeriodDays` is raised in `~/.claude/settings.json`.

## Record types

Each line is a JSON object with a `type`. Only two types are conversation:

| type | meaning | kept |
|---|---|---|
| `user` | a human turn, a tool result, or harness-injected text | yes, after filtering |
| `assistant` | one content block of a model response (text, thinking, tool_use) | yes |
| `system` | compaction metadata, API refusals, local events | no |
| `attachment`, `file-history-*`, `queue-operation`, `api-request*`, `mode`, `last-prompt`, `ai-title`, `cost-state`, ... | runtime bookkeeping | no |

Useful fields on `user`/`assistant` records: `uuid`, `parentUuid`, `sessionId`, `cwd`,
`timestamp`, `isSidechain` (subagent record interleaved in the main file), `isMeta`
(injected, not typed), `isCompactSummary` (Claude Code's own compaction summary, a
user-role message), `isApiErrorMessage`, `message.model`, `message.content`.

## Content blocks

`message.content` is a string or a list of blocks:

| block | handling |
|---|---|
| `text` (user) | kept unless it is harness text: starts with `<system-reminder>`, `<command-...`, `<local-command`, `<task-notification>`, `[SYSTEM NOTIFICATION`, `[Request interrupted` |
| `text` (assistant) | kept |
| `thinking` | plaintext kept when present; the `signature` is dropped. Some configurations store an empty `thinking` string with only a signature; then the row has no thinking |
| `redacted_thinking` | dropped |
| `tool_use` | becomes an assistant `tool_calls` entry: `id`, `name`, `arguments = json.dumps(input)` |
| `tool_result` | becomes a `tool` message with `tool_call_id`; `content` may be a string or a list of text/image blocks; `is_error` prefixes `[tool error]`; long results keep the first 8,000 and last 4,000 characters (errors live at the end) |
| `image`, `input_image`, `image_url` | replaced by `[image omitted]`; a user message that is only an image is dropped |

The top-level `toolUseResult` field duplicates tool output and is ignored.

## Message schema produced by `normalize_sessions.py`

One file per session: a header record, then one message per line.

```
{"type":"session","session_id":"...","cwd":"/Users/jdoe/work/my-repo","started_at":"...","ended_at":"...",
 "models":{"claude-...":123},"messages":480,"assistant_messages":230,"tool_calls":{"Read":80,"Bash":60},
 "normalization":{...counters...}}
{"role":"user","content":"Add a --units flag ..."}
{"role":"assistant","content":[{"type":"thinking","thinking":"..."},{"type":"text","text":"..."}],
 "tool_calls":[{"id":"toolu_...","type":"function","function":{"name":"Read","arguments":"{\"file_path\": \"...\"}"}}]}
{"role":"tool","content":"...","tool_call_id":"toolu_..."}
```

Invariants checked: every tool call is answered by a tool message before the next
assistant turn; a conversation starts with a user message; sessions with fewer than
`--min-assistant` assistant messages are skipped. One assistant message per "step":
the run of thinking, text and tool calls between two observations.

This is the format every later step consumes. To use transcripts from another
agent, emit this schema and continue from `anonymize_sessions.py`.
