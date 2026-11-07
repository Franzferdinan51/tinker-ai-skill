"""Shared helpers for the tinker-ai-skill scripts.

Standard library plus tinker-cookbook. Every script imports from here so the
dataset builder, the trainer and the sampler render conversations identically.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- environment
def load_dotenv(*dirs):
    """Load KEY=VALUE lines from the first .env found in `dirs` (default: cwd,
    then the skill root). Never overrides variables already in the shell."""
    candidates = [Path(d) / ".env" for d in dirs] or []
    candidates += [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    for env_file in candidates:
        if not env_file.is_file():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return env_file
    return None


# ------------------------------------------------------------------- file io
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    """Atomic: write to a temp file beside the target, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    n = 0
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(dump(row) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return n


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def read_session(path):
    """A normalized/anonymized session file: header record, then one message per line."""
    rows = read_jsonl(path)
    header = next(rows)
    if header.get("type") != "session":
        raise ValueError(f"{path}: first record is not a session header")
    return header, list(rows)


# ------------------------------------------------------------- text helpers
# User-role text that the harness injected, not something a person typed.
HARNESS_PREFIXES = ("<command-", "<local-command", "<system-reminder>",
                    "<task-notification>", "[SYSTEM NOTIFICATION", "[Request interrupted")


def is_harness_text(text):
    return text.lstrip().startswith(HARNESS_PREFIXES)


def clip_middle(text, head, tail):
    """Keep the head and the tail of a long tool result; errors live at the end."""
    if len(text) <= head + tail + 200:          # minimum-gain guard
        return text
    return f"{text[:head]}\n\n...[{len(text) - head - tail} chars truncated]...\n\n{text[-tail:]}"


def model_slug(model_name):
    return re.sub(r"[^A-Za-z0-9.]+", "-", model_name.split("/")[-1]).strip("-").lower()


# --------------------------------------------------------- rendering (shared)
def make_renderer(model, renderer_name=None, preserve_thinking=True):
    """Tokenizer + renderer resolved from the model name, the way Tinker's cookbook
    recommends. Returns (tokenizer, renderer, provenance dict).

    preserve_thinking: hybrid-thinking renderers (Qwen3.5/3.6, Nemotron) default to
    stripping thinking from earlier assistant turns, which turns those turns into
    empty-think supervision targets and removes the renderer's "extension
    property" (one example per conversation). We rebuild such renderers with
    strip_thinking_from_history=False so every assistant turn keeps its thinking.
    """
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model)
    name = renderer_name or model_info.get_recommended_renderer_name(model)
    renderer = get_renderer(name, tokenizer)
    note = None
    if preserve_thinking and getattr(renderer, "strip_thinking_from_history", False):
        params = inspect.signature(type(renderer).__init__).parameters
        if "strip_thinking_from_history" in params:
            renderer = type(renderer)(tokenizer, strip_thinking_from_history=False)
            note = "rebuilt with strip_thinking_from_history=False"
    accepts_effort = "effort" in inspect.signature(renderer.build_supervised_example).parameters
    tml = getattr(tokenizer, "tml_tokenizer", None)
    provenance = {
        "model": model, "renderer": name, "renderer_class": type(renderer).__name__,
        "tokenizer": type(tml).__name__ if tml is not None else type(tokenizer).__name__,
        "preserve_thinking": bool(preserve_thinking), "accepts_effort": accepts_effort,
        "has_extension_property": bool(renderer.has_extension_property), "note": note,
        "tinker_cookbook": importlib.metadata.version("tinker-cookbook"),
    }
    return tokenizer, renderer, provenance


def to_render_message(m):
    """Dataset message -> cookbook Message (tool_calls dicts become ToolCall objects)."""
    from tinker_cookbook.renderers.base import ToolCall
    out = {"role": m["role"]}
    if m.get("content") is not None:
        out["content"] = m["content"]
    if m["role"] == "tool" and m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    if m.get("tool_calls"):
        out["tool_calls"] = [ToolCall(id=tc.get("id"),
                                      function=ToolCall.FunctionBody(name=tc["function"]["name"],
                                                                     arguments=tc["function"]["arguments"]))
                             for tc in m["tool_calls"]]
    return out


def tool_specs(tools):
    """OpenAI-style tool dicts -> cookbook ToolSpec list."""
    from tinker_cookbook.renderers.base import ToolSpec
    specs = []
    for t in tools or []:
        f = t["function"] if t.get("type") == "function" else t
        specs.append(ToolSpec(name=f["name"], description=f.get("description", ""),
                              parameters=f.get("parameters") or {"type": "object", "properties": {}}))
    return specs


def conversation_with_tools(renderer, messages, tools):
    """System prompt (if the row has one), then the model-specific tool declaration,
    then the rest of the row. The prefix helper emits nothing for an empty prompt or
    an empty tool list."""
    has_system = bool(messages) and messages[0]["role"] == "system"
    system_prompt = messages[0]["content"] if has_system else ""
    prefix = renderer.create_conversation_prefix_with_tools(tool_specs(tools), system_prompt=system_prompt)
    rest = messages[1:] if has_system else messages
    return prefix + [to_render_message(m) for m in rest]


def supervised_example(renderer, messages, tools, effort=None, provenance=None):
    """(ModelInput, per-token weights) for one row, loss on all assistant messages."""
    from tinker_cookbook.renderers import TrainOnWhat
    conv = conversation_with_tools(renderer, messages, tools)
    kwargs = {}
    accepts = provenance["accepts_effort"] if provenance else \
        "effort" in inspect.signature(renderer.build_supervised_example).parameters
    if accepts:
        kwargs["effort"] = 0.9 if effort is None else float(effort)
    return renderer.build_supervised_example(conv, TrainOnWhat.ALL_ASSISTANT_MESSAGES, **kwargs)


def token_count_and_loss(renderer, messages, tools, effort=None, provenance=None):
    """Sequence length and number of loss-bearing targets, as the trainer sees them
    (the cookbook keeps weights[1:] after next-token shifting)."""
    model_input, weights = supervised_example(renderer, messages, tools, effort, provenance)
    w = weights.tolist()
    return model_input.length, sum(1 for x in w[1:] if x > 0)
