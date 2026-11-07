#!/usr/bin/env python3
"""Write a small synthetic Claude Code transcript so the whole pipeline can be
exercised offline before touching real sessions.

The output mimics ~/.claude/projects/<slug>/<session>.jsonl: user and assistant
records with content blocks (text, thinking, tool_use, tool_result, image), cwd,
timestamps, a sidechain (subagent) record, harness-injected user text, a fake
API key that the anonymizer must catch, and a second session that is too small
to be selected.

Usage:
    python scripts/make_sample_transcript.py --out ./demo-claude-projects --turns 12
    python scripts/collect_sessions.py --claude-dir ./demo-claude-projects --data-dir ./demo-data
"""
import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

FAKE_HOME = "/Users/taylor.quintrell"
PROJECT = f"{FAKE_HOME}/side-projects/weather-cli"
SLUG = "-Users-taylor.quintrell-side-projects-weather-cli"
FAKE_KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v2"
SNIPPET = "def fetch(city: str) -> dict:\n    url = f'https://api.example.com/v1/weather?q={city}'\n    return requests.get(url, timeout=10).json()\n"


def rec(kind, session, cwd, ts, content, **extra):
    r = {"type": kind, "uuid": str(uuid.uuid4()), "parentUuid": extra.pop("parent", None),
         "sessionId": session, "cwd": cwd, "timestamp": ts.isoformat().replace("+00:00", "Z"),
         "isSidechain": extra.pop("sidechain", False), "version": "2.1.0", "gitBranch": "main",
         "message": {"role": "assistant" if kind == "assistant" else "user", "content": content}}
    if kind == "assistant":
        r["message"]["model"] = "claude-example-model"
        r["message"]["id"] = "msg_" + uuid.uuid4().hex[:12]
    r.update(extra)
    return r


def session_records(session, turns, rng, long_results):
    ts = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    out = []

    def step(seconds=7):
        nonlocal ts
        ts += timedelta(seconds=seconds)
        return ts

    out.append(rec("user", session, PROJECT, step(), [{"type": "text", "text":
                   "<system-reminder>\nThis is injected harness text and must not become training data.\n</system-reminder>"}],
                   isMeta=True))
    out.append(rec("user", session, PROJECT, step(),
                   f"Add a --units flag to the weather CLI so I can switch between celsius and fahrenheit. "
                   f"The config lives in {PROJECT}/config.toml and my API key is {FAKE_KEY}; don't commit it."))
    files = ["src/weather/cli.py", "src/weather/fetch.py", "src/weather/format.py", "tests/test_cli.py", "README.md"]
    for i in range(turns):
        f = f"{PROJECT}/{rng.choice(files)}"
        kind = rng.choice(["Read", "Read", "Edit", "Bash", "Grep"])
        tool_id = "toolu_" + uuid.uuid4().hex[:20]
        if kind == "Read":
            tool_input = {"file_path": f}
        elif kind == "Edit":
            tool_input = {"file_path": f, "old_string": "units = 'metric'", "new_string": "units = args.units"}
        elif kind == "Grep":
            tool_input = {"pattern": "units", "path": PROJECT}
        else:
            tool_input = {"command": f"cd {PROJECT} && python -m pytest -q tests/test_cli.py"}
        thinking = (f"The user wants a --units flag. Step {i + 1}: I should look at {f.split('/')[-1]} "
                    f"before changing anything, then run the tests.")
        out.append(rec("assistant", session, PROJECT, step(),
                       [{"type": "thinking", "thinking": thinking, "signature": "sig_" + uuid.uuid4().hex}]))
        out.append(rec("assistant", session, PROJECT, step(1),
                       [{"type": "tool_use", "id": tool_id, "name": kind, "input": tool_input}]))
        if kind == "Read":
            body = SNIPPET * (60 if long_results else 3) + f"# {f}\n"
        elif kind == "Bash":
            body = ("..." * 20 + "\n") * (200 if long_results else 4) + "3 passed in 0.41s\n"
        elif kind == "Edit":
            body = f"The file {f} has been updated successfully."
        else:
            body = "\n".join(f"{PROJECT}/src/weather/cli.py:{n}:    units = 'metric'" for n in range(10, 10 + (40 if long_results else 3)))
        out.append(rec("user", session, PROJECT, step(), [{"type": "tool_result", "tool_use_id": tool_id,
                                                           "content": [{"type": "text", "text": body}]}],
                       toolUseResult={"stdout": body[:200]}))
        if i == 2:
            # A subagent (sidechain) record interleaved in the main file: must be skipped.
            out.append(rec("assistant", session, PROJECT, step(), [{"type": "text", "text": "subagent chatter"}], sidechain=True))
        if i == 4:
            out.append(rec("user", session, PROJECT, step(),
                           [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
                            {"type": "text", "text": "Here is a screenshot of the current output. -- Taylor Quintrell (taylor.q@example-mail.test)"}]))
            out.append(rec("assistant", session, PROJECT, step(),
                           [{"type": "text", "text": "Thanks, the screenshot shows the metric default. Continuing."}]))
    out.append(rec("assistant", session, PROJECT, step(),
                   [{"type": "text", "text": f"Done. I added `--units {{metric,imperial}}` to {PROJECT}/src/weather/cli.py, "
                                             f"threaded it through fetch and format, and the tests pass. Your key stays in config.toml, "
                                             f"which is already gitignored."}]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("./demo-claude-projects"))
    ap.add_argument("--turns", type=int, default=12, help="tool-use steps in the longest session")
    ap.add_argument("--sessions", type=int, default=3, help="number of selectable sessions to write")
    ap.add_argument("--long-results", action="store_true", help="make tool results long enough to force chunking")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    proj = args.out / SLUG
    proj.mkdir(parents=True, exist_ok=True)
    main_id = None
    for k in range(args.sessions):
        sid = str(uuid.UUID(int=rng.getrandbits(128)))
        main_id = main_id or sid
        turns = max(3, args.turns - 3 * k)          # sessions of different lengths
        (proj / f"{sid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in session_records(sid, turns, rng, args.long_results)))
    tiny_id = str(uuid.UUID(int=rng.getrandbits(128)))
    (proj / f"{tiny_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in session_records(tiny_id, 1, rng, False)))
    sub = proj / main_id / "subagents"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "agent-abc123.jsonl").write_text(json.dumps(rec("user", main_id, PROJECT, datetime(2026, 9, 1, tzinfo=timezone.utc),
                                                            "subagent prompt", sidechain=True)) + "\n")
    print(f"wrote {args.sessions} sessions under {proj} (longest: {args.turns} steps), a too-small session, and a subagent file")


if __name__ == "__main__":
    main()
