# shared agent machinery: prompt assembly, audit logging, cost tracking

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

from runmeta import git_sha

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_NOTES_PATH = REPO_ROOT / "DOMAIN_NOTES.md"

# model aliases -- cheap work to haiku, judgment work to sonnet
MODEL_CHEAP = "haiku" # cheap work
MODEL_SMART = "sonnet" # judgment work

DEFAULT_MAX_TURNS = 25 # hard ceiling on turns; a stuck agent fails fast, doesn't spin

# raise the SDK line-buffer cap so image-heavy turns don't overflow its 1 MB default (Section 8)
MAX_BUFFER_SIZE = 32 * 1024 * 1024 # 32 MB

TOOL_LOG_FILENAME = "run_log.jsonl" # per-run tool-call log
COST_FILENAME = "costs.json" # per-run cost ledger
# the exact system prompt the agent saw, and the file the CLI reads it from -- doubles
# as an audit record, the injected DOMAIN_NOTES is reconstructible later
SYSTEM_PROMPT_FILENAME = "system_prompt.txt"


# static definition of one stage agent
@dataclass
class AgentSpec:
    name: str # agent name
    system_prompt: str # role prompt (DOMAIN_NOTES appended later)
    allowed_tools: list[str] # tools the agent may call
    model: str = MODEL_SMART # model alias
    max_turns: int = DEFAULT_MAX_TURNS # turn ceiling
    # top-level DOMAIN_NOTES sections this agent needs, e.g. ("5", "7", "11"); None injects the
    # whole file. subsections ride with their parent. see _select_sections.
    domain_sections: tuple[str, ...] | None = None


# what the orchestrator gets back -- deliberately small
@dataclass
class AgentResult:
    name: str # agent name
    final_text: str # the agent's final message
    cost_usd: float # session spend
    num_turns: int # turns used


# current UTC timestamp, ISO format
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# read DOMAIN_NOTES.md; no agent runs without institutional memory
def _load_domain_notes() -> str:
    if not DOMAIN_NOTES_PATH.exists():
        raise FileNotFoundError(
            f"DOMAIN_NOTES.md not found at {DOMAIN_NOTES_PATH}. "
            "No agent runs without institutional memory."
        )
    return DOMAIN_NOTES_PATH.read_text(encoding="utf-8")


# top-level section headers in DOMAIN_NOTES look like "## 12. Fusion ..."; capture the number
_SECTION_HEADER_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_SECTION_NUM_RE = re.compile(r"^(\d+)\.")


# curate DOMAIN_NOTES to the sections one agent needs; Section 0 (orientation) is always kept.
# a note points at the full file so an unshown cross-referenced section is still recoverable.
def _select_sections(full_text: str, sections: tuple[str, ...]) -> str:
    headers = list(_SECTION_HEADER_RE.finditer(full_text))
    if not headers:
        return full_text
    preamble = full_text[: headers[0].start()].rstrip()
    wanted = {"0", *sections}
    kept: list[str] = []
    for i, h in enumerate(headers):
        num_match = _SECTION_NUM_RE.match(h.group(1))
        num = num_match.group(1) if num_match else None
        end = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)
        if num in wanted:
            kept.append(full_text[h.start():end].rstrip())
    shown = ", ".join(sorted(wanted, key=int))
    note = (
        f"*(Curated subset for this agent - sections {shown}. The full DOMAIN_NOTES.md is at "
        "the repo root; Read it only if you need a section referenced here but not shown.)*"
    )
    return f"{preamble}\n\n{note}\n\n" + "\n\n".join(kept) + "\n"


# role prompt + curated DOMAIN_NOTES, the agent's only channel for domain facts
def _build_system_prompt(spec: AgentSpec) -> str:
    notes = _load_domain_notes()
    if spec.domain_sections is not None:
        notes = _select_sections(notes, spec.domain_sections)
    return (
        f"{spec.system_prompt}\n\n"
        "--- BEGIN DOMAIN NOTES (established findings - do not re-derive, "
        "do not silently contradict) ---\n"
        f"{notes}\n"
        "--- END DOMAIN NOTES ---\n"
    )


# posttooluse hook: append every tool call to the run log; observe only, never block
def _make_audit_hook(log_path: Path, agent_name: str):

    # audit helper
    async def audit(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict:
        record = {
            "ts": _now(),
            "agent": agent_name,
            "event": input_data.get("hook_event_name"),
            "tool": input_data.get("tool_name"),
            "tool_use_id": tool_use_id,
            "tool_input": input_data.get("tool_input"),
            "session_id": input_data.get("session_id"),
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        return {}

    return audit


# append this agent's spend to the run's cost ledger
def _record_cost(run_dir: Path, result: AgentResult) -> None:
    cost_path = run_dir / COST_FILENAME
    ledger = json.loads(cost_path.read_text()) if cost_path.exists() else {"stages": []}
    ledger["stages"].append(
        {
            "ts": _now(),
            "agent": result.name,
            "cost_usd": result.cost_usd,
            "num_turns": result.num_turns,
        }
    )
    ledger["total_usd"] = round(sum(s["cost_usd"] for s in ledger["stages"]), 6)
    cost_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


# every stage agent goes through run_agent(); nothing else talks to the SDK directly
async def run_agent(spec: AgentSpec, prompt: str, run_dir: Path) -> AgentResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / TOOL_LOG_FILENAME

    # system prompt goes to the CLI as a FILE, not an argv string
    prompt_path = run_dir / SYSTEM_PROMPT_FILENAME
    prompt_path.write_text(_build_system_prompt(spec), encoding="utf-8")

    options = ClaudeAgentOptions(
        system_prompt={"type": "file", "path": str(prompt_path)},
        allowed_tools=spec.allowed_tools,
        model=spec.model,
        max_turns=spec.max_turns,
        max_buffer_size=MAX_BUFFER_SIZE,
        cwd=str(REPO_ROOT),
        hooks={"PostToolUse": [HookMatcher(matcher=None, hooks=[_make_audit_hook(log_path, spec.name)])]},
    )

    final_text, cost_usd, num_turns = "", 0.0, 0

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            # ResultMessage carries the spend for the whole session
            if hasattr(message, "total_cost_usd"):
                cost_usd = message.total_cost_usd or 0.0
                num_turns = getattr(message, "num_turns", 0)
                final_text = getattr(message, "result", "") or ""

    result = AgentResult(spec.name, final_text, cost_usd, num_turns)
    _record_cost(run_dir, result)
    return result


# JSON extraction from an agent reply. tolerates prose or a code fence: prefer a fenced ```json block,
# else keep the LAST balanced top-level span that parses to the wanted type. fails safe to None.

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


# each balanced open..close span in s, ignoring delimiters that sit inside a JSON string
def _balanced_spans(s: str, open_ch: str, close_ch: str):
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0:
                yield s[start:i + 1]


# the last balanced JSON value of `want` type in text; a fenced block wins over loose prose
def _extract_json(text: str, open_ch: str, close_ch: str, want: type):
    if not text:
        return None
    fenced = [m.group(1) for m in _FENCE_RE.finditer(text)]
    for src in (*fenced, text):
        found = None
        for span in _balanced_spans(src, open_ch, close_ch):
            try:
                data = json.loads(span)
            except json.JSONDecodeError:
                continue
            if isinstance(data, want):
                found = data
        if found is not None:
            return found
    return None


# the JSON object an agent was asked to emit; tolerant of fences and surrounding prose
def extract_json_object(text: str) -> dict | None:
    return _extract_json(text, "{", "}", dict)


# the JSON array an agent was asked to emit; tolerant of fences and surrounding prose
def extract_json_array(text: str) -> list | None:
    return _extract_json(text, "[", "]", list)


# cross-run analyst ledger.
# append-only ledger of what prior analyst runs kept, read back into the prompt so a later run
# sharpens or contradicts-with-reason rather than restating.


# every record in a cumulative analyst ledger, in order; empty when the file is absent
def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# append validated items to the cumulative ledger, each stamped with its run for traceability;
# no-op on an empty list.
def append_ledger(path: Path, items: list[dict], run_id: str) -> None:
    if not items:
        return
    ts, sha = _now(), git_sha()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps({"ts": ts, "git_sha": sha, "run": run_id, **item},
                                ensure_ascii=False) + "\n")


# gate-and-write: the shared "code judges the agent" sink.
# shared sink for read-only agents: keep the raw reply, run each item through `validate`, write one
# JSONL line per validated item, report how many passed. `is_ok` decides pass (default: validation.ok).

# an item passed the gate iff its validation block says ok
def _validation_ok(validated: dict) -> bool:
    return bool(validated.get("validation", {}).get("ok"))


# validate every item, persist the raw reply + one JSONL line each, return (path, n_passed,
# n_flagged). never raises on a bad item; the raw reply is written next to out_name as <stem>_raw.txt.
def gate_and_write(
    out_dir: Path,
    items: list[dict] | None,
    validate: Callable[[dict], dict],
    final_text: str,
    *,
    out_name: str,
    is_ok: Callable[[dict], bool] = _validation_ok,
    ledger_path: Path | None = None,
    run_id: str = "",
) -> tuple[Path, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / out_name
    (out_dir / f"{path.stem}_raw.txt").write_text(final_text or "", encoding="utf-8")

    if not items:
        path.write_text("", encoding="utf-8")
        return path, 0, 0

    validated = [validate(item) for item in items]
    n_ok = sum(1 for v in validated if is_ok(v))
    with path.open("w", encoding="utf-8") as fh:
        for v in validated:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    # only passers reach the cross-run ledger; flagged items stay only in the raw reply
    if ledger_path is not None:
        append_ledger(ledger_path, [v for v in validated if is_ok(v)], run_id)
    return path, n_ok, len(validated) - n_ok
