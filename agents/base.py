# shared agent machinery: prompt assembly, audit logging, cost tracking

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_NOTES_PATH = REPO_ROOT / "DOMAIN_NOTES.md"

# model aliases -- cheap work to haiku, judgment work to sonnet
MODEL_CHEAP = "haiku" # cheap work
MODEL_SMART = "sonnet" # judgment work

DEFAULT_MAX_TURNS = 25 # hard ceiling on turns; a stuck agent fails fast, doesn't spin

# stream-json carries tool results inline, base64 images included. one S3 turn can batch
# several ~0.5 MB figures (parallel Read) into a single message, past the SDK transport's
# 1 MB default line buffer -- a fatal decode mid-run (the third stdio trap after CreateProcess
# length and cp949, DOMAIN_NOTES Section 8). raise the cap so image-reading agents survive a big turn.
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
    # top-level DOMAIN_NOTES sections this agent needs, e.g. ("5", "7", "11"); None injects
    # the whole file. Section 0 (orientation) is always kept. subsections ride with their
    # parent (giving "12" gives 12.1-12.5). see _select_sections.
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


# curate DOMAIN_NOTES to the sections one agent needs. injecting the whole file into every
# agent does not scale (it once blew the CreateProcess cap, and the cheap agent dropped a
# field and doubled in cost purely on prompt size -- BUILDLOG Appendix A). an agent needs
# only its own stage's sections; Section 0 (orientation) is always kept, and a note points
# at the full file on disk for any cross-referenced section not shown, so under-scoping is
# recoverable rather than fatal (every agent has Read).
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


# role prompt + institutional memory -- subagent contexts start fresh, so this is the
# only guaranteed channel for hard-won domain facts
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
# note: may not fire if the agent hits max_turns -- the session ends first
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


# --- JSON extraction from an agent reply -------------------------------------
# agents are told to emit one JSON value, but models still wrap it in prose or a code
# fence, and sometimes emit more than one brace-delimited span (a worked example, then the
# answer). the old first-brace-to-last-brace regex breaks on both: any second span, or any
# prose brace, and json.loads sees the whole run and fails. so: prefer a fenced ```json
# block, else scan for balanced top-level spans (ignoring delimiters inside strings) and
# keep the LAST one that parses to the wanted type -- the answer tends to come last. every
# path fails safe to None, which every caller already treats as "did not parse".

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
