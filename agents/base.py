# shared agent machinery: prompt assembly, audit logging, cost tracking

from __future__ import annotations

import json
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


# role prompt + institutional memory -- subagent contexts start fresh, so this is the
# only guaranteed channel for hard-won domain facts
def _build_system_prompt(spec: AgentSpec) -> str:
    return (
        f"{spec.system_prompt}\n\n"
        "--- BEGIN DOMAIN NOTES (established findings - do not re-derive, "
        "do not silently contradict) ---\n"
        f"{_load_domain_notes()}\n"
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
