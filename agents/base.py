# shared agent machinery (prompts, audit log, cost); all sdk calls go through here, nothing else

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

# route cheap work to haiku, judgment work to sonnet
MODEL_CHEAP = "haiku"
MODEL_SMART = "sonnet"

# hard celing on agent turns
DEFAULT_MAX_TURNS = 25

TOOL_LOG_FILENAME = "run_log.jsonl"
COST_FILENAME = "costs.json"
SYSTEM_PROMPT_FILENAME = "system_prompt.txt"


@dataclass
class AgentSpec:
    name: str
    system_prompt: str
    allowed_tools: list[str]
    model: str = MODEL_SMART
    max_turns: int = DEFAULT_MAX_TURNS


@dataclass
class AgentResult:
    name: str
    final_text: str
    cost_usd: float
    num_turns: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_domain_notes() -> str:
    if not DOMAIN_NOTES_PATH.exists():
        raise FileNotFoundError(
            f"DOMAIN_NOTES.md not found at {DOMAIN_NOTES_PATH}. "
            "No agent runs without institutional memory."
        )
    return DOMAIN_NOTES_PATH.read_text(encoding="utf-8")

# institutional memory; channel for domain facts
def _build_system_prompt(spec: AgentSpec) -> str:
    return (
        f"{spec.system_prompt}\n\n"
        "--- BEGIN DOMAIN NOTES (established findings — do not re-derive, "
        "do not silently contradict) ---\n"
        f"{_load_domain_notes()}\n"
        "--- END DOMAIN NOTES ---\n"
    )

# append every tool call to log
def _make_audit_hook(log_path: Path, agent_name: str):
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

# json from agent run
def extract_json_array(final_text: str) -> list | None:
    if not final_text:
        return None
    m = re.search(r"\[.*\]", final_text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None

# for type guard
def extract_json_object(final_text: str) -> dict | None:
    if not final_text:
        return None
    m = re.search(r"\{.*\}", final_text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

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

# runnnnnnnn
async def run_agent(spec: AgentSpec, prompt: str, run_dir: Path) -> AgentResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / TOOL_LOG_FILENAME

    # the system prompt goes to the CLI as a FILE, never as an argv string
    prompt_path = run_dir / SYSTEM_PROMPT_FILENAME
    prompt_path.write_text(_build_system_prompt(spec), encoding="utf-8")

    options = ClaudeAgentOptions(
        system_prompt={"type": "file", "path": str(prompt_path)},
        allowed_tools=spec.allowed_tools,
        model=spec.model,
        max_turns=spec.max_turns,
        cwd=str(REPO_ROOT),
        # matcher=None fires for every tool call
        hooks={"PostToolUse": [HookMatcher(matcher=None, hooks=[_make_audit_hook(log_path, spec.name)])]},
    )

    final_text, cost_usd, num_turns = "", 0.0, 0

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if hasattr(message, "total_cost_usd"):
                cost_usd = message.total_cost_usd or 0.0
                num_turns = getattr(message, "num_turns", 0)
                final_text = getattr(message, "result", "") or ""

    result = AgentResult(spec.name, final_text, cost_usd, num_turns)
    _record_cost(run_dir, result)
    return result
