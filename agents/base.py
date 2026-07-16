"""Shared agent machinery: prompt assembly, audit logging, cost tracking.

Every stage agent goes through run_agent(). Nothing else talks to the SDK directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_NOTES_PATH = REPO_ROOT / "DOMAIN_NOTES.md"

# Model aliases. Route cheap work to haiku, judgment work to sonnet.
MODEL_CHEAP = "haiku"
MODEL_SMART = "sonnet"

# Hard ceiling on agent turns. A stuck agent must fail fast, not spin.
DEFAULT_MAX_TURNS = 25

TOOL_LOG_FILENAME = "run_log.jsonl"
COST_FILENAME = "costs.json"


@dataclass
class AgentSpec:
    """Static definition of one stage agent."""

    name: str
    system_prompt: str
    allowed_tools: list[str]
    model: str = MODEL_SMART
    max_turns: int = DEFAULT_MAX_TURNS


@dataclass
class AgentResult:
    """What the orchestrator gets back. Deliberately small."""

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


def _build_system_prompt(spec: AgentSpec) -> str:
    """Role prompt + institutional memory. Subagent contexts start fresh, so this is
    the only guaranteed channel for hard-won domain facts."""
    return (
        f"{spec.system_prompt}\n\n"
        "--- BEGIN DOMAIN NOTES (established findings — do not re-derive, "
        "do not silently contradict) ---\n"
        f"{_load_domain_notes()}\n"
        "--- END DOMAIN NOTES ---\n"
    )


def _make_audit_hook(log_path: Path, agent_name: str):
    """PostToolUse hook: append every tool call to the run log. Observe only, never block.

    Note: hooks may not fire if the agent hits max_turns, since the session ends first.
    """

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


def _record_cost(run_dir: Path, result: AgentResult) -> None:
    """Append this agent's spend to the run's cost ledger."""
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


async def run_agent(spec: AgentSpec, prompt: str, run_dir: Path) -> AgentResult:
    """Run one agent to completion. Logs every tool call and the run's cost."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / TOOL_LOG_FILENAME

    options = ClaudeAgentOptions(
        system_prompt=_build_system_prompt(spec),
        allowed_tools=spec.allowed_tools,
        model=spec.model,
        max_turns=spec.max_turns,
        cwd=str(REPO_ROOT),
        # matcher=None fires for every tool call.
        hooks={"PostToolUse": [HookMatcher(matcher=None, hooks=[_make_audit_hook(log_path, spec.name)])]},
    )

    final_text, cost_usd, num_turns = "", 0.0, 0

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            # ResultMessage carries the spend for the whole session.
            if hasattr(message, "total_cost_usd"):
                cost_usd = message.total_cost_usd or 0.0
                num_turns = getattr(message, "num_turns", 0)
                final_text = getattr(message, "result", "") or ""

    result = AgentResult(spec.name, final_text, cost_usd, num_turns)
    _record_cost(run_dir, result)
    return result
