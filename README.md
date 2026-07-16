# h-care-agents

Staged, agent-assisted pipeline for IMU locomotion data.
Read `PLAN.md` for architecture and phase gates. Read `DOMAIN_NOTES.md` before touching anything.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your key
export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2)
```

## Run

```bash
python orchestrator.py --phase 0
```

Artifacts land in `runs/YYYY-MM-DD_runN/`: `run_log.jsonl` (every tool call), `costs.json` (spend).
