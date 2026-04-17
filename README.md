# Yigent

General-purpose agent harness with streaming execution, Plan mode, self-improving skills, and generalization benchmark.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp configs/default.yaml configs/local.yaml  # edit with your API keys
```

## Run

```bash
python -m src.ui.cli
```
