# Yigent Harness — Common Commands

.PHONY: setup run test eval clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"
	cp -n configs/default.yaml configs/local.yaml || true
	@echo "Edit configs/local.yaml with your API keys"

run:
	. .venv/bin/activate && python -m src.ui.cli

api:
	. .venv/bin/activate && python -m src.ui.api

test:
	. .venv/bin/activate && pytest tests/ -q

eval:
	. .venv/bin/activate && python -m src.eval.benchmark --suite all

eval-coding:
	. .venv/bin/activate && python -m src.eval.benchmark --suite coding

lint:
	. .venv/bin/activate && ruff check src/ tests/

format:
	. .venv/bin/activate && ruff format src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info dist build
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

smoke-test:
	. .venv/bin/activate && python -m src.ui.cli --smoke-test

export-trajectories:
	. .venv/bin/activate && python -m src.learning.trajectory --export sharegpt --output trajectories/

docker-up:
	docker compose up -d

docker-down:
	docker compose down
