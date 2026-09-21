.PHONY: setup data check lint format test smoke oracle view verify figures figures-check

MODEL ?= mockllm/model
LOG_DIR ?= logs

setup:
	uv sync

data:
	uv run sesgo-data build --version all --language all
	uv run sesgo-data verify

check: lint test

lint:
	uv run ruff check
	uv run ruff format --check
	uv run pyright

format:
	uv run ruff format
	uv run ruff check --fix

test:
	uv run pytest

smoke:
	uv run inspect eval sesgo/sesgo --model $(MODEL) --limit 10 --log-dir $(LOG_DIR)/smoke

oracle:
	uv run inspect eval sesgo/sesgo --model mockllm/model --solver sesgo/oracle_solver \
		-T limit_per_category=25 --log-dir $(LOG_DIR)/oracle

verify:
	uv run python scripts/verify_data.py
	uv run python scripts/verify_oracle.py
	uv run python scripts/verify_rescore.py

view:
	uv run inspect view --log-dir $(LOG_DIR)

# Charts for the README, built from results/history.jsonl alone (no logs, no network).
figures:
	uv run --group figures python scripts/make_figures.py

# Fails when the committed docs/figures/*.svg or results.csv differ from a fresh build.
# It also builds twice into temp dirs and compares hashes, so it fails on a
# non-deterministic chart even before anything is committed. PNG bytes are compared but
# only reported: their encoding is not guaranteed across platforms. The committed files
# were built on macOS arm64; on Linux x86-64 the bootstrap bounds differ in the last
# digits, so this check is not part of CI.
figures-check:
	uv run --group figures python scripts/make_figures.py --check
	git diff --exit-code -- docs/figures
