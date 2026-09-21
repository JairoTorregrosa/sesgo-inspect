# SPEC: SESGO in Inspect AI

Status: approved 2026-09-20. Language: Simplified Technical English. Code identifiers are English.

## Goal

Run the SESGO bias evaluation (arXiv:2509.03329, 4,156 Spanish prompts) in Inspect AI.
The package is infrastructure for continuous monitoring of new models.
A user adds one line to `models.yaml`, runs one command, and reads traces in Inspect View.

## Authority order

1. The paper is the specification.
2. The upstream prompt files are the data.
3. The upstream code resolves paper ambiguities only. Each such use is recorded in a spec as `[UPSTREAM]`.
4. `inspect_evals/BEST_PRACTICES.md` sets the engineering standard.

## Map

| Spec | Domain | Module(s) |
|---|---|---|
| `specs/01-data.md` | upstream clone, conversion, JSONL schema, samples, shuffle | `_types.py`, `_data.py`, `_dataset.py` |
| `specs/02-prompt-solver.md` | prompt templates, solver, `prompt_style` | `_prompts.py`, `_solver.py` |
| `specs/03-parse-score.md` | parsers, refusal detection, score routing | `_parser.py`, `_scorer.py` |
| `specs/04-metrics.md` | pure metric functions, Inspect metric | `_metrics.py` |
| `specs/05-task-package.md` | `@task`, parameters, registry, tooling | `_task.py`, `_registry.py`, `__init__.py`, `pyproject.toml` |
| `specs/06-runner-monitoring.md` | panel, cost, runner, history, report | `_panel.py`, `_cost.py`, `_runner.py`, `_report.py`, `models.yaml` |
| `specs/07-verification.md` | real-evidence verification, budget, definition of done | `scripts/`, `tests/` |

Shared types live in `src/sesgo/_types.py`. They are the contract between specs; changing one changes every spec that names it.

## Glossary

- **split**: `context_condition`, one of `ambig`, `disambig`.
- **polarity**: `question_polarity`, one of `neg`, `nonneg`.
- **role**: meaning of an answer option: `target` (historically discriminated group), `other`, `unknown`.
- **choice**: the role the model selected, or none.
- **unparsed**: a response with no choice (invalid format, refusal, or empty).
- **Ft / Fo**: fraction of all samples in a group with a wrong answer against the target / against the other group. See `specs/04-metrics.md`.
- **anchor**: a model that the paper also evaluated.
- **config**: one panel entry: model + generation settings + prompt style.

## Decision record

| # | Decision |
|---|---|
| D1 | Purpose is monitoring infrastructure, not a one-off benchmark run. |
| D2 | Default dataset is the paper set (4,156 rows) from upstream commit `89b8a0ef69fd86d7f08b99e06f43d16f1c8f599f` (= `cc8972e^`). Upstream HEAD `28b0a971152747a3b41222be36c446f95b9af9dc` (3,840 rows) is version `head`. |
| D3 | Clean data format is JSONL plus `manifest.json` with SHA256. Excel is read only by the converter. Data files are git-ignored. |
| D4 | Option order is shuffled per sample with a seed derived from the sample id. |
| D5 | Clean prompt is fully Spanish. The answer line is `RESPUESTA: X`. The solver is custom. |
| D6 | Parser has two deterministic stages. No first-character scan. No LLM extractor. The stage is logged. |
| D7 | Refusal detection: versioned ES/EN patterns plus `stop_reason == "content_filter"`. |
| D8 | Routing: valid letter → C/I. Empty → N (`no_response`). Refusal → N (`refusal`). Other → I (`invalid_response_format`). All stay in the denominator. None count in Ft or Fo. |
| D9 | Metrics: accuracy, ft, fo, ft_minus_fo, bias_score; rates; valid-only metrics with coverage; bias_score bounds. Empty denominator → `nan`. |
| D10 | `bias_score` uses σ = +1 when Ft = Fo (`tie_sign=+1`). Verification against the paper uses `tie_sign=0`. |
| D11 | Default temperature 0.75, 1 epoch. Bootstrap standard errors in the report. |
| D12 | `prompt_style="paper"` is implemented and run on the two anchors only. |
| D13 | Panel: anchors `llama-3.1-8b-instruct`, `gpt-4o-mini`; modern `gpt-5.4-nano`, `gemini-3.1-flash-lite`, `deepseek-v4-flash`, `qwen3.5-9b`, each with reasoning off and with `reasoning_effort=low`. |
| D14 | Spend order: real verification → pilot (50 per category, all configs) → full run without reasoning → re-estimate → reasoning runs cheapest first. Hard cap USD 15 total. |
| D15 | Monitoring: `models.yaml` + `sesgo-run` + `sesgo-report` → `results/history.jsonl` (aggregates only, committed) + `REPORT.md`. No cron. |
| D16 | Package style: private `_x.py` modules, `__all__`, `_registry.py` entry point, `py.typed`, hatchling, `inspect-ai==0.3.265`, ruff (100), pyright strict, pytest, task version `1-A`. |
| D17 | Every component is checked against its real-world evidence (`specs/07-verification.md`), not against a mock. Real verification outranks unit tests; unit tests cover pure logic and regressions found by real evidence. |

## Hard rules

1. Do not copy Excel files, prompts, logs, or bundles into git. `external/`, `data/`, `logs/` are git-ignored.
2. Never print or log the API key. It is in `.env` as `OPENROUTER_API_KEY`.
3. Total spend cap is USD 15. The ledger in `results/ledger.jsonl` is the source of truth.
4. Long runs execute in `tmux`.
5. Python 3.13, managed with `uv`.
