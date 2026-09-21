# 06 Runner and monitoring

Modules: `src/sesgo/_panel.py` (load and validate `models.yaml`), `src/sesgo/_cost.py` (prices, token counts, ledger), `src/sesgo/_runner.py` (`sesgo-run`), `src/sesgo/_report.py` (`sesgo-report`), `src/sesgo/_paper.py` (paper reference values), `models.yaml`.

Depends on the public API of specs 01–05 only: `sesgo(...)` task, `load_records`, `format_*_prompt`, `summarize`, `bootstrap_se`, `Row`.

## models.yaml (D13)

```yaml
defaults:
  temperature: 0.75
  max_connections: 16
  max_tokens: 512
configs:
  - name: llama-3.1-8b-instruct          # unique, used in paths
    model: openrouter/meta-llama/llama-3.1-8b-instruct
    group: anchor                        # anchor | modern
    reasoning: "off"                     # off | low
    prompt_styles: [clean, paper]        # default [clean]
    model_args: {}                       # passed as Inspect model_args
    generate: {}                         # GenerateConfig overrides
  - name: gpt-4o-mini
    model: openrouter/openai/gpt-4o-mini
    group: anchor
    reasoning: "off"
    prompt_styles: [clean, paper]
  - name: gpt-5.4-nano
    model: openrouter/openai/gpt-5.4-nano
    group: modern
    reasoning: "off"
    model_args: {reasoning_enabled: false}
  - name: gpt-5.4-nano-low
    model: openrouter/openai/gpt-5.4-nano
    group: modern
    reasoning: "low"
    generate: {reasoning_effort: low, max_tokens: 4096}
  # same two entries for: google/gemini-3.1-flash-lite, deepseek/deepseek-v4-flash, qwen/qwen3.5-9b
```

To add a model a user adds one entry. Validation errors name the entry and the field. How to switch reasoning off differs by provider: find the working setting per model with real 3-sample calls (spec 07 V4) and check in the log that reasoning tokens are 0 or near 0. If a model cannot switch reasoning off, keep one entry, mark `reasoning: "low"`, and say so in the entry's `note`, which the report prints.

## sesgo-run

```
sesgo-run estimate [--only NAMES] [--reasoning off|low|all] [--stage pilot|full]
sesgo-run pilot    [--only NAMES] [--reasoning off|low|all] [--budget 15]
sesgo-run full     [--only NAMES] [--reasoning off|low|all] [--budget 15]
sesgo-run ledger
```

- One unit of work = (config, prompt_style, stage). Log dir: `logs/{stage}/{name}__{style}/`. The runner calls `inspect_ai.eval_set` for the unit (retry and resume are native; a complete unit is skipped). `pilot` sets `limit_per_category=50`. `full` sets no limit.
- `log_format="eval"`, `fail_on_error=0.02`, `retry_on_error=3`, a `max_connections` from the panel, `display="plain"` or `"log"` so it runs in tmux without a TTY problem.
- Order inside a command: cheapest estimated unit first (D14).
- **Estimate**: input tokens are counted, not guessed: format every prompt of the stage with the real templates and count with `tiktoken` (`o200k_base`) as an approximation for every model (state this in the table). Output tokens per sample: measured mean from the pilot logs of the same config when they exist, else 20 (reasoning off) or 600 (reasoning low). Prices: `GET https://openrouter.ai/api/v1/models` (no key), cached to `results/prices.json` with a timestamp. The table prints config, style, samples, input tokens, output tokens, USD, and the cumulative USD.
- **Budget gate**: before each unit, `spent (ledger) + estimate(unit) > budget` → stop with exit code 3 and a clear message. Never start a unit that would pass the cap. Default budget 15.
- **Ledger** `results/ledger.jsonl` (committed; no prompt text): one line per finished unit: timestamp, unit, samples, input/output/reasoning tokens from `EvalLog.stats.model_usage`, `cost_tokens_usd` (tokens × price), and `cost_credits_usd` = delta of `total_usage` from `GET https://openrouter.ai/api/v1/credits` (auth header; key from env, loaded with `python-dotenv`; never printed) measured around the unit. Spent = `cost_credits_usd` when available, else `cost_tokens_usd`. Runs are sequential so the delta is attributable. Manual real calls made during verification are added with `sesgo-run ledger --add "note" USD`.

## sesgo-report

```
sesgo-report [--logs logs/full] [--history results/history.jsonl] [--out REPORT.md]
```

- Reads every successful `.eval` log under `--logs` with `inspect_ai.log.read_eval_log`. Builds `Row`s from sample scores and metadata. Calls `summarize` and `bootstrap_se` for headline keys (`accuracy`, `ft_minus_fo`, `bias_score` for each split and group).
- `results/history.jsonl` (committed, aggregates only): one record per log, keyed by `eval.run_id` + task id, idempotent (re-running does not duplicate): timestamps, config name, model, group, reasoning, prompt_style, task version, dataset version and commit, parser version, n, all metrics, standard errors, token usage, cost, log file name. This file is the input of the future web site.
- `REPORT.md` is written in **Spanish**. Content:
  1. Summary table per split: model × (pooled + 4 categories) with `accuracy` and `bias_score ± se`.
  2. Anchors against the paper: our `paper` style, our `clean` style, and the paper value (from `_paper.py`: Table A3 pooled at T = 0.75, and Tables 2–3 per category with the caption swap corrected: Table 2 values are ambiguous, Table 3 values are disambiguated). Columns: protocol effect (clean − paper style) and port fidelity (our paper style − paper).
  3. Reasoning off against low for the modern models.
  4. Rates: unparsed, invalid, refusal, no response, truncation, parse stage; bias_score bounds; a flag when `unparsed_rate > 0.02` or when the bound gap exceeds the difference to the next model.
  5. Real cost per unit and total, from the ledger.
  6. Design differences from the paper and their measured effect.
  7. Commands: `uv run inspect view --log-dir logs/full`, and how to add a model.
- The report must not contain prompt text or model responses.

## Edge cases

- A unit with a failed log is resumed by `eval_set`; after 3 failed attempts the runner records the failure in the ledger and continues with the next unit.
- A model that rejects `temperature` (some reasoning models): retry the unit without temperature only if the API error says so; record it.
- Missing price for a model: the estimate fails closed (unit not run).
- `history.jsonl` lines from older task versions stay; the report uses the newest record per (config, style, task version).

## Acceptance

1. `sesgo-run estimate --stage full --reasoning all` prints the table with counted input tokens and a total.
2. `sesgo-run pilot --only llama-3.1-8b-instruct` writes a log, a ledger line with both cost fields, and is skipped when run again.
3. The budget gate is proven with `--budget 0.0001` (exit code 3, nothing called).
4. `sesgo-report` on the pilot logs writes `history.jsonl` and a Spanish `REPORT.md`; a second run adds no duplicate lines.
5. Unit tests: panel validation, estimate arithmetic with a fake price table, ledger sum, history idempotency, report rendering from synthetic rows. No network in unit tests.
