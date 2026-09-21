# Monitoring: `sesgo-run` and `sesgo-report`

`models.yaml` is the panel: one entry per configuration (model + generation settings).
A configuration crossed with a prompt style and a stage is a **unit of work**, which is
what the runner evaluates, estimates and bills.

```bash
uv run sesgo-run estimate --stage full --reasoning all   # cost table, calls nothing
uv run sesgo-run smoke                                   # 3 samples per category, every config
uv run sesgo-run pilot  --only gpt-4o-mini               # 50 per category
uv run sesgo-run full   --reasoning off                  # the whole dataset
uv run sesgo-run ledger                                  # what was really spent
uv run sesgo-report --logs logs/full                     # results/history.jsonl + REPORT.md
```

| Stage | `limit_per_category` | Log directory |
|---|---|---|
| `smoke` | 3 | `logs/smoke/panel/{config}__{style}/` |
| `pilot` | 50 | `logs/pilot/{config}__{style}/` |
| `full` | none (4,156) | `logs/full/{config}__{style}/` |

Only one `sesgo-run` at a time, and no other traffic on the same OpenRouter key while it
runs: the ledger reads the account's credits around each unit, so a second caller makes
`cost_credits_usd` wrong. A plain `inspect eval` is safe to book afterwards with
`sesgo-run ledger --add`.

## Adding a model

`models.yaml` lives at the repo root and ships with the panel of the report. One entry is
enough:

```yaml
  - name: my-model                       # unique, becomes a directory name
    model: openrouter/vendor/my-model
    group: modern                        # anchor | modern
    reasoning: "off"                     # off | low
    model_args: {reasoning_enabled: false}
```

Then `uv run sesgo-run smoke --only my-model`. The smoke stage ends with the **V4
check**: it reads `reasoning_tokens` out of the real logs and fails (exit code 4) when a
configuration declared `off` still reasons, or when one declared `low` does not.
How reasoning is switched off is provider-specific, so it is measured, never assumed;
the measured settings and their evidence are in the comment at the top of `models.yaml`.
A model that cannot switch reasoning off keeps a single entry marked `reasoning: "low"`
and a `note:`, which the report prints.

**Check the throughput before you launch the full stage.** `max_connections` defaults to
16, which the panel's providers absorb (4,156 samples in 3–7 minutes each). A model
served by a single rate-limited provider answers far more slowly, and the cost estimate
does not see it: a rejected request is free, so only the clock suffers. The pilot is the
measurement — divide 200 samples by its wall time. If a sample costs seconds instead of
milliseconds, look for HTTP 429 in the log and lower `max_connections` on that entry:

```bash
uv run python -c "
from inspect_ai.log import read_eval_log
log = read_eval_log('logs/pilot/my-model__clean/<file>.eval')
calls = [e for s in log.samples for e in s.events if e.event == 'model']
print(len(calls), 'model calls for', len(log.samples), 'samples;',
      sum(1 for e in calls if e.error), 'of them failed')
"
```

Measured on `mistral-small-2603` (the only provider is Mistral): 200 pilot samples took
**30 min 34 s** at `max_connections: 16` because **274 of its 474 calls came back 429**
and were retried with backoff — against **38 s** for the same 200 samples on
`gpt-4.1-nano`, a 48× difference for the same money. At that rate the full stage is
about six hours, so measure the pilot before you launch it.

## Budget

Nothing runs without an estimate. Input tokens are **counted**: every prompt of the
stage is formatted with the real templates and tokenized with `tiktoken o200k_base`
(exact for OpenAI models, an approximation for the others). Output tokens come from the
pilot logs of the same configuration when they exist, else 20 (reasoning off) or 600
(reasoning low). Prices come from `https://openrouter.ai/api/v1/models`, cached in
`results/prices.json`. A model with no price is never run.

Before each unit the gate checks `spent + estimate > budget`. The default budget is
`min(15, spent + OpenRouter credits - 0.30)`, read live, so it honours both the project
cap and the balance on the account and always leaves USD 0.30 unspent. Passing the gate
exits with code **3** and makes no call. `--budget N` overrides it.

`results/ledger.jsonl` is the source of truth, one line per finished unit, no prompt
text. Each line carries two costs: `cost_tokens_usd` (reported tokens × list price) and
`cost_credits_usd` (the delta of `total_usage` from the credits endpoint around the
unit). That endpoint settles late, so the runner polls it for up to a minute and records
whether it settled; the budget uses the **larger** of the two, so a late reading can
never make a run look cheaper than it was. Real calls made outside the runner are added
with `uv run sesgo-run ledger --add "note" 0.01`.

## Unattended runs

The runner assumes no TTY (`display="plain"`), prints one timestamped and flushed line
per event, records a failed unit and continues with the next, and is resumable: a unit
that is already `ok` in the ledger is skipped, and an interrupted one is resumed by
`eval_set`. Long runs belong in tmux:

```bash
tmux new -s sesgo -d 'uv run sesgo-run full --reasoning off 2>&1 | tee logs/run-full.txt'
tmux attach -t sesgo
```

Exit codes: `0` done, `1` configuration error, `2` at least one unit failed, `3` the
budget gate stopped the run, `4` the V4 reasoning check failed.

**A killed run still spent money.** A ledger line is only written when a unit finishes,
so `tmux kill-session`, Ctrl-C or a crash would otherwise lose the spend of the attempt.
Before its first call the runner writes `results/inflight.json` (git-ignored) with the
`total_usage` reading of that moment; the next `sesgo-run smoke|pilot|full` finds the
leftover marker, reads the credits endpoint again and appends the difference as a line
with `status: "interrupted"` **before** the budget gate reads the ledger. That line has
no tokens (the killed process wrote no log to count) and does not mark the unit done, so
`eval_set` still resumes it and only the missing samples are requested again.
`sesgo-run ledger` warns while such a marker is still pending.

## Report

`sesgo-report` reads the `.eval` logs and writes two files that are safe to commit,
because they hold aggregates only:

* `results/history.jsonl` — one record per log, keyed by `run_id` + task id, so running
  the report again adds nothing. Every field that can move a number (task version,
  dataset commit, parser version, temperature, reasoning) travels with the metrics.
* `REPORT.md` — in Spanish: summary per split, the anchors against the paper (protocol
  effect and port fidelity), reasoning off against low, unparsed and refusal rates with
  the bias-score bounds, the real cost, the design differences from the paper, and the
  commands. Sections whose configurations have not run say so instead of failing.

`SESGO_RESULTS_DIR` moves `results/` elsewhere, which is how the runner is exercised
without touching the committed ledger.
