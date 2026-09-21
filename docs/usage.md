# Usage

Installation, the local data build, running a model, the trace viewer and every task parameter.
The short version is in the [README](../README.md).

## Install

```bash
make setup                 # uv sync
cp .env.example .env       # then write OPENROUTER_API_KEY=... into .env
```

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). The key comes from
[openrouter.ai/keys](https://openrouter.ai/keys). Inspect reads `.env` from the working
directory; the package never reads the key itself. `.env` is git-ignored — keep it that way.

## Build the data

```bash
make data                  # clones upstream into external/, writes data/*.jsonl + manifest.json
```

Needs `git` and network: it clones `github.com/mvrobles/SESGO` into `external/` (~15 s)
and writes three files — `sesgo-paper-es.jsonl` (4,156 rows, the commit of the paper),
`sesgo-head-es.jsonl` (3,840) and `sesgo-head-en.jsonl` (2,280). There is no English
paper set upstream, so `paper/en` is skipped on purpose.

The build is deterministic: `data/manifest.json` records the upstream commit, the row
counts and a SHA256 per file, and every run checks the hash before it reads a file. A
fresh clone reproduces the same three hashes.

`data/` and `external/` are git-ignored and must stay so — see the
[data notice](../README.md#data-license-citation).

## Run a model

These commands call OpenRouter and spend real money. Start small:

```bash
# 8 real samples, 2 per category: fractions of a cent on a small model
uv run inspect eval sesgo/sesgo --model openrouter/meta-llama/llama-3.1-8b-instruct \
  -T limit_per_category=2
uv run inspect eval sesgo/sesgo --model openrouter/openai/gpt-4o-mini -T categories=racismo,genero
make view                  # open the traces in Inspect View (http://localhost:7575)
```

The `.eval` log lands in `./logs/` (`--log-dir DIR` moves it) and its path is printed on
the last line. Prefer `-T limit_per_category=N` over Inspect's `--limit N`: `--limit`
takes the first N samples in id order, which is one category only, while
`limit_per_category` is a deterministic stratified subset (see the parameter table).

A free dry run without a model: `make smoke` runs 10 samples through `mockllm`, which
answers nothing parseable, so `unparsed_rate` is 1.0 and the bias numbers are the
degenerate ±1.414 — it only proves the pipeline runs end to end. `make oracle` answers
100 samples (25 per category) from the metadata with `sesgo/oracle_solver` and must reach
accuracy 1.0; `uv run python scripts/verify_oracle.py` does the same over all 4,156
samples in both prompt styles (~100 s).

## Viewing traces

```bash
uv run inspect view --log-dir logs --port 7676          # everything under logs/
uv run inspect view --log-dir logs/pilot --port 7676    # one directory
make view                                               # logs/, default port 7575
```

Open `http://localhost:7676`. A missing `--log-dir` is not an error: the server starts
and shows an empty listing. Verified in a real browser on real llama-3.1-8b logs
(V6, `inspect_ai` 0.3.265) and from a fresh clone (V11).

**The header.** The five numbers next to the task name are
`ambig/all/bias_score`, `disambig/all/bias_score`, `ambig/all/accuracy`,
`disambig/all/accuracy`, `ambig/all/unparsed_rate` — the summary of the run. The view
only ever renders five, and this scorer emits 94 metrics, so the rest live behind
**All scoring…**, which opens one wide row you scroll sideways. For the remaining 88
numbers read `REPORT.md` or `results/history.jsonl` instead (both are written by
`sesgo-report`; neither exists before you run it); the modal is for spot checks. A run
that covers only one split — `-T limit_per_category=2` gives ambiguous samples only —
has no `disambig/*` metrics at all, so the header shows fewer than five numbers. The log
**listing** shows `ambig/all/bias_score` in its `SCORE` column, so runs are comparable at
a glance.

**One sample.** Click any row. The transcript shows the system message (paper style),
the full user prompt with the three options, and the raw model answer. The `SCORE`
panel below shows `TARGET`, `ANSWER` (the parsed option: a letter in clean style, the
index `0|1|2` in paper style, empty when nothing parsed), `EXPLANATION` (the stripped
completion), `REASON` and all 13 score keys. Its `METADATA` tab shows `outcome`,
`chosen_role`, `parse_stage`, `parse_detail` and the parser versions. The sample-level
`METADATA` tab shows `category`, `context_condition`, `question_polarity`, `roles`,
`label_role`, `bbq` and `category_split`.

**Filtering.** The `FILTER` box takes a Python-like expression over the sample metadata
and the score keys. These are the expressions that really work:

| Goal | Filter expression |
|---|---|
| One category | `metadata.category == "genero"` |
| One split | `metadata.context_condition == "ambig"` |
| Both at once | `metadata.category == "genero" and metadata.context_condition == "ambig"` |
| Category + split in one key | `metadata.category_split == "racismo/ambig"` |
| Only the negative questions | `metadata.question_polarity == "neg"` |
| Every wrong answer | `correct == 0` |
| Everything the parser could not read | `unparsed == 1` |
| Refusals only | `refusal == 1` |
| Parses that needed the lenient stage (the V5 audit set) | `parse_lenient == 1` |
| Bias against the target group | `ft == 1` |
| The model picked the target group | `chose_target == 1` |
| Combined | `metadata.category == "racismo" and metadata.context_condition == "ambig" and correct == 0` |
| Free text in the prompt | `input_contains("gimnasio")` |

Typing `metadata.` or a score key offers autocompletion with the real values of the log.
Every column header sorts (click `FT`, `CORRECT`, …), and `Columns` hides the score
columns you do not want. Sample ids start with the category
(`genero-es-0001`), so the id column is a usable fallback.

Two limits worth knowing: `Score.metadata` is **not** filterable, so `chosen_role` and
`outcome` cannot be used in an expression — filter on `chose_target` / `chose_other` /
`chose_unknown` and `refusal` / `invalid` / `no_response` instead, which carry the same
information. And the header metrics do not recompute for a filtered subset; they always
describe the whole log.

## Task parameters

`-T name=value` on the command line, or keyword arguments to `sesgo(...)`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `categories` | `str \| list[str] \| None` | all four | `racismo`, `genero`, `clasismo`, `xenofobia`; `-T categories=racismo,genero` |
| `language` | `"es" \| "en"` | `"es"` | Prompt language. The paper set exists in Spanish only |
| `dataset_version` | `"paper" \| "head"` | `"paper"` | `paper` = 4,156 prompts at the commit of the paper; `head` = upstream HEAD |
| `prompt_style` | `"clean" \| "paper"` | `"clean"` | `clean` = Spanish prompt, answer line `RESPUESTA: X`; `paper` = the upstream English system message and `Option N` numbering |
| `shuffle` | `bool \| None` | style default | Shuffle the three options per sample, seeded by the sample id. `None` means `True` for `clean` and `False` for `paper` |
| `temperature` | `float` | `0.75` | Sampling temperature, in `[0, 2]`. The paper uses 0.75 |
| `max_tokens` | `int \| None` | `512` | Token budget per response. Use `None` (`-T max_tokens=null`) for reasoning models |
| `limit_per_category` | `int \| None` | `None` | Deterministic stratified subset per category, for pilots. Rows are taken round-robin over the four cells `ambig/neg`, `ambig/nonneg`, `disambig/neg`, `disambig/nonneg`, so 1 or 2 yields **ambiguous samples only**, 3 is the smallest value that reaches the disambiguated split and 4 covers the four cells evenly |
| `tie_sign` | `-1 \| 0 \| 1` | `1` | Sign of the bias score when Ft equals Fo. `0` reproduces the paper |

Every parameter that can change a number is written into `Task.metadata` of the log,
together with the upstream commit, the data SHA256 and the parser version, so a log
describes itself.

