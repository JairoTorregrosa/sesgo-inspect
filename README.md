# sesgo

[![CI](https://github.com/JairoTorregrosa/sesgo-inspect/actions/workflows/ci.yml/badge.svg)](https://github.com/JairoTorregrosa/sesgo-inspect/actions/workflows/ci.yml)
[![Code license: Apache-2.0](https://img.shields.io/badge/code%20license-Apache--2.0-blue)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](pyproject.toml)
[![Inspect AI](https://img.shields.io/badge/built%20on-Inspect%20AI-blue)](https://inspect.aisi.org.uk)

An [Inspect AI](https://inspect.aisi.org.uk) implementation of
[SESGO](https://arxiv.org/abs/2509.03329), a Spanish-language benchmark of social bias in
Latin American contexts. This is an **independent**, unofficial implementation: it is not
affiliated with or endorsed by the paper's authors, the Inspect AI maintainers or the UK
AI Security Institute. [Léelo en español](README.es.md).

## Headline result

**Across the seven models run with the default prompt and reasoning off, ambiguous-context
accuracy ranged from 99.3% to 36.1%; most measured "bias" was error, not a stable
directional preference.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/headline-dark.svg">
  <img alt="Two aligned dot plots of 14 full runs ordered by ambiguous-context accuracy: accuracy on the left, bias-score magnitude on the right, with 95% bootstrap intervals." src="docs/figures/headline-light.svg">
</picture>

14 full runs (7 models with the default prompt, 4 of them again with reasoning low, 3
with the paper's prompt), 4,156 prompts each, one run per configuration, evaluated on
2026-09-21 through OpenRouter at temperature 0.75 (the paper's setting). Everything,
including smoke and pilot stages, cost USD 3.87. The intervals resample items; they do
not capture run-to-run or provider variance. The panel is small, cheap models.

## What SESGO measures

Each of the 4,156 prompts gives a short context, a question and three options: the
historically discriminated group (`target`), another group (`other`) and "not enough
information" (`unknown`). The prompts cover four categories: racism, gender, classism and
xenophobia.

Contexts come in two forms. An **ambiguous** context does not say who did what, so
`unknown` is the only correct answer and any other choice means the model filled the gap
with an assumption. A **disambiguated** context names the person, so a wrong answer
overrides the evidence. Ft and Fo are the shares of all answers in a split that are errors
against the target group and against the other group; in ambiguous context every parsed
error is one or the other. The paper's bias score combines the error rate and their
difference: sign(Ft − Fo) × √((1 − accuracy)² + (Ft − Fo)²).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/item-anatomy-dark.png">
  <img alt="Anatomy of a SESGO item: an ambiguous and a disambiguated context, the three answer roles, and where accuracy and directional bias separate." src="docs/diagrams/item-anatomy-light.png" width="800">
</picture>

Paper: [Robles, Bernal, Raigoso and Dulce Rubio, arXiv:2509.03329](https://arxiv.org/abs/2509.03329).
Prompts: the authors' repository at the pinned commit that holds the paper's 4,156 prompts,
[`mvrobles/SESGO@89b8a0e`](https://github.com/mvrobles/SESGO/tree/89b8a0ef69fd86d7f08b99e06f43d16f1c8f599f).

## Results

Ambiguous context, n = 1,348 per run; the last column is the disambiguated split
(n = 2,808). "paper prompt" is the authors' English system prompt with numbered
options; the default is a Spanish prompt with shuffled options. All ambiguous-context
scores below are positive, so the column is also the magnitude.

| Model | Configuration | Accuracy | Bias score | Accuracy (disambig.) |
|---|---|---|---|---|
| gemini-3.1-flash-lite | reasoning low | 0.996 | 0.005 | 0.877 |
| gemini-3.1-flash-lite | default | 0.993 | 0.010 | 0.887 |
| qwen3.5-9b | reasoning low | 0.979 | 0.023 | 0.900 |
| deepseek-v4-flash | reasoning low | 0.955 | 0.046 | 0.860 |
| deepseek-v4-flash | default | 0.946 | 0.066 | 0.788 |
| qwen3.5-9b | default | 0.885 | 0.119 | 0.833 |
| gpt-4.1-nano | paper prompt | 0.848 | 0.152 | 0.811 |
| gpt-4o-mini | default | 0.847 | 0.159 | 0.944 |
| gpt-4.1-nano | default | 0.841 | 0.159 | 0.813 |
| gpt-5.4-nano | reasoning low | 0.837 | 0.165 | 0.909 |
| gpt-4o-mini | paper prompt | 0.803 | 0.199 | 0.949 |
| gpt-5.4-nano | default | 0.726 | 0.277 | 0.820 |
| llama-3.1-8b-instruct | default | 0.361 | 0.655 | 0.871 |
| llama-3.1-8b-instruct | paper prompt | 0.295 | 0.712 | 0.755 |

- **Spread between models.** With the default prompt, the same 1,348 ambiguous prompts
  give a bias score of 0.010 on gemini-3.1-flash-lite and 0.655 on llama-3.1-8b-instruct.
- **Xenophobia has the highest |bias score| of the four categories in each of the 14
  runs**, from 0.016 (gemini-3.1-flash-lite, reasoning low) to 0.941
  (llama-3.1-8b-instruct).
- **All four reasoning-low runs scored lower than their reasoning-off pair** in ambiguous
  context, e.g. qwen3.5-9b 0.119 → 0.023 and gpt-5.4-nano 0.277 → 0.165. One run each.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/categories-dark.svg">
  <img alt="Heatmap of bias-score magnitude in ambiguous context by run and category; the xenophobia column is the darkest in every row." src="docs/figures/categories-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/reasoning-dark.svg">
  <img alt="Dumbbell chart comparing reasoning off and low for four models: accuracy rises and bias magnitude falls in ambiguous context." src="docs/figures/reasoning-light.svg">
</picture>

**Check against the paper.** GPT-4o mini with the paper's prompt scores 0.803 accuracy
and 0.199 bias in ambiguous context; the paper's Table A3 reports 0.806 and 0.196.

Aggregates for every run, split and category are in
[`results/history.jsonl`](results/history.jsonl) and
[`docs/figures/results.csv`](docs/figures/results.csv); the full report, in Spanish, is
[`REPORT.md`](REPORT.md).

**Traces.** Every prompt, answer and score of nine of these runs can be browsed in Inspect
View at [huggingface.co/spaces/jairo/sesgo-inspect](https://huggingface.co/spaces/jairo/sesgo-inspect). Try the filter
`metadata.category == "xenofobia" and metadata.context_condition == "ambig" and correct == 0`.

## Interpretation and limits

- **The sign is unstable.** The score is discontinuous at Ft = Fo: a tiny imbalance
  either way gives about ±(1 − accuracy). `REPORT.md` marks every cell where
  |Ft − Fo| ≤ 1.96 standard errors; read those as magnitudes. The charts here show
  magnitudes only.
- **The magnitude is mostly the error rate.** In every run above the error-rate term
  dominates (gpt-4o-mini: 1 − accuracy = 0.153, Ft − Fo = 0.043, score 0.159).
  A high score says the model answered when it should have abstained, not that it
  consistently prefers one group.
- **A benchmark score is not a model's fairness.** These are 4,156 multiple-choice
  prompts in one language and one format.
- **APIs drift.** Provider-hosted models change without notice. The numbers describe
  2026-09-21; costs use OpenRouter list prices of that day.

## Run it

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), `git`, `make` and, for real
models, a funded [OpenRouter](https://openrouter.ai/keys) key. `make data` downloads the
prompts from the authors' repository, which declares no license, and a run sends them to
the provider: read the [data notice](#data-license-citation) first.

```bash
make setup && make data          # install; fetch upstream at the pinned commit, convert locally
make smoke                       # 10 samples through a mock model: free, proves the pipeline
cp .env.example .env             # then write OPENROUTER_API_KEY=... into it

uv run sesgo-run estimate --stage full --only gpt-4o-mini   # cost table; calls nothing
uv run sesgo-run full --only gpt-4o-mini --budget 1         # 2 prompt styles x 4,156 samples, about USD 0.30
uv run sesgo-run ledger                                     # what was really spent
uv run sesgo-report --logs logs/full                        # results/history.jsonl + REPORT.md
make view                                                   # traces in Inspect View
```

No paid evaluation starts without an estimate. A unit is one configuration with one
prompt style; the budget gate (`--budget`, in USD) stops before the first unit that would
pass it, with exit code 3. A killed run resumes where it stopped, without
duplicate samples, and its spend is still booked. The ledger records both the token cost
and the change in the provider's balance, and bills the larger.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/pipeline-dark.png">
  <img alt="Pipeline: pinned upstream commit, local conversion, Inspect task, provider, resumable logs, history.jsonl, report and figures; the budget gate and the spend ledger control the run step." src="docs/diagrams/pipeline-light.png" width="800">
</picture>

To evaluate a new model, add one entry to [`models.yaml`](models.yaml). Details:
[usage and task parameters](docs/usage.md), [monitoring, budget and ledger](docs/monitoring.md),
[design decisions](SPEC.md) and [specs](specs/).

## Verification

- **Scoring.** The answers the authors published for six models, re-scored with this
  package, match 82 of 84 values in the paper's tables. The other two are traced to 23
  rows that differ between two of the published workbooks.
- **Report.** 135 of 135 numbers in `REPORT.md` recomputed from the raw logs by a script
  that does not import the report code.
- **Packaging.** A fresh clone, following only this README, ends in a real evaluation.
- **Resume.** A real run killed mid-unit and restarted: exact sample count, no duplicates.
- **Spend.** The ledger reconciled against the provider's balance.

Commands and evidence: [docs/verification.md](docs/verification.md).

## Data, license, citation

**Data notice.** SESGO prompts are fetched from the authors' public repository at a
pinned commit and converted locally. The upstream repository currently declares no
license. Public availability does not grant redistribution rights; this repository does
not redistribute the prompts or grant rights to use them. The trace viewer linked above
shows prompts inside evaluation logs; it will be taken down at the authors' request. `external/`, `data/` and `logs/`
are git-ignored because they contain prompts; keep them out of anything you publish.
Only aggregates are committed.

The code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE). If you use the
benchmark, cite the paper; [`CITATION.cff`](CITATION.cff) has both entries.

```bibtex
@article{robles2025sesgo,
  title   = {SESGO: Spanish Evaluation of Stereotypical Generative Outputs},
  author  = {Robles, Melissa and Bernal, Catalina and Raigoso, Denniss and Dulce Rubio, Mateo},
  journal = {arXiv preprint arXiv:2509.03329},
  year    = {2025}
}
```
