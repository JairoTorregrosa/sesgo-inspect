# 07 Verification

Real evidence outranks unit tests (D17). A check is independent: it reads the spec and the code, and reports findings with evidence (command, output excerpt, log file, sample id). It fixes nothing silently: every fix is recorded with the evidence that it works.

## Principle: every component has a real verifier

The question for each piece of work is "what is the real-world verifier of this?", and the answer is never a mock, a stub, or a unit test alone. Unit tests guard pure arithmetic and regressions found by real evidence; they are not acceptance.

| Component | Real verifier |
|---|---|
| Data conversion | The real upstream git objects at the pinned commits; counts against paper Table 1 (total and original/BBQ-adapted split per category) |
| Scoring + metrics | The authors' published per-row answers re-scored against the paper tables (V2), plus an independent recomputation that imports nothing from `sesgo` |
| Prompt + parser | Real completions from real models; a human-style read of every non-strict parse (V5); A/B of prompt wording on the same real samples |
| Reasoning switches | Reasoning tokens measured in real logs per model (V4) |
| Cost model + budget gate | OpenRouter credits balance before/after real runs (V7); the gate proven by a real refused run |
| Resume / failure handling | A real run killed mid-way (`tmux kill`/SIGINT) and resumed: no duplicated samples, no double spend beyond the in-flight requests (V9) |
| Traces | The real `inspect view` server opened in a real browser on a real log, screenshots read by a verifier (V6) |
| Report + history | Numbers in `REPORT.md` recomputed independently from the raw `.eval` files by a script that does not import `sesgo._report` (V10) |
| Packaging / "new user" path | A fresh `git clone` of this repo into a temp dir: `uv sync`, `sesgo-data build`, real 8-sample eval, `inspect view` — following only the README (V11) |
| Monitoring path | Adding one new model line to `models.yaml` and running `sesgo-run smoke/pilot` on it for real, then seeing it in the report (V12) |

Scripts live in `scripts/` and are committed. They print a PASS/FAIL line per check and exit non-zero on failure.

## Checks

| Id | Check | Evidence | Cost |
|---|---|---|---|
| V1 | Data: counts, roles, label rule, shuffle balance, hash failure (spec 01 acceptance 1–6) | `scripts/verify_data.py` output | 0 |
| V2 | Metrics against the paper: re-score `results_prompts/Resultados_agregados_vf_T075.xlsx`, sheet `TODOS`, from `external/SESGO` HEAD | `scripts/verify_rescore.py` table: ours vs Table A3 | 0 |
| V3 | Oracle: full dataset, `oracle_solver`, both styles → accuracy 1.0, all rates 0, bias_score 0.0; `wrong=True` → accuracy 0.0 and a known sign | log + printed metrics | 0 |
| V4 | Real API smoke: each panel config, 3 samples per category: response arrives, parses, reasoning off really gives ~0 reasoning tokens, low gives > 0 | logs in `logs/smoke/` | < $0.05 |
| V5 | Real parser audit: `llama-3.1-8b-instruct`, 200 samples, clean and paper styles. Read every sample whose stage is not `strict`, and 30 random `strict` ones. Every parse must be right | list of sample ids with verdicts | < $0.01 |
| V6 | Traces: open the V5 log with `inspect view` (headless browser screenshot or `inspect log dump`): prompt, raw response, parsed answer, correct flag, bias metadata are visible; samples can be filtered by `category` and `context_condition`; metrics are named | screenshot paths + notes | 0 |
| V7 | Cost: ledger total vs OpenRouter credits delta agree within 20 % | `sesgo-run ledger` | 0 |
| V8 | Quality gate: `make check` green; `git status` clean of data | output | 0 |
| V9 | Resume: kill a real pilot unit mid-run, run the same command again; sample count is exact, no duplicates, ledger is sane | log + ledger lines | < $0.01 |
| V10 | Report truth: recompute 10 headline numbers of `REPORT.md` from raw `.eval` files without importing `sesgo._report` | script output | 0 |
| V11 | New-user path from a fresh clone, README only, real 8-sample eval | transcript | < $0.01 |
| V12 | Monitoring path: add one new cheap model to `models.yaml`, real smoke + pilot, appears in report | logs + report diff | < $0.05 |

### V2 details

Columns of `TODOS`: `tipo` (category), `question_polarity`, `context_condition`, `label`, `target`, `other`, and one column per model with the parsed option (0, 1, 2, NaN, or `'Rev'`). NaN and `'Rev'` are unparsed. Upstream roles: index 1 = target, 0 = other, 2 = unknown. Build `Row`s through `score_choice` (spec 03) and `summarize` (spec 04) with `tie_sign=0`. Compare pooled values per model and split against Table A3 (T = 0.75):

| Model | disambig acc / Ft−Fo / bias | ambig acc / Ft−Fo / bias |
|---|---|---|
| GPT-4o mini | 0.926 / −0.002 / −0.074 | 0.806 / 0.024 / 0.196 |
| Llama 3.1 Instruct | 0.858 / −0.018 / −0.143 | 0.320 / 0.139 / 0.694 |
| Llama 3.1 Uncensored | 0.880 / −0.006 / −0.120 | 0.384 / 0.148 / 0.633 |
| DeepSeek R1 | 0.738 / 0.007 / 0.262 | 0.899 / −0.001 / −0.101 |
| Gemini 2.0 Flash | 0.890 / 0.009 / 0.110 | 0.898 / 0.041 / 0.110 |
| Claude 3.5 Haiku | 0.773 / −0.013 / −0.227 | 0.772 / 0.077 / 0.241 |

Tolerance ±0.001. Also compare per-category bias scores with paper Tables 2 (values are **ambiguous**) and 3 (values are **disambiguated**); the captions in the paper are swapped. Known paper inconsistencies are documented, not forced: Llama Uncensored ambiguous 0.633 (A3) vs 0.638 (Table 2); rotated model headers in sheet `XENOFOBIA` (use `TODOS`). Any other difference must be explained with row-level evidence. The reference values live in `src/sesgo/_paper.py`.

## Spend plan (D14), cap USD 15

1. V4 + V5 smoke.
2. `sesgo-run pilot --reasoning all` (200 samples × every unit). Check: unparsed rates, truncation, tokens per sample, real cost against estimate. Fix problems before the full run. A parser or prompt fix after the pilot means the pilot is re-run for the affected units.
3. `sesgo-run full --reasoning off` (6 configs clean + 2 anchors paper style). This reaches "done".
4. `sesgo-run estimate --stage full --reasoning low` with measured tokens. Run cheapest first. The budget gate stops before the unit that would pass USD 15. Units that did not fit are listed in `REPORT.md` as pending.

Long commands run in tmux session `sesgo` with output teed to `logs/run-*.txt`.

## Definition of done

- Full Spanish run complete for at least the six reasoning-off configs; `.eval` logs open in `inspect view`.
- V1–V8 pass, with the evidence of each one summarized in `REPORT.md`.
- `make check` green. `results/history.jsonl`, `results/ledger.jsonl`, `REPORT.md` committed. No data, logs, or key in git.
