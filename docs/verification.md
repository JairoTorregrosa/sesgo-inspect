# Verification

Every part of the package was checked by the thing that can actually prove it wrong: the
authors' published answers for the metrics, real model responses for the parser, the
provider's balance for the ledger, a killed process for resume, a fresh clone for the
README. The full list (V1–V12) with the evidence required for each is in
[`specs/07-verification.md`](../specs/07-verification.md).

## Offline checks you can run

They cost nothing. They need `make data` first; `verify_rescore` also reads the results
workbook that the authors publish in their repository (cloned into `external/`).

```bash
make verify                                  # verify_data + verify_oracle + verify_rescore
uv run python scripts/verify_data.py         # counts, roles, label rule, hashes        (~5 s)
uv run python scripts/verify_oracle.py       # oracle solver over all 4,156 samples     (~100 s)
uv run python scripts/verify_rescore.py      # our metrics against the paper's tables   (~20 s)
uv run python scripts/verify_report.py       # REPORT.md recomputed from raw .eval logs
```

`verify_report.py` needs the `.eval` logs of the runs it checks, so it only works after
you have run a panel yourself.

## Scoring against the published answers: 82 of 84

`verify_rescore.py` feeds the answers the authors published for six models through this
package's scorer and metrics, and compares 84 values with the paper's Tables 2, 3 and A3
at a tolerance of ±0.001. 82 match. The other two are the ambiguous-context accuracy and
bias score of Llama 3.1 8B Uncensored (ours 0.379 / 0.638, Table A3 0.384 / 0.633).

The cause is in the published files, not in rounding. Two upstream workbooks hold
different answer sets for that model: they disagree on exactly 23 rows, all in the gender
category, where one workbook holds "unknown" and the other an unparsed answer. Seven of
those rows are ambiguous, so they are correct in one file and unparsed in the other:
7 / 1,348 = 0.0052, exactly the accuracy gap. Table A3 was computed from one workbook and
Tables 2 and 3 from the other; the pooled row of Table 2 prints 0.638, which this
re-scoring reproduces. The script prints the row-level evidence.

Two more things in the paper are recorded rather than forced: the captions of Tables 2
and 3 are swapped, and the `XENOFOBIA` sheet of the results workbook has rotated model
headers (the script reads the sheet `TODOS` instead).

## The report, recomputed: 135 of 135

`verify_report.py` recomputes 135 numbers of `REPORT.md` from the raw `.eval` logs
without importing the report code. All 135 agree. The same audit found the one known
parser error in 49,872 real samples (a refusal that quoted the text of a single option
was read as an answer); parser version 4 fixes it for new runs, and its effect on the
affected bias score is +0.00005.

## Checks that needed the real world

| Check | How it was verified |
|---|---|
| Protocol replication | GPT-4o mini with the paper's prompt: accuracy 0.803 and bias score 0.199 in ambiguous context, against 0.806 and 0.196 published |
| Parser | Every non-strict parse of 400 real Llama 3.1 8B responses read by hand, plus a random sample of strict ones |
| Reasoning off / low | Reasoning tokens measured in real logs for every configuration; a configuration declared `off` that still reasons fails the smoke stage |
| Ledger | Spend reconciled against the OpenRouter balance; each line keeps both the token cost and the credit delta and bills the larger |
| Resume | A real run killed mid-unit and restarted: exact sample count, no duplicates, the interrupted spend booked |
| Trace viewer | Real logs opened in a real browser; the filter expressions in [usage](usage.md#viewing-traces) are the ones that worked |
| README | Fresh clone, following only the README, ending in a real 8-sample evaluation |

Unit tests exist as regression guards (`make check`), many of them taken from real model
responses. They are not the acceptance criterion.
