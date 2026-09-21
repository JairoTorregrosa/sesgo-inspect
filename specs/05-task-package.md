# 05 Task and package

Modules: `src/sesgo/_task.py`, `src/sesgo/_registry.py`, `src/sesgo/__init__.py`, `pyproject.toml`, `Makefile`, `README.md`.

## Contract

```python
TASK_VERSION: Final[str] = "1-A"
DEFAULT_TEMPERATURE: Final[float] = 0.75
DEFAULT_MAX_TOKENS: Final[int] = 512

@task
def sesgo(
    categories: str | list[str] | None = None,     # None = all four; CLI passes "racismo,genero" or a list
    language: Language = "es",
    dataset_version: DatasetVersion = "paper",
    prompt_style: PromptStyle = "clean",
    shuffle: bool | None = None,                   # None -> True for clean, False for paper
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,   # None -> provider default (use for reasoning configs)
    limit_per_category: int | None = None,
    tie_sign: int = 1,
) -> Task
```

- Returns `Task(dataset=sesgo_dataset(...), solver=get_default_solver(prompt_style), scorer=sesgo_scorer(prompt_style), metrics=[sesgo_metrics(tie_sign)], config=GenerateConfig(temperature=..., max_tokens=...), version=TASK_VERSION, metadata={...})`.
- `Task.metadata`: `dataset_version`, upstream commit, `prompt_style`, `shuffle`, `parser_version`, `refusal_patterns_version`, `tie_sign`, manifest SHA256 of the data file. These make a log self-describing for monitoring.
- Validation with informative `ValueError`: unknown category / language / version / style; `prompt_style="paper"` with `shuffle=True`; `prompt_style="paper"` with `language="en"` is allowed; `paper` + `dataset_version`/`language` pair without data (spec 01); temperature outside [0, 2]; `limit_per_category < 1`; `tie_sign` not in (−1, 0, 1).
- Dataset, solver, and scorer are overridable with `task_with` (BEST_PRACTICES). Defaults come from `get_default_*` factories.
- Version policy: bump N (`2-A`) for a change in scoring, prompt, dataset, parser, refusal patterns, or routing. Bump the letter for interface changes.

### Oracle solver (test utility, public)

```python
@solver
def oracle_solver(wrong: bool = False) -> Solver
```

Writes `RESPUESTA: <letter>` of the correct option (or, with `wrong=True`, of a deterministic incorrect non-`unknown`-biased option) into `state.output` without a model call. With `prompt_style="paper"` it writes `Option N: <text>`. Used by spec 07 check V3.

### Package

- `pyproject.toml`: build `hatchling`; `requires-python = ">=3.13"`; dependencies `inspect-ai==0.3.265`, `openai` (needed by the OpenRouter provider), `openpyxl`, `pandas` (converter only), `pyyaml`, `tiktoken`, `python-dotenv`, `httpx`; dev group `pytest`, `pytest-asyncio`, `ruff`, `pyright`.
- `[project.entry-points.inspect_ai] sesgo = "sesgo._registry"`. `_registry.py` holds imports only (`# ruff: noqa: F401`).
- `[project.scripts]`: `sesgo-data = "sesgo._data:main"`, `sesgo-run = "sesgo._runner:main"`, `sesgo-report = "sesgo._report:main"`.
- `src/sesgo/py.typed` exists. `__init__.py` exports with `__all__`: `sesgo`, `sesgo_scorer`, `sesgo_metrics`, `sesgo_prompt`, `oracle_solver`, `get_default_solver`, `load_records`, `record_to_sample`, `summarize`, `bias_score`, `TASK_VERSION`.
- ruff: `line-length = 100`, `target-version = "py313"`, rules `E,F,I,UP,B,SIM,D` with Google docstring convention (tests exempt from `D`). pyright: `typeCheckingMode = "strict"` for `src/`; if `inspect_ai` stubs force ignores, scope them to the line with a reason.
- pytest markers: `slow`, `dataset_download` (needs `external/` or `data/`), `real_api` (spends money; never runs by default).
- `Makefile`: `setup`, `data`, `check` (ruff + pyright + pytest), `smoke` (mockllm, limit 10), `view`.
- `README.md`: short, declarative: what, install, data build, run one model, open view, add a model, where the specs are. States that data and logs must not be published (no upstream license).

## Edge cases

- `inspect eval sesgo/sesgo -T categories=racismo,genero` and `-T categories=racismo` both work.
- The task must import without the data present; only calling `sesgo()` needs data.
- `.env` is loaded by Inspect automatically from the working directory; the package does not read the key itself.

## Acceptance

1. `uv sync` then `uv run inspect eval sesgo/sesgo --model mockllm/model --limit 10` runs and writes a log.
2. `make check` is green: ruff, pyright strict (0 errors), pytest.
3. Each validation rule has a unit test.
4. `uv run inspect eval sesgo/sesgo --model mockllm/model --solver sesgo/oracle_solver` (or the Python equivalent with `task_with`) gives accuracy 1.0 in both splits.
