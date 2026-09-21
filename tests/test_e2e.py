"""End-to-end tests of the SESGO task over a synthetic dataset (spec 05).

The oracle solver answers every sample without a model call, so these tests exercise the
whole chain - dataset, prompt, parser, scorer, metrics, log - in about a second and with
no network. `scripts/verify_oracle.py` runs the same idea over the real paper dataset.
"""

import math
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

from sesgo import TASK_VERSION, oracle_solver, sesgo
from sesgo._data import jsonl_path, write_dataset
from sesgo._metrics import ALL_GROUP, CATEGORY_METRIC_NAMES, GROUP_METRIC_NAMES, metric_key
from sesgo._parser import PARSER_VERSION, REFUSAL_PATTERNS_VERSION
from sesgo._types import CATEGORIES, SPLITS, Category, Option, Polarity, PromptStyle, Record, Role

POLARITIES: tuple[Polarity, ...] = ("neg", "nonneg")
MODEL = "mockllm/model"


def synthetic_corpus() -> list[Record]:
    """One record per category, split, polarity and disambiguated label: 32 records."""
    records: list[Record] = []
    for category in CATEGORIES:
        counter = 0
        for split in SPLITS:
            for polarity in POLARITIES:
                for label in ("target", "other"):
                    counter += 1
                    role: Role = "unknown" if split == "ambig" else label
                    options: list[Option] = [
                        {"text": f"{category} otro", "role": "other"},
                        {"text": f"{category} objetivo", "role": "target"},
                        {"text": "No hay suficiente informacion", "role": "unknown"},
                    ]
                    records.append(
                        {
                            "id": f"{category}-es-{counter:04d}",
                            "category": category,
                            "language": "es",
                            "dataset_version": "paper",
                            "context_condition": split,
                            "question_polarity": polarity,
                            "context": "Un contexto de prueba.",
                            "question": "¿Quien fue?",
                            "options": options,
                            "label_role": role,
                            "bbq": False,
                            "source": {
                                "commit": "0" * 40,
                                "file": "prompts/test.xlsx",
                                "sheet": "Sheet1",
                                "row": counter,
                            },
                        }
                    )
    records.sort(key=lambda record: record["id"])
    return records


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the synthetic dataset and make it the default data directory."""
    write_dataset(synthetic_corpus(), "paper", "es", tmp_path)
    monkeypatch.setenv("SESGO_DATA_DIR", str(tmp_path))
    return tmp_path


def run_oracle(
    log_dir: Path,
    prompt_style: PromptStyle = "clean",
    wrong: bool = False,
    tie_sign: int = 1,
) -> EvalLog:
    """Run the task with the oracle solver and return the log."""
    task = sesgo(prompt_style=prompt_style, tie_sign=tie_sign)
    logs = inspect_eval(
        task,
        model=MODEL,
        solver=oracle_solver(wrong=wrong, prompt_style=prompt_style),
        log_dir=str(log_dir),
        display="none",
    )
    log = logs[0]
    assert log.status == "success", log.error
    return log


def metrics_of(log: EvalLog) -> dict[str, float]:
    """Read the named metrics of the single scorer of a log."""
    assert log.results is not None
    assert len(log.results.scores) == 1
    return {name: metric.value for name, metric in log.results.scores[0].metrics.items()}


@pytest.mark.parametrize("prompt_style", ["clean", "paper"])
def test_oracle_is_perfect(data_dir: Path, tmp_path: Path, prompt_style: PromptStyle) -> None:
    assert data_dir.is_dir()
    log = run_oracle(tmp_path / "logs", prompt_style)
    values = metrics_of(log)
    groups: list[str] = [ALL_GROUP, *CATEGORIES]
    for split in SPLITS:
        for group in groups:
            assert values[metric_key(split, group, "accuracy")] == 1.0
            assert values[metric_key(split, group, "bias_score")] == 0.0
            assert values[metric_key(split, group, "bias_score_lo")] == 0.0
            assert values[metric_key(split, group, "bias_score_hi")] == 0.0
            assert values[metric_key(split, group, "unparsed_rate")] == 0.0
        for name in ("ft", "fo", "ft_minus_fo", "invalid_rate", "refusal_rate"):
            assert values[metric_key(split, ALL_GROUP, name)] == 0.0
        for name in ("no_response_rate", "truncation_rate", "parse_lenient_rate"):
            assert values[metric_key(split, ALL_GROUP, name)] == 0.0
        assert values[metric_key(split, ALL_GROUP, "parse_strict_rate")] == 1.0
        assert values[metric_key(split, ALL_GROUP, "coverage")] == 1.0
        assert values[metric_key(split, ALL_GROUP, "valid_accuracy")] == 1.0


@pytest.mark.parametrize("prompt_style", ["clean", "paper"])
def test_oracle_wrong_is_a_known_failure(
    data_dir: Path, tmp_path: Path, prompt_style: PromptStyle
) -> None:
    assert data_dir.is_dir()
    log = run_oracle(tmp_path / "logs", prompt_style, wrong=True)
    values = metrics_of(log)
    for split in SPLITS:
        for group in [ALL_GROUP, *CATEGORIES]:
            assert values[metric_key(split, group, "accuracy")] == 0.0
            assert values[metric_key(split, group, "unparsed_rate")] == 0.0
    # Every ambiguous error is an error against the target group, so Ft is 1 and Fo is 0.
    assert values[metric_key("ambig", ALL_GROUP, "ft")] == 1.0
    assert values[metric_key("ambig", ALL_GROUP, "fo")] == 0.0
    assert values[metric_key("ambig", ALL_GROUP, "bias_score")] == pytest.approx(math.sqrt(2.0))
    # The synthetic corpus is balanced, so the disambiguated errors tie: sigma is tie_sign.
    assert values[metric_key("disambig", ALL_GROUP, "ft")] == 0.5
    assert values[metric_key("disambig", ALL_GROUP, "fo")] == 0.5
    assert values[metric_key("disambig", ALL_GROUP, "bias_score")] == 1.0


def test_tie_sign_reaches_the_metric(data_dir: Path, tmp_path: Path) -> None:
    assert data_dir.is_dir()
    values = metrics_of(run_oracle(tmp_path / "logs", wrong=True, tie_sign=-1))
    assert values[metric_key("disambig", ALL_GROUP, "bias_score")] == -1.0


def test_log_carries_the_task_metadata_and_every_metric(data_dir: Path, tmp_path: Path) -> None:
    assert data_dir.is_dir()
    log = run_oracle(tmp_path / "logs")
    assert log.eval.task_version == TASK_VERSION
    metadata: dict[str, Any] = log.eval.metadata or {}
    assert metadata["dataset_version"] == "paper"
    assert metadata["prompt_style"] == "clean"
    assert metadata["parser_version"] == PARSER_VERSION
    assert metadata["refusal_patterns_version"] == REFUSAL_PATTERNS_VERSION
    assert metadata["tie_sign"] == 1
    assert metadata["shuffle"] is True
    assert len(metadata["data_sha256"]) == 64
    assert metadata["samples"] == 32

    expected = {
        metric_key(split, ALL_GROUP, name) for split in SPLITS for name in GROUP_METRIC_NAMES
    } | {
        metric_key(split, category, name)
        for split in SPLITS
        for category in CATEGORIES
        for name in CATEGORY_METRIC_NAMES
    }
    assert set(metrics_of(log)) == expected


def test_sample_log_is_readable(data_dir: Path, tmp_path: Path) -> None:
    assert data_dir.is_dir()
    log = run_oracle(tmp_path / "logs")
    assert log.samples is not None
    sample = log.samples[0]
    score = next(iter(sample.scores.values())) if sample.scores else None
    assert score is not None
    assert score.answer in {"A", "B", "C"}
    assert score.metadata is not None
    assert score.metadata["outcome"] == "answered"
    assert score.metadata["parse_stage"] == "strict"
    metadata: dict[str, Any] = sample.metadata
    assert metadata["category"] in CATEGORIES
    assert metadata["context_condition"] in SPLITS
    assert "Contexto:" in str(sample.messages[0].content)
    assert "RESPUESTA" in str(sample.messages[-1].content)


def data_is_built() -> bool:
    """Report whether the real paper dataset is present."""
    return jsonl_path("paper", "es").is_file()


@pytest.mark.dataset_download
@pytest.mark.skipif(not data_is_built(), reason="data/ is not built")
def test_registry_resolves_the_task_by_name(tmp_path: Path) -> None:
    logs = inspect_eval(
        "sesgo/sesgo",
        model=MODEL,
        task_args={"limit_per_category": 1},
        solver=oracle_solver(),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    log = logs[0]
    assert log.status == "success"
    assert log.eval.task == "sesgo/sesgo"
    values = metrics_of(log)
    assert values[metric_key("ambig", ALL_GROUP, "accuracy")] == 1.0
    categories: set[Category] = set()
    assert log.samples is not None
    for sample in log.samples:
        categories.add(sample.metadata["category"])
    assert categories == set(CATEGORIES)
