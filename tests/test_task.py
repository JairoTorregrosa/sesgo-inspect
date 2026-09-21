"""Unit tests for `sesgo._task` and `sesgo._oracle` (spec 05).

No network, no model call and no upstream text: the data-dependent tests write a
synthetic dataset into a temporary directory and point `SESGO_DATA_DIR` at it.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import GenerateConfig

from sesgo import TASK_VERSION, sesgo
from sesgo._data import COMMITS, CONVERTER_VERSION, write_dataset
from sesgo._oracle import oracle_answer, oracle_index, oracle_role, oracle_solver
from sesgo._parser import PARSER_VERSION, REFUSAL_PATTERNS_VERSION
from sesgo._task import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    resolve_categories,
    resolve_shuffle,
)
from sesgo._types import (
    CATEGORIES,
    SPLITS,
    Category,
    DatasetVersion,
    Language,
    Option,
    Polarity,
    PromptStyle,
    Record,
    Role,
    Split,
)

POLARITIES: tuple[Polarity, ...] = ("neg", "nonneg")
ROLES_IN_ORDER: list[Role] = ["other", "target", "unknown"]


def make_record(
    record_id: str,
    category: Category,
    split: Split,
    polarity: Polarity,
    label_role: Role,
    dataset_version: DatasetVersion = "paper",
    language: Language = "es",
) -> Record:
    """Build one synthetic record; no upstream text is used."""
    options: list[Option] = [
        {"text": "opcion other", "role": "other"},
        {"text": "opcion target", "role": "target"},
        {"text": "opcion unknown", "role": "unknown"},
    ]
    return {
        "id": record_id,
        "category": category,
        "language": language,
        "dataset_version": dataset_version,
        "context_condition": split,
        "question_polarity": polarity,
        "context": "contexto",
        "question": "¿pregunta?",
        "options": options,
        "label_role": label_role,
        "bbq": False,
        "source": {"commit": "0" * 40, "file": "prompts/test.xlsx", "sheet": "Sheet1", "row": 1},
    }


def synthetic_corpus(
    dataset_version: DatasetVersion = "paper", language: Language = "es"
) -> list[Record]:
    """Two records per category and split/polarity cell, both disambiguated labels."""
    records: list[Record] = []
    for category in CATEGORIES:
        counter = 0
        for split in SPLITS:
            for polarity in POLARITIES:
                for label_role in ("target", "other"):
                    counter += 1
                    role: Role = "unknown" if split == "ambig" else label_role
                    records.append(
                        make_record(
                            f"{category}-{language}-{counter:04d}",
                            category,
                            split,
                            polarity,
                            role,
                            dataset_version,
                            language,
                        )
                    )
    records.sort(key=lambda record: record["id"])
    return records


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the synthetic datasets and make them the default data directory."""
    write_dataset(synthetic_corpus(), "paper", "es", tmp_path)
    write_dataset(synthetic_corpus("head", "en"), "head", "en", tmp_path)
    monkeypatch.setenv("SESGO_DATA_DIR", str(tmp_path))
    return tmp_path


# --- resolve_categories ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, CATEGORIES),
        ("racismo", ("racismo",)),
        ("racismo,genero", ("racismo", "genero")),
        ("genero, racismo", ("racismo", "genero")),
        (["racismo", "genero"], ("racismo", "genero")),
        (["genero,racismo"], ("racismo", "genero")),
        (["racismo", "racismo"], ("racismo",)),
    ],
)
def test_resolve_categories_forms(value: str | list[str] | None, expected: tuple[str, ...]) -> None:
    assert resolve_categories(value) == expected


@pytest.mark.parametrize("value", ["", " ", "racismo,", [], ["racismo", ""]])
def test_resolve_categories_rejects_empty(value: str | list[str]) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_categories(value)


@pytest.mark.parametrize("value", ["racism", "racismo,edad", ["edad"]])
def test_resolve_categories_rejects_unknown(value: str | list[str]) -> None:
    with pytest.raises(ValueError, match="unknown categories"):
        resolve_categories(value)


# --- resolve_shuffle ------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt_style", "shuffle", "expected"),
    [
        ("clean", None, True),
        ("paper", None, False),
        ("clean", False, False),
        ("clean", True, True),
        ("paper", False, False),
    ],
)
def test_resolve_shuffle(prompt_style: PromptStyle, shuffle: bool | None, expected: bool) -> None:
    assert resolve_shuffle(prompt_style, shuffle) is expected


def test_resolve_shuffle_rejects_paper_with_shuffle() -> None:
    with pytest.raises(ValueError, match="requires shuffle=False"):
        resolve_shuffle("paper", True)


# --- one row per validation rule of spec 05 -------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"categories": "edad"}, "unknown categories"),
        ({"language": "fr"}, "unknown language"),
        ({"dataset_version": "v2"}, "unknown dataset_version"),
        ({"prompt_style": "raw"}, "unknown prompt_style"),
        ({"prompt_style": "paper", "shuffle": True}, "requires shuffle=False"),
        ({"dataset_version": "paper", "language": "en"}, "does not exist upstream"),
        ({"temperature": -0.1}, "temperature must be in"),
        ({"temperature": 2.1}, "temperature must be in"),
        ({"max_tokens": 0}, "max_tokens must be"),
        ({"limit_per_category": 0}, "limit_per_category must be"),
        ({"tie_sign": 2}, "tie_sign must be"),
        ({"tie_sign": -2}, "tie_sign must be"),
    ],
)
def test_rejects_invalid_parameters(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        sesgo(**kwargs)


def test_paper_style_with_english_is_allowed(data_dir: Path) -> None:
    assert data_dir.is_dir()
    task = sesgo(prompt_style="paper", dataset_version="head", language="en", shuffle=False)
    assert task.metadata is not None
    assert task.metadata["upstream_commit"] == COMMITS["head"]
    assert task.metadata["shuffle"] is False
    assert task.metadata["data_file"] == "sesgo-head-en.jsonl"


# --- task construction ----------------------------------------------------------


def test_task_metadata_is_self_describing(data_dir: Path) -> None:
    digest = hashlib.sha256((data_dir / "sesgo-paper-es.jsonl").read_bytes()).hexdigest()
    task = sesgo(categories="racismo", limit_per_category=4, tie_sign=0)
    assert task.metadata == {
        "task_version": TASK_VERSION,
        "dataset_version": "paper",
        "language": "es",
        "upstream_commit": COMMITS["paper"],
        "converter_version": CONVERTER_VERSION,
        "data_file": "sesgo-paper-es.jsonl",
        "data_sha256": digest,
        "categories": ["racismo"],
        "limit_per_category": 4,
        "samples": 4,
        "prompt_style": "clean",
        "shuffle": True,
        "parser_version": PARSER_VERSION,
        "refusal_patterns_version": REFUSAL_PATTERNS_VERSION,
        "tie_sign": 0,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }


def test_task_config_and_version(data_dir: Path) -> None:
    assert data_dir.is_dir()
    task = sesgo(temperature=0.0, max_tokens=None)
    assert task.version == TASK_VERSION
    assert task.config == GenerateConfig(temperature=0.0, max_tokens=None)
    assert task.config.max_tokens is None


def test_paper_style_keeps_upstream_option_order(data_dir: Path) -> None:
    assert data_dir.is_dir()
    task = sesgo(prompt_style="paper")
    assert task.metadata is not None
    assert task.metadata["shuffle"] is False
    for sample in task.dataset:
        assert (sample.metadata or {})["roles"] == ROLES_IN_ORDER


# --- the package imports without data -------------------------------------------


def test_package_imports_without_data(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code = "import sesgo, sesgo._registry; print(sesgo.TASK_VERSION)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "SESGO_DATA_DIR": str(empty)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == TASK_VERSION


def test_task_needs_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("SESGO_DATA_DIR", str(empty))
    with pytest.raises(FileNotFoundError, match="sesgo-data build"):
        sesgo()


# --- oracle ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label_role", "polarity", "wrong", "expected"),
    [
        ("unknown", "neg", False, "unknown"),
        ("unknown", "nonneg", False, "unknown"),
        ("target", "neg", False, "target"),
        ("other", "nonneg", False, "other"),
        # ambiguous and wrong: always the choice that counts in Ft
        ("unknown", "neg", True, "target"),
        ("unknown", "nonneg", True, "other"),
        # disambiguated and wrong: the opposite non-unknown role
        ("target", "neg", True, "other"),
        ("target", "nonneg", True, "other"),
        ("other", "neg", True, "target"),
        ("other", "nonneg", True, "target"),
    ],
)
def test_oracle_role(label_role: Role, polarity: Polarity, wrong: bool, expected: Role) -> None:
    assert oracle_role(label_role, polarity, wrong) == expected


def test_oracle_role_rejects_unknown_polarity() -> None:
    with pytest.raises(ValueError, match="Unknown polarity"):
        oracle_role("unknown", "positive", False)  # pyright: ignore[reportArgumentType]


def test_oracle_role_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="Unknown label_role"):
        oracle_role("victim", "neg", False)  # pyright: ignore[reportArgumentType]


def test_oracle_index_follows_the_presented_order() -> None:
    roles: list[Role] = ["unknown", "other", "target"]
    assert oracle_index(roles, "unknown", "neg", False) == 0
    assert oracle_index(roles, "unknown", "neg", True) == 2
    assert oracle_index(roles, "unknown", "nonneg", True) == 1


def test_oracle_index_rejects_incomplete_roles() -> None:
    with pytest.raises(ValueError, match="one option per role"):
        oracle_index(["target", "target", "unknown"], "unknown", "neg", False)


@pytest.mark.parametrize(
    ("index", "prompt_style", "expected"),
    [
        (0, "clean", "RESPUESTA: A"),
        (1, "clean", "RESPUESTA: B"),
        (2, "clean", "RESPUESTA: C"),
        (0, "paper", "Option 0: uno"),
        (2, "paper", "Option 2: tres"),
    ],
)
def test_oracle_answer(index: int, prompt_style: PromptStyle, expected: str) -> None:
    assert oracle_answer(index, ["uno", "dos", "tres"], prompt_style) == expected


def test_oracle_answer_rejects_unknown_style() -> None:
    with pytest.raises(ValueError, match="Unknown prompt_style"):
        oracle_answer(0, ["uno", "dos", "tres"], "raw")  # pyright: ignore[reportArgumentType]


def test_oracle_answer_rejects_bad_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        oracle_answer(3, ["uno", "dos", "tres"], "clean")


def test_oracle_solver_rejects_unknown_style() -> None:
    with pytest.raises(ValueError, match="Unknown prompt_style"):
        oracle_solver(prompt_style="raw")  # pyright: ignore[reportArgumentType]
