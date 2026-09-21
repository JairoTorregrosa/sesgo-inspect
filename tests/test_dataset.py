"""Unit tests for `sesgo._dataset` (synthetic records, no network)."""

from collections import Counter
from pathlib import Path

import pytest

from sesgo._data import jsonl_path, write_dataset
from sesgo._dataset import (
    limit_per_category_records,
    load_records,
    record_to_sample,
    sesgo_dataset,
    shuffle_permutation,
)
from sesgo._types import CATEGORIES, SPLITS, Category, Option, Polarity, Record, Role, Split

POLARITIES: tuple[Polarity, ...] = ("neg", "nonneg")


def make_record(
    record_id: str = "racismo-es-0001",
    *,
    category: Category = "racismo",
    split: Split = "ambig",
    polarity: Polarity = "neg",
    label_role: Role | None = None,
) -> Record:
    """Build a synthetic record; no upstream text is used."""
    options: list[Option] = [
        {"text": "opcion other", "role": "other"},
        {"text": "opcion target", "role": "target"},
        {"text": "opcion unknown", "role": "unknown"},
    ]
    return {
        "id": record_id,
        "category": category,
        "language": "es",
        "dataset_version": "paper",
        "context_condition": split,
        "question_polarity": polarity,
        "context": "contexto",
        "question": "¿pregunta?",
        "options": options,
        "label_role": label_role or ("unknown" if split == "ambig" else "target"),
        "bbq": False,
        "source": {"commit": "0" * 40, "file": "prompts/test.xlsx", "sheet": "Sheet1", "row": 1},
    }


def synthetic_corpus() -> list[Record]:
    """Twelve records per category, three per split/polarity cell."""
    records: list[Record] = []
    for category in CATEGORIES:
        counter = 0
        for split in SPLITS:
            for polarity in POLARITIES:
                for _ in range(3):
                    counter += 1
                    records.append(
                        make_record(
                            f"{category}-es-{counter:04d}",
                            category=category,
                            split=split,
                            polarity=polarity,
                        )
                    )
    records.sort(key=lambda record: record["id"])
    return records


def write_corpus(tmp_path: Path, records: list[Record] | None = None) -> Path:
    write_dataset(records if records is not None else synthetic_corpus(), "paper", "es", tmp_path)
    return tmp_path


def test_shuffle_permutation_is_a_permutation_and_stable() -> None:
    for record_id in ("racismo-es-0001", "genero-es-0100", "xenofobia-es-1344"):
        permutation = shuffle_permutation(record_id)
        assert sorted(permutation) == [0, 1, 2]
        assert permutation == shuffle_permutation(record_id)


def test_shuffle_permutation_differs_between_ids() -> None:
    permutations = {tuple(shuffle_permutation(f"racismo-es-{n:04d}")) for n in range(1, 200)}
    assert len(permutations) == 6


def test_record_to_sample_without_shuffle_keeps_upstream_order() -> None:
    sample = record_to_sample(make_record(), shuffle=False)
    metadata = sample.metadata or {}
    assert sample.choices == ["opcion other", "opcion target", "opcion unknown"]
    assert metadata["roles"] == ["other", "target", "unknown"]
    assert sample.target == "C"
    assert sample.id == "racismo-es-0001"
    assert sample.input == "contexto\n\n¿pregunta?"
    assert metadata["category_split"] == "racismo/ambig"
    assert metadata["label_role"] == "unknown"
    assert metadata["bbq"] is False
    disambig = record_to_sample(make_record(split="disambig", label_role="target"), shuffle=False)
    assert disambig.target == "B"


def test_record_to_sample_remaps_roles_and_target() -> None:
    for record in synthetic_corpus():
        sample = record_to_sample(record, shuffle=True)
        metadata = sample.metadata or {}
        roles = metadata["roles"]
        choices = sample.choices or []
        permutation = shuffle_permutation(record["id"])
        assert choices == [record["options"][index]["text"] for index in permutation]
        assert roles == [record["options"][index]["role"] for index in permutation]
        assert roles[ord(str(sample.target)) - 65] == record["label_role"]
        assert sorted(choices) == sorted(option["text"] for option in record["options"])


def test_record_to_sample_rejects_broken_roles() -> None:
    record = make_record()
    record["options"][2]["role"] = "target"
    with pytest.raises(ValueError, match="exactly one"):
        record_to_sample(record, shuffle=False)


@pytest.mark.parametrize(
    ("limit", "total", "cells"),
    [
        (4, 16, [1, 1, 1, 1]),
        (6, 24, [1, 1, 2, 2]),  # the remainder spreads over the cells
        (100, 48, [3, 3, 3, 3]),  # a limit above the corpus keeps everything
    ],
)
def test_limit_per_category_is_stratified(limit: int, total: int, cells: list[int]) -> None:
    kept = limit_per_category_records(synthetic_corpus(), limit)
    assert len(kept) == total
    for category in CATEGORIES:
        counted = Counter(
            (record["context_condition"], record["question_polarity"])
            for record in kept
            if record["category"] == category
        )
        assert sorted(counted.values()) == cells


def test_limit_per_category_is_deterministic_and_sorted() -> None:
    records = synthetic_corpus()
    kept = limit_per_category_records(records, 4)
    assert [record["id"] for record in kept] == [
        record["id"] for record in limit_per_category_records(records, 4)
    ]
    assert [record["id"] for record in kept] == sorted(record["id"] for record in kept)
    assert limit_per_category_records(records, None) == records


def test_limit_per_category_rejects_zero() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        limit_per_category_records(synthetic_corpus(), 0)


def test_load_records_roundtrip(tmp_path: Path) -> None:
    records = synthetic_corpus()
    write_corpus(tmp_path, records)
    assert load_records("paper", "es", tmp_path) == records


def test_load_records_without_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run: uv run sesgo-data build"):
        load_records("paper", "es", tmp_path)


def test_load_records_detects_a_changed_byte(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    path = jsonl_path("paper", "es", tmp_path)
    path.write_bytes(path.read_bytes().replace(b"contexto", b"contexta"))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_records("paper", "es", tmp_path)


def test_load_records_rejects_unknown_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown dataset_version"):
        load_records("v9", "es", tmp_path)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="unknown language"):
        load_records("paper", "fr", tmp_path)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="does not exist"):
        load_records("paper", "en", tmp_path)


def test_sesgo_dataset_filters_categories_and_applies_the_limit(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    dataset = sesgo_dataset(["racismo", "genero"], "es", "paper", True, None, tmp_path)
    categories = {(sample.metadata or {})["category"] for sample in dataset}
    assert categories == {"racismo", "genero"}
    assert len(dataset) == 24
    assert dataset.name == "sesgo-paper-es"
    assert len(sesgo_dataset(None, "es", "paper", True, 4, tmp_path)) == 16


def test_sesgo_dataset_rejects_unknown_category(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    with pytest.raises(ValueError, match="unknown categories"):
        sesgo_dataset(
            ["racism"],  # pyright: ignore[reportArgumentType]
            "es",
            "paper",
            True,
            None,
            tmp_path,
        )


def test_sesgo_dataset_is_deterministic(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    first = sesgo_dataset(None, "es", "paper", True, None, tmp_path)
    second = sesgo_dataset(None, "es", "paper", True, None, tmp_path)
    assert [(sample.id, sample.choices, sample.target) for sample in first] == [
        (sample.id, sample.choices, sample.target) for sample in second
    ]


def data_is_built() -> bool:
    return jsonl_path("paper", "es").is_file()


@pytest.mark.dataset_download
@pytest.mark.skipif(not data_is_built(), reason="data/ is not built")
def test_real_paper_dataset_counts() -> None:
    records = load_records("paper", "es")
    counts = Counter(record["category"] for record in records)
    assert dict(counts) == {
        "racismo": 1318,
        "genero": 684,
        "clasismo": 810,
        "xenofobia": 1344,
    }
    assert len(records) == 4156


@pytest.mark.dataset_download
@pytest.mark.skipif(not data_is_built(), reason="data/ is not built")
def test_real_paper_shuffle_balance() -> None:
    positions: Counter[int] = Counter()
    records = load_records("paper", "es")
    for record in records:
        roles = (record_to_sample(record, True).metadata or {})["roles"]
        positions[roles.index("unknown")] += 1
    for index in range(3):
        assert 0.30 <= positions[index] / len(records) <= 0.37
