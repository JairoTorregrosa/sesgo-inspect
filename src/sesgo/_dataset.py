"""Loader: clean JSONL records to Inspect samples.

The loader never clones or converts. It reads `data/sesgo-{version}-{language}.jsonl`,
checks the SHA256 against `data/manifest.json`, and builds a `MemoryDataset`. Option order
is shuffled per sample with a seed derived from the sample id (decision D4), and the roles
and the target letter follow the same permutation.
"""

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from inspect_ai.dataset import MemoryDataset, Sample

from sesgo._data import (
    POLARITIES,
    jsonl_path,
    manifest_entry,
    sha256_file,
    validate_version_language,
)
from sesgo._types import (
    CATEGORIES,
    LETTERS,
    ROLES,
    SPLITS,
    Category,
    DatasetVersion,
    Language,
    Record,
    Role,
    SampleMeta,
)

BUILD_HINT: str = "run: uv run sesgo-data build"
"""Text included in the error raised when a data file is absent."""


def shuffle_permutation(record_id: str) -> list[int]:
    """Return the option permutation of a record (decision D4).

    The record id is the only source of the seed, so the permutation is stable.

    Returns:
        A permutation of `[0, 1, 2]`. Position `k` of the presentation order holds the
        upstream option with index `permutation[k]`.
    """
    digest = hashlib.sha256(record_id.encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed).sample(range(3), 3)


def load_records(
    dataset_version: DatasetVersion = "paper",
    language: Language = "es",
    data_dir: Path | None = None,
) -> list[Record]:
    """Read the records of one data file and check its SHA256.

    Args:
        dataset_version: Dataset version.
        language: Language.
        data_dir: Data directory; `None` uses `SESGO_DATA_DIR` or `<repo root>/data`.

    Returns:
        The records, in file order (sorted by id).

    Raises:
        ValueError: If the version/language pair is unknown or does not exist upstream,
            or if the file does not match the manifest.
        FileNotFoundError: If the data file or the manifest is absent.
    """
    validate_version_language(dataset_version, language)
    path = jsonl_path(dataset_version, language, data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no data file at {path}; {BUILD_HINT}")
    entry = manifest_entry(dataset_version, language, data_dir)

    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ValueError(
            f"sha256 mismatch for {path}: file {actual}, manifest {entry['sha256']}; "
            f"the data file changed after it was built; {BUILD_HINT}"
        )
    with path.open("r", encoding="utf-8") as handle:
        records = [cast(Record, json.loads(line)) for line in handle if line.strip()]
    if len(records) != entry["total"]:
        raise ValueError(
            f"{path}: {len(records)} records, manifest says {entry['total']}; {BUILD_HINT}"
        )
    return records


def record_to_sample(record: Record, shuffle: bool) -> Sample:
    """Convert a record into an Inspect sample.

    Args:
        record: One record of the data file.
        shuffle: Shuffle the option order with the seed derived from the record id.
            `False` keeps the upstream order (other, target, unknown).

    Returns:
        The sample. `choices`, `metadata["roles"]`, and `target` share one permutation.

    Raises:
        ValueError: If the record does not hold exactly one option per role.
    """
    options = record["options"]
    roles: list[Role] = [option["role"] for option in options]
    for role in ROLES:
        if roles.count(role) != 1:
            raise ValueError(f"{record['id']}: expected exactly one {role!r} option, got {roles}")

    permutation = shuffle_permutation(record["id"]) if shuffle else [0, 1, 2]
    choices: list[str] = [options[index]["text"] for index in permutation]
    shown_roles: list[Role] = [roles[index] for index in permutation]
    correct_index = roles.index(record["label_role"])
    target_letter = LETTERS[permutation.index(correct_index)]

    metadata: SampleMeta = {
        "category": record["category"],
        "language": record["language"],
        "dataset_version": record["dataset_version"],
        "context_condition": record["context_condition"],
        "question_polarity": record["question_polarity"],
        "context": record["context"],
        "question": record["question"],
        "roles": shown_roles,
        "label_role": record["label_role"],
        "bbq": record["bbq"],
        "category_split": f"{record['category']}/{record['context_condition']}",
    }
    return Sample(
        input=f"{record['context']}\n\n{record['question']}",
        choices=choices,
        target=target_letter,
        id=record["id"],
        metadata=dict(metadata),
    )


def limit_per_category_records(
    records: Sequence[Record], limit_per_category: int | None
) -> list[Record]:
    """Take a deterministic stratified subset of each category.

    Inside a category the records are grouped in the four cells `split x polarity`,
    each cell ordered by `sha256(id)`, and taken in a round robin over the cells until
    `limit_per_category` records are kept.

    Args:
        records: Records to subset.
        limit_per_category: Records to keep per category; `None` keeps all.

    Returns:
        The kept records, sorted by id.

    Raises:
        ValueError: If `limit_per_category` is smaller than 1.
    """
    if limit_per_category is None:
        return list(records)
    if limit_per_category < 1:
        raise ValueError(f"limit_per_category must be >= 1, got {limit_per_category}")

    kept: list[Record] = []
    for category in CATEGORIES:
        cells: list[list[Record]] = []
        for split in SPLITS:
            for polarity in POLARITIES:
                cell = [
                    record
                    for record in records
                    if record["category"] == category
                    and record["context_condition"] == split
                    and record["question_polarity"] == polarity
                ]
                cell.sort(key=lambda record: hashlib.sha256(record["id"].encode()).hexdigest())
                cells.append(cell)
        taken = 0
        position = 0
        while taken < limit_per_category and any(position < len(cell) for cell in cells):
            for cell in cells:
                if taken >= limit_per_category:
                    break
                if position < len(cell):
                    kept.append(cell[position])
                    taken += 1
            position += 1
    kept.sort(key=lambda record: record["id"])
    return kept


def sesgo_dataset(
    categories: Sequence[Category] | None,
    language: Language,
    dataset_version: DatasetVersion,
    shuffle: bool,
    limit_per_category: int | None,
    data_dir: Path | None = None,
) -> MemoryDataset:
    """Build the Inspect dataset of the SESGO task.

    Args:
        categories: Categories to keep; `None` keeps all four.
        language: Language.
        dataset_version: Dataset version.
        shuffle: Shuffle the option order per sample (decision D4).
        limit_per_category: Deterministic stratified subset size per category, or `None`.
        data_dir: Data directory; `None` uses `SESGO_DATA_DIR` or `<repo root>/data`.

    Returns:
        A `MemoryDataset` of samples, ordered by record id.

    Raises:
        ValueError: If a category, the language, or the version is unknown, or if
            `limit_per_category` is smaller than 1.
        FileNotFoundError: If the data file is absent.
    """
    if categories is not None:
        unknown = [category for category in categories if category not in CATEGORIES]
        if unknown:
            raise ValueError(f"unknown categories {unknown}; valid: {list(CATEGORIES)}")
    selected = tuple(categories) if categories is not None else CATEGORIES

    records = [
        record
        for record in load_records(dataset_version, language, data_dir)
        if record["category"] in selected
    ]
    records = limit_per_category_records(records, limit_per_category)
    samples = [record_to_sample(record, shuffle) for record in records]
    return MemoryDataset(
        samples=samples,
        name=f"sesgo-{dataset_version}-{language}",
        location=str(jsonl_path(dataset_version, language, data_dir)),
        shuffled=False,
    )
