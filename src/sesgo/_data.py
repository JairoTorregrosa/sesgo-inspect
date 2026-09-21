"""Conversion of the upstream SESGO Excel prompts into clean JSONL data files.

The upstream repository is cloned to `external/SESGO` (git-ignored). Files are read from
git objects with `git show <commit>:<path>`; the work tree is never checked out to an old
commit and upstream code is never executed.

This module also owns the on-disk layout of `data/` (paths, manifest) because both the
converter and the loader in `sesgo._dataset` need it. Heavy dependencies (pandas, openpyxl)
are imported lazily inside the converter, so importing this module stays cheap.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, TypedDict, cast

from sesgo._types import (
    CATEGORIES,
    DATASET_VERSIONS,
    LANGUAGES,
    SPLITS,
    Category,
    DatasetVersion,
    Language,
    Option,
    Polarity,
    Record,
    Role,
    Source,
    Split,
)

UPSTREAM_URL: Final[str] = "https://github.com/mvrobles/SESGO.git"
"""Upstream repository (read-only)."""

COMMITS: Final[dict[DatasetVersion, str]] = {
    "paper": "89b8a0ef69fd86d7f08b99e06f43d16f1c8f599f",
    "head": "28b0a971152747a3b41222be36c446f95b9af9dc",
}
"""Pinned upstream commit per dataset version (decision D2)."""

CONVERTER_VERSION: Final[str] = "1"
"""Bumped when the conversion output changes."""

POLARITIES: Final[tuple[Polarity, ...]] = ("neg", "nonneg")

UNKNOWN_TEXT_ES: Final[str] = "No hay suficiente información"
"""The single Spanish unknown-option string at both pinned commits."""

UNKNOWN_TEXTS_EN: Final[dict[Category, str]] = {
    "racismo": "There is not enough information",
    "genero": "There is not enough information",
    "clasismo": "Cannot be determined",
    "xenofobia": "Not enough information",
}
"""One English unknown-option string per file, derived at the pinned `head` commit."""


class SourceFile(TypedDict):
    """One upstream Excel file."""

    file: str
    sheet: str
    filter_bbq_false: bool


SOURCES: Final[dict[tuple[DatasetVersion, Language], dict[Category, SourceFile]]] = {
    ("paper", "es"): {
        "racismo": {
            "file": "prompts/prompts_racismo.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
        "genero": {
            "file": "prompts/prompts_genero.xlsx",
            "sheet": "prompts_genero_es",
            "filter_bbq_false": False,
        },
        "clasismo": {
            "file": "prompts/prompts_clasismo.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
        "xenofobia": {
            "file": "prompts/prompts_xenofobia.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": True,
        },
    },
    ("head", "es"): {
        "racismo": {
            "file": "prompts/prompts_racismo_es.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
        "genero": {
            "file": "prompts/prompts_genero_es.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
        "clasismo": {
            "file": "prompts/prompts_clasismo_es.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
        "xenofobia": {
            "file": "prompts/prompts_xenofobia_es.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
    },
    ("head", "en"): {
        "racismo": {
            "file": "prompts/prompts_racismo_en.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
        "genero": {
            "file": "prompts/prompts_genero_EN.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
        "clasismo": {
            "file": "prompts/prompts_clasismo_en.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
        "xenofobia": {
            "file": "prompts/prompts_xenofobia_en.xlsx",
            "sheet": "Sheet1",
            "filter_bbq_false": False,
        },
    },
}
"""Upstream file per (dataset version, language, category). `paper`/`en` does not exist."""

EXPECTED_COUNTS: Final[dict[tuple[DatasetVersion, Language], dict[Category, int]]] = {
    ("paper", "es"): {"racismo": 1318, "genero": 684, "clasismo": 810, "xenofobia": 1344},
    ("head", "es"): {"racismo": 1086, "genero": 600, "clasismo": 810, "xenofobia": 1344},
    ("head", "en"): {"racismo": 306, "genero": 738, "clasismo": 312, "xenofobia": 924},
}
"""Rows kept per category, verified at the pinned commits. A mismatch is a hard error."""


class ManifestEntry(TypedDict):
    """Manifest record of one generated JSONL file."""

    dataset_version: DatasetVersion
    language: Language
    commit: str
    file: str
    converter_version: str
    total: int
    categories: dict[str, int]
    splits: dict[str, int]
    category_splits: dict[str, int]
    duplicate_ids: int
    duplicate_prompts: int
    sha256: str


class Manifest(TypedDict):
    """Content of `data/manifest.json`."""

    converter_version: str
    files: dict[str, ManifestEntry]


# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------


def repo_root() -> Path:
    """First parent directory of this file holding `pyproject.toml`, else the cwd."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def default_data_dir() -> Path:
    """Directory of the JSONL files and `manifest.json`: `SESGO_DATA_DIR`, else `data/`."""
    env = os.environ.get("SESGO_DATA_DIR")
    return Path(env).expanduser() if env else repo_root() / "data"


def default_repo_dir() -> Path:
    """Upstream clone directory: `SESGO_EXTERNAL_DIR`, else `external/SESGO`."""
    env = os.environ.get("SESGO_EXTERNAL_DIR")
    return Path(env).expanduser() if env else repo_root() / "external" / "SESGO"


def jsonl_name(dataset_version: DatasetVersion, language: Language) -> str:
    """File name of a data file, without directory."""
    return f"sesgo-{dataset_version}-{language}.jsonl"


def jsonl_path(
    dataset_version: DatasetVersion, language: Language, data_dir: Path | None = None
) -> Path:
    """Full path of a JSONL data file. `data_dir=None` uses `default_data_dir()`."""
    return (data_dir or default_data_dir()) / jsonl_name(dataset_version, language)


def manifest_path(data_dir: Path | None = None) -> Path:
    """Full path of `manifest.json`. `data_dir=None` uses `default_data_dir()`."""
    return (data_dir or default_data_dir()) / "manifest.json"


def sha256_file(path: Path) -> str:
    """Lower-case hexadecimal SHA256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(data_dir: Path | None = None) -> Manifest:
    """Read `manifest.json`.

    Raises:
        FileNotFoundError: If the manifest is absent.
    """
    path = manifest_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no data manifest at {path}; run: uv run sesgo-data build")
    return cast(Manifest, json.loads(path.read_text(encoding="utf-8")))


def manifest_entry(
    dataset_version: DatasetVersion, language: Language, data_dir: Path | None = None
) -> ManifestEntry:
    """Read the manifest entry of one data file.

    Raises:
        FileNotFoundError: If the manifest or the entry is absent.
    """
    manifest = read_manifest(data_dir)
    name = jsonl_name(dataset_version, language)
    entry = manifest["files"].get(name)
    if entry is None:
        raise FileNotFoundError(
            f"{name} is not in the data manifest; "
            f"run: uv run sesgo-data build --version {dataset_version} --language {language}"
        )
    return entry


def validate_version_language(dataset_version: str, language: str) -> None:
    """Check a dataset version and language pair.

    Raises:
        ValueError: If a value is unknown or the pair does not exist upstream.
    """
    if dataset_version not in DATASET_VERSIONS:
        raise ValueError(
            f"unknown dataset_version {dataset_version!r}; valid: {list(DATASET_VERSIONS)}"
        )
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}; valid: {list(LANGUAGES)}")
    if (dataset_version, language) not in SOURCES:
        available = sorted(f"{v}/{lang}" for v, lang in SOURCES)
        raise ValueError(
            f"dataset_version={dataset_version!r} with language={language!r} does not exist "
            f"upstream in complete form; available: {available}"
        )


# --------------------------------------------------------------------------------------
# cell helpers
# --------------------------------------------------------------------------------------


def _unwrap(value: object) -> object:
    """`value.item()` for a numpy scalar, `value` otherwise."""
    if type(value).__module__.startswith("numpy"):
        item = getattr(value, "item", None)
        if callable(item):
            unwrapped: object = item()
            return unwrapped
    return value


def _as_text(row: Mapping[str, object], key: str, where: str) -> str:
    """Read a text cell, stripped.

    Raises:
        ValueError: If the column is missing or the cell is empty.
    """
    if key not in row:
        raise ValueError(f"{where}: missing column {key!r}")
    value = _unwrap(row[key])
    if value is None or (isinstance(value, float) and value != value):
        raise ValueError(f"{where}: empty cell in column {key!r}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{where}: empty cell in column {key!r}")
    return text


def _as_int(row: Mapping[str, object], key: str, where: str) -> int:
    """Read an integer cell.

    Raises:
        ValueError: If the column is missing or the value is not an integer.
    """
    if key not in row:
        raise ValueError(f"{where}: missing column {key!r}")
    value = _unwrap(row[key])
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{where}: column {key!r} is not an integer: {value!r}")
    try:
        number = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"{where}: column {key!r} is not an integer: {value!r}") from error
    return number


def _as_bool(row: Mapping[str, object], key: str, where: str) -> bool:
    """Read a boolean cell.

    Raises:
        ValueError: If the column is missing or the value is not boolean.
    """
    if key not in row:
        raise ValueError(f"{where}: missing column {key!r}")
    value = _unwrap(row[key])
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{where}: column {key!r} is not a boolean: {value!r}")


# --------------------------------------------------------------------------------------
# pure conversion
# --------------------------------------------------------------------------------------


def parse_answer_info(raw: str, where: str) -> list[str]:
    """Parse the upstream `answer_info` cell, a dict literal with keys `ans0`..`ans2`.

    Returns:
        The three option texts in upstream order, stripped.

    Raises:
        ValueError: If the text does not parse or a key is missing.
    """
    try:
        parsed = cast(object, ast.literal_eval(raw))
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"{where}: answer_info does not parse: {raw!r}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{where}: answer_info is not a dict: {raw!r}")
    mapping = cast(dict[object, object], parsed)
    texts: list[str] = []
    for index in range(3):
        key = f"ans{index}"
        if key not in mapping:
            raise ValueError(f"{where}: answer_info has no key {key!r}: {raw!r}")
        text = str(mapping[key]).strip()
        if not text:
            raise ValueError(f"{where}: answer_info key {key!r} is empty: {raw!r}")
        texts.append(text)
    return texts


def roles_for(target_index: int, other_index: int, where: str) -> list[Role]:
    """Derive the role of each option position from the upstream `target`/`other` columns.

    Returns:
        Three roles, aligned with the upstream option order.

    Raises:
        ValueError: If the indices are out of range or equal.
    """
    if not 0 <= target_index <= 2 or not 0 <= other_index <= 2:
        raise ValueError(f"{where}: target/other out of range: {target_index}/{other_index}")
    if target_index == other_index:
        raise ValueError(f"{where}: target and other are the same index: {target_index}")
    roles: list[Role] = ["unknown", "unknown", "unknown"]
    roles[target_index] = "target"
    roles[other_index] = "other"
    return roles


def record_from_row(
    row: Mapping[str, object],
    *,
    category: Category,
    language: Language,
    dataset_version: DatasetVersion,
    commit: str,
    file: str,
    sheet: str,
    row_number: int,
    position: int,
    unknown_text: str,
) -> Record:
    """Convert one upstream row into a `Record`.

    Args:
        row: Row of the sheet, as a mapping of column name to cell value.
        category: Category of the file (the upstream `category` column is ignored).
        language: Language of the file.
        dataset_version: Dataset version.
        commit: Upstream commit the file was read from.
        file: Upstream path of the file.
        sheet: Sheet name.
        row_number: 1-based data row position in the file (before any filter).
        position: 1-based position in the kept rows of the category, used in the id.
        unknown_text: Expected text of the `unknown` option in this file.

    Returns:
        The record.

    Raises:
        ValueError: If a cell is missing or a documented invariant does not hold.
    """
    where = f"{file}[{sheet}] row {row_number}"
    context_condition = _as_text(row, "context_condition", where)
    if context_condition not in SPLITS:
        raise ValueError(f"{where}: unknown context_condition {context_condition!r}")
    question_polarity = _as_text(row, "question_polarity", where)
    if question_polarity not in POLARITIES:
        raise ValueError(f"{where}: unknown question_polarity {question_polarity!r}")
    polarity: Polarity = question_polarity

    texts = parse_answer_info(_as_text(row, "answer_info", where), where)
    target_index = _as_int(row, "target", where)
    other_index = _as_int(row, "other", where)
    roles = roles_for(target_index, other_index, where)

    unknown_index = roles.index("unknown")
    if texts[unknown_index] != unknown_text:
        raise ValueError(
            f"{where}: unexpected unknown-option text {texts[unknown_index]!r}; "
            f"expected {unknown_text!r}"
        )

    label = _as_int(row, "label", where)
    if not 0 <= label <= 2:
        raise ValueError(f"{where}: label out of range: {label}")
    label_role = roles[label]
    split: Split = context_condition
    if split == "ambig" and label_role != "unknown":
        raise ValueError(f"{where}: ambig row with label role {label_role!r}")
    if split == "disambig" and label_role == "unknown":
        raise ValueError(f"{where}: disambig row with label role 'unknown'")

    options: list[Option] = [
        {"text": text, "role": role} for text, role in zip(texts, roles, strict=True)
    ]
    source: Source = {"commit": commit, "file": file, "sheet": sheet, "row": row_number}
    return {
        "id": f"{category}-{language}-{position:04d}",
        "category": category,
        "language": language,
        "dataset_version": dataset_version,
        "context_condition": split,
        "question_polarity": polarity,
        "context": _as_text(row, "context", where),
        "question": _as_text(row, "question", where),
        "options": options,
        "label_role": label_role,
        "bbq": _as_bool(row, "bbq", where),
        "source": source,
    }


# --------------------------------------------------------------------------------------
# upstream access
# --------------------------------------------------------------------------------------


def _git(repo_dir: Path, *args: str) -> bytes:
    """Run a git command in the clone and return its standard output as bytes.

    Raises:
        RuntimeError: If git exits non-zero.
    """
    result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
        ["git", "-C", str(repo_dir), *args], capture_output=True, check=False
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {repo_dir}: {message}")
    return result.stdout


def ensure_clone(repo_dir: Path | None = None) -> Path:
    """Clone the upstream repository if absent, fetch it, and check the pinned commits.

    Returns:
        The clone directory.

    Raises:
        RuntimeError: If the clone fails or a pinned commit is missing.
    """
    target = repo_dir or default_repo_dir()
    if not (target / ".git").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {UPSTREAM_URL} into {target}")
        result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            ["git", "clone", UPSTREAM_URL, str(target)], capture_output=True, check=False
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"git clone of {UPSTREAM_URL} failed: {message}")
    else:
        try:
            _git(target, "fetch", "--all", "--tags", "--quiet")
        except RuntimeError as error:  # offline is fine when the commits are present
            print(f"warning: git fetch failed ({error}); using the local clone")
    missing = [
        commit
        for commit in COMMITS.values()
        if subprocess.run(  # noqa: S603 - fixed argument list, no shell
            ["git", "-C", str(target), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    ]
    if missing:
        raise RuntimeError(f"pinned commits missing in {target}: {missing}")
    return target


def read_sheet(repo_dir: Path, commit: str, file: str, sheet: str) -> list[dict[str, object]]:
    """Read one sheet of an upstream Excel file from a git object.

    Returns:
        The rows, each a mapping of column name to cell value, in file order.

    Raises:
        RuntimeError: If the blob cannot be read.
        ValueError: If the sheet is absent.
    """
    import pandas as pd  # local: pandas is a converter-only dependency

    blob = _git(repo_dir, "show", f"{commit}:{file}")
    with io.BytesIO(blob) as buffer:
        excel = pd.ExcelFile(buffer, engine="openpyxl")
        if sheet not in excel.sheet_names:
            raise ValueError(f"{file}: no sheet {sheet!r}; sheets: {list(excel.sheet_names)}")
        frame = excel.parse(sheet)
    records = cast(list[dict[object, object]], frame.to_dict(orient="records"))
    return [{str(key): value for key, value in record.items()} for record in records]


def convert_category(
    repo_dir: Path, dataset_version: DatasetVersion, language: Language, category: Category
) -> list[Record]:
    """Convert one upstream category file.

    Returns:
        The kept records, in upstream row order.

    Raises:
        ValueError: If the kept row count differs from the verified count.
    """
    spec = SOURCES[(dataset_version, language)][category]
    commit = COMMITS[dataset_version]
    unknown_text = UNKNOWN_TEXT_ES if language == "es" else UNKNOWN_TEXTS_EN[category]
    rows = read_sheet(repo_dir, commit, spec["file"], spec["sheet"])

    records: list[Record] = []
    for row_number, row in enumerate(rows, start=1):
        if spec["filter_bbq_false"] and _as_bool(row, "bbq", f"{spec['file']} row {row_number}"):
            continue
        records.append(
            record_from_row(
                row,
                category=category,
                language=language,
                dataset_version=dataset_version,
                commit=commit,
                file=spec["file"],
                sheet=spec["sheet"],
                row_number=row_number,
                position=len(records) + 1,
                unknown_text=unknown_text,
            )
        )
    expected = EXPECTED_COUNTS[(dataset_version, language)][category]
    if len(records) != expected:
        raise ValueError(
            f"{spec['file']}: kept {len(records)} rows, expected {expected} "
            f"({len(rows)} rows in the file, filter_bbq_false={spec['filter_bbq_false']})"
        )
    return records


def convert(repo_dir: Path, dataset_version: DatasetVersion, language: Language) -> list[Record]:
    """Convert every category of one dataset version and language.

    Returns:
        All records, sorted by id.

    Raises:
        ValueError: If the pair does not exist or a count does not match.
    """
    validate_version_language(dataset_version, language)
    records: list[Record] = []
    for category in CATEGORIES:
        records.extend(convert_category(repo_dir, dataset_version, language, category))
    records.sort(key=lambda record: record["id"])
    return records


def count_duplicates(records: Sequence[Record]) -> tuple[int, int]:
    """Count duplicate ids and duplicate prompts.

    Returns:
        A pair: extra rows that repeat an id, and extra rows that repeat
        `(category, context, question, options)`.
    """
    ids: dict[str, int] = {}
    prompts: dict[tuple[str, str, str, tuple[str, ...]], int] = {}
    for record in records:
        ids[record["id"]] = ids.get(record["id"], 0) + 1
        key = (
            record["category"],
            record["context"],
            record["question"],
            tuple(option["text"] for option in record["options"]),
        )
        prompts[key] = prompts.get(key, 0) + 1
    return (
        sum(count - 1 for count in ids.values() if count > 1),
        sum(count - 1 for count in prompts.values() if count > 1),
    )


def write_dataset(
    records: Sequence[Record],
    dataset_version: DatasetVersion,
    language: Language,
    data_dir: Path | None = None,
) -> ManifestEntry:
    """Write a JSONL file and update `manifest.json`, returning the new manifest entry."""
    directory = data_dir or default_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / jsonl_name(dataset_version, language)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    categories = {category: 0 for category in CATEGORIES}
    splits = {split: 0 for split in SPLITS}
    category_splits: dict[str, int] = {}
    for record in records:
        categories[record["category"]] += 1
        splits[record["context_condition"]] += 1
        key = f"{record['category']}/{record['context_condition']}"
        category_splits[key] = category_splits.get(key, 0) + 1
    duplicate_ids, duplicate_prompts = count_duplicates(records)

    entry: ManifestEntry = {
        "dataset_version": dataset_version,
        "language": language,
        "commit": COMMITS[dataset_version],
        "file": path.name,
        "converter_version": CONVERTER_VERSION,
        "total": len(records),
        "categories": dict(categories),
        "splits": dict(splits),
        "category_splits": dict(sorted(category_splits.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_prompts": duplicate_prompts,
        "sha256": sha256_file(path),
    }

    manifest: Manifest = {"converter_version": CONVERTER_VERSION, "files": {}}
    existing = manifest_path(directory)
    if existing.is_file():
        manifest = cast(Manifest, json.loads(existing.read_text(encoding="utf-8")))
        manifest["converter_version"] = CONVERTER_VERSION
    manifest["files"][path.name] = entry
    manifest["files"] = dict(sorted(manifest["files"].items()))
    existing.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return entry


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------


def _selected_pairs(version_arg: str, language_arg: str) -> list[tuple[DatasetVersion, Language]]:
    """Expand `--version`/`--language` into (dataset version, language) pairs."""
    versions: list[DatasetVersion] = (
        list(DATASET_VERSIONS) if version_arg == "all" else [cast(DatasetVersion, version_arg)]
    )
    languages: list[Language] = (
        list(LANGUAGES) if language_arg == "all" else [cast(Language, language_arg)]
    )
    return [(version, language) for version in versions for language in languages]


def _print_entry(entry: ManifestEntry, path: Path) -> None:
    """Print the counts of one built file."""
    counts = "  ".join(f"{category} {entry['categories'][category]}" for category in CATEGORIES)
    splits = "  ".join(f"{split} {entry['splits'][split]}" for split in SPLITS)
    print(f"built {path} ({entry['total']} records, commit {entry['commit'][:7]})")
    print(f"  {counts}")
    print(
        f"  {splits}  duplicate_ids {entry['duplicate_ids']}  "
        f"duplicate_prompts {entry['duplicate_prompts']}"
    )
    print(f"  sha256 {entry['sha256']}")


def build(
    version_arg: str = "all",
    language_arg: str = "all",
    data_dir: Path | None = None,
    repo_dir: Path | None = None,
) -> int:
    """Clone, convert, and write the data files. Returns a process exit code."""
    requested = _selected_pairs(version_arg, language_arg)
    pairs = [pair for pair in requested if pair in SOURCES]
    skipped = [pair for pair in requested if pair not in SOURCES]
    if not pairs:
        print(f"FAIL no data exists for --version {version_arg} --language {language_arg}")
        return 1
    clone = ensure_clone(repo_dir)
    print(f"upstream clone {clone}")
    for dataset_version, language in skipped:
        print(f"skip {dataset_version}/{language}: not available upstream in complete form")
    for dataset_version, language in pairs:
        records = convert(clone, dataset_version, language)
        entry = write_dataset(records, dataset_version, language, data_dir)
        _print_entry(entry, jsonl_path(dataset_version, language, data_dir))
    print(f"manifest {manifest_path(data_dir)}")
    return 0


def _verify_entry(entry: ManifestEntry, data_dir: Path | None) -> list[str]:
    """Problems found comparing one manifest entry with the file on disk; empty when ok."""
    problems: list[str] = []
    path = (data_dir or default_data_dir()) / entry["file"]
    if not path.is_file():
        return [f"{entry['file']}: missing"]
    actual_hash = sha256_file(path)
    if actual_hash != entry["sha256"]:
        problems.append(f"{entry['file']}: sha256 {actual_hash} != manifest {entry['sha256']}")
    counts = {category: 0 for category in CATEGORIES}
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = cast(Record, json.loads(line))
            counts[record["category"]] += 1
            total += 1
    if total != entry["total"]:
        problems.append(f"{entry['file']}: {total} lines, manifest says {entry['total']}")
    for category in CATEGORIES:
        if counts[category] != entry["categories"].get(category):
            problems.append(
                f"{entry['file']}: {category} {counts[category]}, "
                f"manifest says {entry['categories'].get(category)}"
            )
    expected = EXPECTED_COUNTS.get((entry["dataset_version"], entry["language"]))
    if expected is not None:
        for category, value in expected.items():
            if counts[category] != value:
                problems.append(
                    f"{entry['file']}: {category} {counts[category]}, upstream verified {value}"
                )
    if entry["converter_version"] != CONVERTER_VERSION:
        problems.append(
            f"{entry['file']}: converter version {entry['converter_version']} "
            f"!= {CONVERTER_VERSION}; rebuild"
        )
    return problems


def verify(data_dir: Path | None = None) -> int:
    """Re-check the hashes and counts of every file in the manifest (exit code)."""
    try:
        manifest = read_manifest(data_dir)
    except FileNotFoundError as error:
        print(f"FAIL {error}")
        return 1
    if not manifest["files"]:
        print("FAIL the manifest has no files; run: uv run sesgo-data build")
        return 1
    failures = 0
    for name, entry in manifest["files"].items():
        problems = _verify_entry(entry, data_dir)
        if problems:
            failures += 1
            for problem in problems:
                print(f"FAIL {problem}")
        else:
            print(
                f"PASS {name}: {entry['total']} records, sha256 ok, "
                f"counts ok ({', '.join(f'{k} {v}' for k, v in entry['categories'].items())})"
            )
    return 1 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sesgo-data", description="Build and verify the SESGO data files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="clone upstream and write data files")
    build_parser.add_argument("--version", default="all", choices=[*DATASET_VERSIONS, "all"])
    build_parser.add_argument("--language", default="all", choices=[*LANGUAGES, "all"])
    build_parser.add_argument("--data-dir", type=Path, default=None)
    build_parser.add_argument("--repo-dir", type=Path, default=None)

    verify_parser = subparsers.add_parser("verify", help="re-check hashes and counts")
    verify_parser.add_argument("--data-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the `sesgo-data` console script. Returns a process exit code."""
    args = _parser().parse_args(list(argv) if argv is not None else None)
    command = cast(str, args.command)
    data_dir = cast(Path | None, args.data_dir)
    try:
        if command == "build":
            return build(
                cast(str, args.version),
                cast(str, args.language),
                data_dir,
                cast(Path | None, args.repo_dir),
            )
        return verify(data_dir)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"FAIL {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
