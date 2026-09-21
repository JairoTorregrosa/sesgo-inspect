"""Check V1 of `specs/07-verification.md`: the data (spec 01, acceptance 1-6).

Run: `uv run python scripts/verify_data.py`. Needs `data/` built
(`uv run sesgo-data build --version all --language all`). Costs nothing and calls no model.
Prints one PASS/FAIL line per check and exits non-zero when a check fails.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import cast

from inspect_ai.dataset import Sample

from sesgo._data import (
    EXPECTED_COUNTS,
    SOURCES,
    default_data_dir,
    jsonl_name,
    manifest_entry,
    repo_root,
    verify,
)
from sesgo._dataset import load_records, record_to_sample, sesgo_dataset
from sesgo._types import CATEGORIES, LETTERS, ROLES, DatasetVersion, Language, Record

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        FAILURES.append(name)
    return ok


def sample_key(sample: Sample) -> tuple[object, ...]:
    metadata = sample.metadata or {}
    return (
        sample.id,
        sample.input,
        tuple(sample.choices or []),
        str(sample.target),
        tuple(sorted((key, repr(value)) for key, value in metadata.items())),
    )


def check_counts() -> dict[tuple[DatasetVersion, Language], list[Record]]:
    """Acceptance 1: every built file has the verified per-category counts."""
    loaded: dict[tuple[DatasetVersion, Language], list[Record]] = {}
    for pair in sorted(SOURCES):
        dataset_version, language = pair
        name = jsonl_name(dataset_version, language)
        try:
            records = load_records(dataset_version, language)
        except (FileNotFoundError, ValueError) as error:
            check(f"V1.1 counts {name}", False, str(error))
            continue
        loaded[pair] = records
        counts = Counter(record["category"] for record in records)
        expected = EXPECTED_COUNTS[pair]
        ok = all(counts[category] == expected[category] for category in CATEGORIES)
        detail = "  ".join(f"{category} {counts[category]}" for category in CATEGORIES)
        check(f"V1.1 counts {name}", ok, f"{detail}  total {len(records)}")
    return loaded


def check_verify_and_hash(loaded: dict[tuple[DatasetVersion, Language], list[Record]]) -> None:
    """Acceptance 2: `sesgo-data verify` passes and a changed byte breaks `load_records`."""
    code = verify()
    check("V1.2 sesgo-data verify", code == 0, f"exit code {code}")

    if ("paper", "es") not in loaded:
        check("V1.2 hash failure", False, "data/sesgo-paper-es.jsonl is absent")
        return
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        source = default_data_dir()
        name = jsonl_name("paper", "es")
        shutil.copy(source / "manifest.json", tmp / "manifest.json")
        shutil.copy(source / name, tmp / name)
        # load_records must accept the intact copy
        load_records("paper", "es", tmp)
        content = (tmp / name).read_bytes()
        middle = len(content) // 2
        edited = content[:middle] + (b"X" if content[middle : middle + 1] != b"X" else b"Y")
        edited += content[middle + 1 :]
        (tmp / name).write_bytes(edited)
        try:
            load_records("paper", "es", tmp)
        except ValueError as error:
            check("V1.2 hash failure", "sha256 mismatch" in str(error), str(error)[:120])
        else:
            check("V1.2 hash failure", False, "one changed byte did not raise")

    with tempfile.TemporaryDirectory() as raw:
        try:
            load_records("paper", "es", Path(raw))
        except FileNotFoundError as error:
            check(
                "V1.2 missing data hint",
                "run: uv run sesgo-data build" in str(error),
                str(error)[:120],
            )
        else:
            check("V1.2 missing data hint", False, "an absent data file did not raise")


def check_roles(loaded: dict[tuple[DatasetVersion, Language], list[Record]]) -> None:
    """Acceptance 3: one option per role, the label rule, unique well-formed ids."""
    problems: list[str] = []
    checked = 0
    for (dataset_version, language), records in sorted(loaded.items()):
        seen: set[str] = set()
        pattern = re.compile(rf"^({'|'.join(CATEGORIES)})-{language}-\d{{4}}$")
        for record in records:
            checked += 1
            roles = [option["role"] for option in record["options"]]
            for role in ROLES:
                if roles.count(role) != 1:
                    problems.append(f"{record['id']}: roles {roles}")
            if record["context_condition"] == "ambig" and record["label_role"] != "unknown":
                problems.append(f"{record['id']}: ambig with label_role {record['label_role']}")
            if record["context_condition"] == "disambig" and record["label_role"] == "unknown":
                problems.append(f"{record['id']}: disambig with label_role 'unknown'")
            if not pattern.match(record["id"]) or record["id"] in seen:
                problems.append(f"{record['id']}: bad or duplicate id in {dataset_version}")
            seen.add(record["id"])
    check(
        "V1.3 roles and label rule",
        not problems,
        f"{checked} records checked"
        if not problems
        else f"{len(problems)} problems: {problems[:3]}",
    )


def check_shuffle(loaded: dict[tuple[DatasetVersion, Language], list[Record]]) -> None:
    """Acceptance 4: unknown lands on A/B/C in 30-37 % each; two loads are identical."""
    records = loaded.get(("paper", "es"))
    if records is None:
        check("V1.4 shuffle balance", False, "data/sesgo-paper-es.jsonl is absent")
        return
    positions: Counter[int] = Counter()
    for record in records:
        sample = record_to_sample(record, True)
        roles = cast(list[str], (sample.metadata or {})["roles"])
        positions[roles.index("unknown")] += 1
    total = len(records)
    shares = {LETTERS[index]: positions[index] / total for index in range(3)}
    ok = all(0.30 <= share <= 0.37 for share in shares.values())
    check(
        "V1.4 shuffle balance",
        ok,
        "  ".join(f"{letter} {share:.3f}" for letter, share in shares.items()),
    )

    first = [sample_key(sample) for sample in sesgo_dataset(None, "es", "paper", True, None)]
    second = [sample_key(sample) for sample in sesgo_dataset(None, "es", "paper", True, None)]
    check("V1.4 shuffle determinism", first == second, f"{len(first)} samples equal on two loads")

    unshuffled = sesgo_dataset(None, "es", "paper", False, None)
    roles_first = cast(list[str], (unshuffled[0].metadata or {})["roles"])
    check(
        "V1.4 unshuffled order",
        all(
            cast(list[str], (sample.metadata or {})["roles"]) == ["other", "target", "unknown"]
            for sample in unshuffled
        ),
        f"upstream order kept, first sample {roles_first}",
    )


def check_target_letter(loaded: dict[tuple[DatasetVersion, Language], list[Record]]) -> None:
    """Acceptance 5: the target letter points at the option with `label_role`."""
    problems: list[str] = []
    checked = 0
    for (dataset_version, language), records in sorted(loaded.items()):
        for shuffle in (True, False):
            for record in records:
                sample = record_to_sample(record, shuffle)
                metadata = sample.metadata or {}
                roles = cast(list[str], metadata["roles"])
                letter = str(sample.target)
                checked += 1
                if roles[ord(letter) - 65] != metadata["label_role"]:
                    problems.append(
                        f"{dataset_version}/{language} {record['id']} shuffle={shuffle}"
                    )
                choices = sample.choices or []
                texts = [option["text"] for option in record["options"]]
                if sorted(choices) != sorted(texts):
                    problems.append(f"{record['id']}: choices are not a permutation of the options")
    check(
        "V1.5 target letter",
        not problems,
        f"{checked} samples checked"
        if not problems
        else f"{len(problems)} problems: {problems[:3]}",
    )


def check_git_clean() -> None:
    """Acceptance 6: no file of `external/` or `data/` reaches git."""
    root = repo_root()
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "data", "external"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    dirty = [line for line in status if "data/" in line or "external/" in line]
    check(
        "V1.6 git clean of data",
        not tracked and not dirty,
        "no tracked or untracked data/external paths"
        if not tracked and not dirty
        else f"tracked={tracked[:3]} status={dirty[:3]}",
    )


def check_manifest() -> None:
    """The manifest describes every built file with its commit and duplicate counts."""
    problems: list[str] = []
    for dataset_version, language in sorted(SOURCES):
        try:
            entry = manifest_entry(dataset_version, language)
        except FileNotFoundError as error:
            problems.append(str(error))
            continue
        if entry["duplicate_ids"]:
            problems.append(f"{entry['file']}: {entry['duplicate_ids']} duplicate ids")
    check(
        "V1.7 manifest",
        not problems,
        "entries present, 0 duplicate ids" if not problems else str(problems[:2]),
    )


def main() -> int:
    print(f"data dir {default_data_dir()}")
    loaded = check_counts()
    check_verify_and_hash(loaded)
    check_roles(loaded)
    check_shuffle(loaded)
    check_target_letter(loaded)
    check_manifest()
    check_git_clean()
    print(f"{'FAIL' if FAILURES else 'PASS'} V1 data: {len(FAILURES)} failing checks")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
