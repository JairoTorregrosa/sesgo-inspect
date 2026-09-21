"""Unit tests for `sesgo._data` (pure logic, synthetic rows, no network)."""

import json
from pathlib import Path
from typing import Any

import pytest

from sesgo._data import (
    CONVERTER_VERSION,
    EXPECTED_COUNTS,
    SOURCES,
    UNKNOWN_TEXT_ES,
    UNKNOWN_TEXTS_EN,
    _selected_pairs,
    count_duplicates,
    default_data_dir,
    jsonl_name,
    jsonl_path,
    main,
    manifest_entry,
    parse_answer_info,
    read_manifest,
    record_from_row,
    roles_for,
    validate_version_language,
    verify,
    write_dataset,
)
from sesgo._types import CATEGORIES, Record

# Synthetic row in the upstream shape. No upstream text is copied into the repository.
ROW: dict[str, Any] = {
    "question_polarity": " neg ",
    "context_condition": "ambig",
    "category": "wrong-on-purpose",
    "answer_info": f"{{'ans0': ' grupo A ', 'ans1': 'grupo B', 'ans2': {UNKNOWN_TEXT_ES!r}}}",
    "context": "  contexto de prueba  ",
    "question": "¿pregunta de prueba?",
    "target": 1,
    "other": 0,
    "label": 2,
    "bbq": False,
}


def row(**changes: Any) -> dict[str, Any]:
    merged = dict(ROW)
    merged.update(changes)
    return merged


def make_record(position: int = 3, **changes: Any) -> Record:
    return record_from_row(
        row(**changes),
        category="racismo",
        language="es",
        dataset_version="paper",
        commit="0" * 40,
        file="prompts/test.xlsx",
        sheet="Sheet1",
        row_number=7,
        position=position,
        unknown_text=UNKNOWN_TEXT_ES,
    )


def test_roles_for_maps_indices() -> None:
    assert roles_for(1, 0, "x") == ["other", "target", "unknown"]
    assert roles_for(0, 2, "x") == ["target", "unknown", "other"]


@pytest.mark.parametrize(("target", "other"), [(1, 1), (3, 0), (0, -1)])
def test_roles_for_rejects_bad_indices(target: int, other: int) -> None:
    with pytest.raises(ValueError):
        roles_for(target, other, "x")


def test_parse_answer_info_strips_text() -> None:
    assert parse_answer_info("{'ans0': ' a ', 'ans1': 'b', 'ans2': 'c'}", "x") == ["a", "b", "c"]


@pytest.mark.parametrize(
    "raw",
    ["not a dict", "{'ans0': 'a', 'ans1': 'b'}", "{'ans0': 'a', 'ans1': 'b', 'ans2': '  '}", "[1]"],
)
def test_parse_answer_info_rejects_bad_text(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_answer_info(raw, "prompts/test.xlsx row 1")


def test_record_from_row_builds_the_record() -> None:
    record = make_record()
    assert record["id"] == "racismo-es-0003"
    assert record["category"] == "racismo"
    assert record["question_polarity"] == "neg"
    assert record["context"] == "contexto de prueba"
    assert [option["role"] for option in record["options"]] == ["other", "target", "unknown"]
    assert [option["text"] for option in record["options"]][0] == "grupo A"
    assert record["label_role"] == "unknown"
    assert record["bbq"] is False
    assert record["source"] == {
        "commit": "0" * 40,
        "file": "prompts/test.xlsx",
        "sheet": "Sheet1",
        "row": 7,
    }


def test_record_from_row_accepts_disambig_with_target_label() -> None:
    record = make_record(context_condition="disambig", label=1)
    assert record["label_role"] == "target"


@pytest.mark.parametrize(
    ("context_condition", "label", "match"),
    [
        ("ambig", 1, "ambig row with label role"),
        ("disambig", 2, "disambig row with label role"),
    ],
)
def test_record_from_row_rejects_label_role_against_split(
    context_condition: str, label: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        make_record(context_condition=context_condition, label=label)


def test_record_from_row_rejects_unexpected_unknown_text() -> None:
    changed = "{'ans0': 'a', 'ans1': 'b', 'ans2': 'Otra cosa'}"
    with pytest.raises(ValueError, match="unexpected unknown-option text"):
        make_record(answer_info=changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_condition", "other"),
        ("question_polarity", "positive"),
        ("label", 5),
        ("bbq", "maybe"),
        ("context", ""),
    ],
)
def test_record_from_row_rejects_bad_cells(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        make_record(**{field: value})


def test_record_from_row_reports_the_row_location() -> None:
    with pytest.raises(ValueError, match=r"prompts/test.xlsx\[Sheet1\] row 7"):
        make_record(answer_info="{'ans0': 'a'}")


def test_record_from_row_accepts_english_unknown_text() -> None:
    record = record_from_row(
        row(answer_info=f"{{'ans0': 'a', 'ans1': 'b', 'ans2': {UNKNOWN_TEXTS_EN['clasismo']!r}}}"),
        category="clasismo",
        language="en",
        dataset_version="head",
        commit="1" * 40,
        file="prompts/prompts_clasismo_en.xlsx",
        sheet="Sheet1",
        row_number=1,
        position=1,
        unknown_text=UNKNOWN_TEXTS_EN["clasismo"],
    )
    assert record["id"] == "clasismo-en-0001"
    assert record["language"] == "en"


def test_count_duplicates() -> None:
    first = make_record()
    second = make_record()
    assert count_duplicates([first]) == (0, 0)
    assert count_duplicates([first, second]) == (1, 1)


def test_validate_version_language() -> None:
    validate_version_language("paper", "es")
    with pytest.raises(ValueError, match="unknown dataset_version"):
        validate_version_language("v2", "es")
    with pytest.raises(ValueError, match="unknown language"):
        validate_version_language("paper", "fr")
    with pytest.raises(ValueError, match="does not exist"):
        validate_version_language("paper", "en")


def test_sources_and_expected_counts_agree() -> None:
    assert set(SOURCES) == set(EXPECTED_COUNTS)
    for pair, files in SOURCES.items():
        assert set(files) == set(CATEGORIES)
        assert set(EXPECTED_COUNTS[pair]) == set(CATEGORIES)
    assert EXPECTED_COUNTS[("paper", "es")] == {
        "racismo": 1318,
        "genero": 684,
        "clasismo": 810,
        "xenofobia": 1344,
    }
    assert sum(EXPECTED_COUNTS[("paper", "es")].values()) == 4156


def test_default_data_dir_uses_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SESGO_DATA_DIR", "/tmp/sesgo-test-data")
    assert default_data_dir() == Path("/tmp/sesgo-test-data")
    monkeypatch.delenv("SESGO_DATA_DIR")
    assert default_data_dir().name == "data"


def test_write_dataset_and_verify(tmp_path: Path) -> None:
    records = [make_record(), make_record(context_condition="disambig", label=0)]
    records[1] = {**records[1], "id": "racismo-es-0004"}
    records.sort(key=lambda record: record["id"])
    entry = write_dataset(records, "paper", "es", tmp_path)

    path = jsonl_path("paper", "es", tmp_path)
    assert path.name == jsonl_name("paper", "es")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "racismo-es-0003"
    assert "¿pregunta de prueba?" in lines[0]  # ensure_ascii=False

    assert entry["total"] == 2
    assert entry["categories"]["racismo"] == 2
    assert entry["splits"] == {"ambig": 1, "disambig": 1}
    assert entry["category_splits"] == {"racismo/ambig": 1, "racismo/disambig": 1}
    assert entry["converter_version"] == CONVERTER_VERSION
    assert entry["commit"].startswith("89b8a0")

    manifest = read_manifest(tmp_path)
    assert list(manifest["files"]) == [jsonl_name("paper", "es")]
    assert manifest_entry("paper", "es", tmp_path)["sha256"] == entry["sha256"]


def test_write_dataset_keeps_other_manifest_entries(tmp_path: Path) -> None:
    write_dataset([make_record()], "paper", "es", tmp_path)
    write_dataset([make_record()], "head", "es", tmp_path)
    assert set(read_manifest(tmp_path)["files"]) == {
        jsonl_name("paper", "es"),
        jsonl_name("head", "es"),
    }


@pytest.fixture
def synthetic_expected_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the upstream count check of `verify` accept a one-record synthetic file."""
    monkeypatch.setitem(
        EXPECTED_COUNTS,
        ("paper", "es"),
        {"racismo": 1, "genero": 0, "clasismo": 0, "xenofobia": 0},
    )


@pytest.mark.usefixtures("synthetic_expected_counts")
def test_verify_detects_a_changed_byte(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_dataset([make_record()], "paper", "es", tmp_path)
    assert verify(tmp_path) == 0
    path = jsonl_path("paper", "es", tmp_path)
    content = path.read_bytes()
    path.write_bytes(content.replace(b"contexto", b"contexta"))
    assert verify(tmp_path) == 1
    assert "sha256" in capsys.readouterr().out


def test_verify_without_a_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert verify(tmp_path) == 1
    assert "run: uv run sesgo-data build" in capsys.readouterr().out


@pytest.mark.usefixtures("synthetic_expected_counts")
def test_main_verify_command(tmp_path: Path) -> None:
    write_dataset([make_record()], "paper", "es", tmp_path)
    assert main(["verify", "--data-dir", str(tmp_path)]) == 0


def test_main_build_without_data_pair(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["build", "--version", "paper", "--language", "en", "--data-dir", str(tmp_path)])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("version_arg", "language_arg", "expected"),
    [
        # `--version all --language all` is the default of `sesgo-data build`, so the four
        # candidate pairs must be offered; `build` then drops paper/en, which does not exist.
        ("all", "all", [("paper", "es"), ("paper", "en"), ("head", "es"), ("head", "en")]),
        ("all", "es", [("paper", "es"), ("head", "es")]),
        ("head", "all", [("head", "es"), ("head", "en")]),
        ("paper", "es", [("paper", "es")]),
    ],
)
def test_selected_pairs_expands_the_build_arguments(
    version_arg: str, language_arg: str, expected: list[tuple[str, str]]
) -> None:
    assert _selected_pairs(version_arg, language_arg) == expected
    assert [pair for pair in _selected_pairs(version_arg, language_arg) if pair in SOURCES] == [
        pair for pair in expected if pair in SOURCES
    ]
