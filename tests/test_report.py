"""History records and REPORT.md rendering (spec 06). Synthetic rows, no logs, no network."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from sesgo._cost import LedgerEntry
from sesgo._metrics import ALL_GROUP, Row, bootstrap_ses, headline_keys, metric_key, summarize
from sesgo._report import (
    UNPARSED_FLAG,
    UNSTABLE_SIGN_MARK,
    HistoryRecord,
    _record_from_json,
    build_report,
    latest_records,
    read_history,
    write_history,
)
from sesgo._types import SCORE_KEYS

ZERO: dict[str, float] = dict.fromkeys(SCORE_KEYS, 0.0)


def row(
    split: str = "ambig",
    category: str = "racismo",
    correct: float = 1.0,
    ft: float = 0.0,
    fo: float = 0.0,
    unparsed: float = 0.0,
) -> Row:
    value = dict(ZERO)
    value.update(
        {
            "correct": correct,
            "ft": ft,
            "fo": fo,
            "unparsed": unparsed,
            "parse_strict": 1.0 - unparsed,
            "chose_unknown": correct,
        }
    )
    return Row(category=category, split=split, value=value)  # type: ignore[arg-type]


def synthetic_rows(bias_to_target: int = 4) -> list[Row]:
    rows: list[Row] = []
    for split in ("ambig", "disambig"):
        for category in ("racismo", "genero", "clasismo", "xenofobia"):
            rows.extend(row(split, category) for _ in range(6))
            rows.extend(row(split, category, correct=0.0, ft=1.0) for _ in range(bias_to_target))
            rows.extend(row(split, category, correct=0.0, fo=1.0) for _ in range(2))
    return rows


def record(
    config: str = "modelo-a",
    style: str = "clean",
    group: str = "modern",
    reasoning: str = "off",
    rows: list[Row] | None = None,
    reasoning_tokens: int = 0,
    created: str = "2026-09-21T00:00:00",
) -> HistoryRecord:
    scored = rows if rows is not None else synthetic_rows()
    metrics = summarize(scored)
    ses = bootstrap_ses(scored, headline_keys(scored), n_boot=50)
    return HistoryRecord(
        key=f"run-{config}-{style}/task-{config}-{style}",
        run_id=f"run-{config}-{style}",
        task_id=f"task-{config}-{style}",
        created=created,
        recorded_at="2026-09-21T00:00:01",
        config=config,
        model=f"openrouter/x/{config}",
        group=group,
        reasoning=reasoning,
        prompt_style=style,
        stage="full",
        task_version="1-A",
        dataset_version="paper",
        dataset_commit="89b8a0e",
        data_sha256="deadbeef",
        parser_version="2",
        refusal_patterns_version="1",
        language="es",
        tie_sign=1,
        temperature=0.75,
        max_tokens=512,
        samples=len(scored),
        input_tokens=1000,
        output_tokens=200,
        reasoning_tokens=reasoning_tokens,
        total_tokens=1200,
        cost_tokens_usd=0.01,
        cost_credits_usd=0.011,
        log_file=f"{config}.eval",
        metrics=metrics,
        ses=ses,
    )


# --------------------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------------------


def test_history_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    records = [record("a"), record("b")]
    assert write_history(records, path) == (2, 0)
    assert write_history(records, path) == (0, 2)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_history_adds_only_the_new_records(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    write_history([record("a")], path)
    assert write_history([record("a"), record("b")], path) == (1, 1)
    assert {item["config"] for item in read_history(path)} == {"a", "b"}


def test_history_lines_are_valid_json_without_nan(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    empty = record("empty", rows=[])
    empty.metrics[metric_key("ambig", ALL_GROUP, "accuracy")] = float("nan")
    write_history([empty], path)
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["metrics"][metric_key("ambig", ALL_GROUP, "accuracy")] is None
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_history_holds_no_prompt_or_response(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    write_history([record("a")], path)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8").strip())
    assert not set(payload) & {"input", "messages", "completion", "output", "samples_detail"}


def test_record_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    original = record("a")
    write_history([original], path)
    again = _record_from_json(read_history(path)[0])
    assert again.config == original.config
    assert again.tie_sign == original.tie_sign
    assert again.value("ambig", ALL_GROUP, "accuracy") == pytest.approx(
        original.value("ambig", ALL_GROUP, "accuracy"), abs=1e-6
    )


def test_record_from_an_older_line_takes_neutral_defaults() -> None:
    older = _record_from_json({"config": "old", "metrics": {}})
    assert older.group == "unknown"
    assert older.prompt_style == "clean"
    assert older.samples == 0
    assert math.isnan(older.value("ambig", ALL_GROUP, "accuracy"))


def test_latest_records_keeps_the_newest_per_config_style_and_version() -> None:
    old = record("a", created="2026-01-01T00:00:00")
    new = record("a", created="2026-09-21T00:00:00")
    other = record("b")
    kept = latest_records([old.to_json(), new.to_json(), other.to_json()])
    assert len(kept) == 2
    assert {item.created for item in kept if item.config == "a"} == {"2026-09-21T00:00:00"}


def test_latest_records_keeps_both_prompt_styles() -> None:
    kept = latest_records([record("a", "clean").to_json(), record("a", "paper").to_json()])
    assert {item.prompt_style for item in kept} == {"clean", "paper"}


# --------------------------------------------------------------------------------------
# REPORT.md
#
# The exact Markdown is not a contract: `scripts/verify_report.py` recomputes the
# published report from the real logs. These tests only check that the report renders
# and that the numbers it puts in the tables are the ones the metrics computed.
# --------------------------------------------------------------------------------------


def test_report_renders_the_numbers_of_the_run() -> None:
    text = build_report([record()], [], Path("logs/full"))
    assert "modelo-a" in text
    assert "0.500" in text  # accuracy: 6 correct out of 12 rows per cell
    assert "0.527" in text  # bias_score of the same rows


@pytest.mark.parametrize(
    "records",
    [[], [record("vacio", rows=[])]],
    ids=["no-runs", "run-without-samples"],
)
def test_report_renders_when_there_is_nothing_to_aggregate(
    records: list[HistoryRecord],
) -> None:
    text = build_report(records, [], Path("logs/full"))
    assert text.strip()
    assert "±" not in text  # no aggregate was rendered, and nothing raised


# A partial run must still produce a readable report (module docstring of `_report`), and
# the sections that cannot complete a comparison must say which config is missing its half
# rather than render an empty or half-filled table.


def test_report_skips_the_reasoning_pair_table_when_only_one_variant_ran() -> None:
    text = build_report([record("solo", reasoning="off")], [], Path("logs/full"))
    assert "solo (off)" in text
    assert "Δ bias" not in text  # no pair, so no comparison table at all


def test_report_names_the_anchor_whose_other_prompt_style_is_missing() -> None:
    text = build_report([record("gpt-4o-mini", "clean", group="anchor")], [], Path("logs/full"))
    assert "Faltan estilos" in text
    assert "gpt-4o-mini" in text


def test_report_computes_the_reasoning_off_against_low_deltas() -> None:
    records = [
        record("m", reasoning="off"),
        record("m-low", reasoning="low", reasoning_tokens=600, rows=synthetic_rows(2)),
    ]
    text = build_report(records, [], Path("logs/full"))
    # accuracy 0.500 -> 0.600, bias 0.527 -> 0.400, 600 reasoning tokens over 80 samples.
    for number in ("0.500", "0.600", "0.100", "0.527", "0.400", "-0.127", "7.5"):
        assert number in text


def test_report_compares_the_anchor_against_the_published_numbers() -> None:
    records = [
        record("gpt-4o-mini", "clean", group="anchor"),
        record("gpt-4o-mini", "paper", group="anchor"),
    ]
    text = build_report(records, [], Path("logs/full"))
    assert "0.926" in text  # Table A3, disambiguated accuracy
    assert "-0.426" in text  # port fidelity: our paper style minus the published value


def test_report_flags_a_high_unparsed_rate() -> None:
    rows = synthetic_rows()
    rows += [row("ambig", "racismo", correct=0.0, unparsed=1.0) for _ in range(10)]
    text = build_report([record("ruidoso", rows=rows)], [], Path("logs/full"))
    assert "0.172" in text  # 10 unparsed out of 58 ambiguous samples
    assert f"unparsed > {UNPARSED_FLAG:.0%}" in text


def test_report_shows_the_real_cost_from_the_ledger() -> None:
    ledger: list[LedgerEntry] = [
        {
            "unit": "full/a__clean",
            "samples": 100,
            "input_tokens": 10,
            "output_tokens": 20,
            "reasoning_tokens": 0,
            "cost_tokens_usd": 1.0,
            "cost_credits_usd": 1.25,
        }
    ]
    text = build_report([record()], ledger, Path("logs/full"))
    assert "full/a__clean" in text
    assert "1.2500" in text  # the charged spend is the credit delta, not the token cost


def test_report_carries_panel_notes() -> None:
    text = build_report([record()], [], Path("logs/full"), ["`m`: no puede apagar razonamiento"])
    assert "no puede apagar razonamiento" in text


def test_report_never_contains_a_prompt_or_a_response() -> None:
    text = build_report([record()], [], Path("logs/full"))
    for forbidden in ("RESPUESTA: A", "Lee el contexto", "##context", "Option 0:"):
        assert forbidden not in text


# --------------------------------------------------------------------------------------
# sign-stability caveat (V10)
# --------------------------------------------------------------------------------------


def tied_rows() -> list[Row]:
    """Rows whose Ft equals Fo, so the bias score's sign is the `tie_sign` convention."""
    rows: list[Row] = []
    for split in ("ambig", "disambig"):
        for category in ("racismo", "genero", "clasismo", "xenofobia"):
            rows.extend(row(split, category) for _ in range(6))
            rows.extend(row(split, category, correct=0.0, ft=1.0) for _ in range(2))
            rows.extend(row(split, category, correct=0.0, fo=1.0) for _ in range(2))
    return rows


def decided_rows() -> list[Row]:
    """Rows where Ft is far above Fo in every cell, so the sign is not in doubt."""
    rows: list[Row] = []
    for split in ("ambig", "disambig"):
        for category in ("racismo", "genero", "clasismo", "xenofobia"):
            rows.extend(row(split, category) for _ in range(68))
            rows.extend(row(split, category, correct=0.0, ft=1.0) for _ in range(30))
            rows.extend(row(split, category, correct=0.0, fo=1.0) for _ in range(2))
    return rows


@pytest.mark.parametrize(
    ("rows", "bias", "marked"),
    [(tied_rows(), "0.400", True), (decided_rows(), "0.425", False)],
    ids=["undecided-sign", "decided-sign"],
)
def test_report_marks_only_a_bias_score_whose_sign_the_data_do_not_decide(
    rows: list[Row], bias: str, marked: bool
) -> None:
    text = build_report([record(rows=rows)], [], Path("logs/full"))
    section_one = text.split("## 2.")[0]  # section 6 uses `**` for bold
    assert bias in section_one
    assert (UNSTABLE_SIGN_MARK in section_one) is marked
