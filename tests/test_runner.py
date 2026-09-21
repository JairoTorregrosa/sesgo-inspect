"""Runner orchestration that can be checked offline (spec 06, V9).

The runner itself is verified by real runs (D17); what lives here is the part that
decides *money*, because a mistake in it is silent: the reconciliation of an attempt
that was killed before it could write its ledger line.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sesgo._cost import (
    CostError,
    Credits,
    Inflight,
    LedgerEntry,
    ledger_path,
    ledger_spent_usd,
    read_inflight,
    read_ledger,
    unit_is_done,
    write_inflight,
)
from sesgo._runner import _settled_credits, prepaid_usd, reconcile_inflight


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `results/` at a temporary directory, never the committed ledger."""
    monkeypatch.setenv("SESGO_RESULTS_DIR", str(tmp_path))
    return tmp_path


MARKER: Inflight = {
    "timestamp": "2026-09-21T10:00:00+00:00",
    "unit": "full/m__clean",
    "stage": "full",
    "config": "m",
    "model": "openrouter/x/m",
    "prompt_style": "clean",
    "credits_before": 20.0,
}


def test_no_marker_means_nothing_to_reconcile(results: Path) -> None:
    del results
    assert reconcile_inflight() is None
    assert read_ledger() == []


def test_a_killed_attempt_is_booked_from_the_credits_delta(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del results
    write_inflight(MARKER)
    monkeypatch.setattr("sesgo._runner.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "sesgo._runner.fetch_credits",
        lambda: Credits(total_credits=25.0, total_usage=20.042, read_at=time.time()),
    )
    entry = reconcile_inflight()
    assert entry is not None
    assert entry.get("status") == "interrupted"
    assert float(entry.get("cost_credits_usd") or 0.0) == pytest.approx(0.042)

    ledger = read_ledger()
    assert len(ledger) == 1
    # The money reaches the budget gate, the unit is still not done, and the marker is
    # gone, so a third invocation does not book the same spend twice.
    assert ledger_spent_usd(ledger) == pytest.approx(0.042)
    assert unit_is_done(ledger, "full/m__clean") is False
    assert read_inflight() is None
    assert reconcile_inflight() is None
    assert len(read_ledger()) == 1


def test_settled_credits_waits_until_total_usage_stops_growing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([20.0, 20.03, 20.05, 20.05, 20.05])
    monkeypatch.setattr("sesgo._runner.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "sesgo._runner.fetch_credits",
        lambda: Credits(total_credits=25.0, total_usage=next(readings), read_at=time.time()),
    )
    assert _settled_credits().total_usage == pytest.approx(20.05)


def test_prepaid_only_counts_interrupted_lines_of_the_same_unit() -> None:
    ledger: list[LedgerEntry] = [
        {"unit": "full/m__clean", "status": "interrupted", "cost_credits_usd": 0.03},
        {"unit": "full/m__clean", "status": "interrupted", "cost_credits_usd": 0.01},
        {"unit": "full/m__clean", "status": "failed", "cost_tokens_usd": 0.5},
        {"unit": "full/other__clean", "status": "interrupted", "cost_credits_usd": 0.9},
    ]
    assert prepaid_usd(ledger, "full/m__clean") == pytest.approx(0.04)
    assert prepaid_usd(ledger, "full/never__clean") == 0.0
    assert prepaid_usd([], "full/m__clean") == 0.0


def test_the_marker_survives_an_unreachable_credits_endpoint(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del results
    write_inflight(MARKER)

    def boom() -> Credits:
        raise CostError("no network")

    monkeypatch.setattr("sesgo._runner.fetch_credits", boom)
    assert reconcile_inflight() is None
    # Nothing booked, nothing lost: the next invocation tries again.
    assert not ledger_path().exists()
    assert read_inflight() == MARKER
