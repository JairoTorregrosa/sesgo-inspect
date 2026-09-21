"""Cost arithmetic, ledger and budget (spec 06). Pure logic: no network, no API key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sesgo._cost import (
    BUDGET_CAP_USD,
    DEFAULT_OUTPUT_TOKENS,
    CostError,
    Credits,
    Inflight,
    LedgerEntry,
    MissingPriceError,
    Price,
    PriceTable,
    _parse_models_payload,
    _read_prices,
    _write_prices,
    append_ledger,
    clear_inflight,
    count_tokens,
    entry_spend_usd,
    estimate_unit,
    interrupted_entry,
    ledger_spent_usd,
    measured_output_tokens,
    poll_credits,
    price_for,
    read_inflight,
    read_ledger,
    resolve_budget,
    token_cost_usd,
    unit_is_done,
    write_inflight,
)
from sesgo._panel import PanelConfig, Unit

FAKE = PriceTable(
    fetched_at="2026-09-21T00:00:00+00:00",
    prices={
        "x/cheap": Price("x/cheap", input_usd=1e-7, output_usd=2e-7),
        "x/dear": Price("x/dear", input_usd=1e-6, output_usd=1e-5),
    },
)


def config(
    name: str = "cheap", reasoning: str = "off", model: str = "openrouter/x/cheap"
) -> PanelConfig:
    return PanelConfig(
        name=name,
        model=model,
        group="modern",
        reasoning="low" if reasoning == "low" else "off",
        prompt_styles=("clean",),
        temperature=0.75,
        max_tokens=512,
        max_connections=16,
        timeout=600,
        attempt_timeout=180,
        price_model=model.removeprefix("openrouter/"),
    )


# --------------------------------------------------------------------------------------
# prices
# --------------------------------------------------------------------------------------


def test_price_lookup_tolerates_the_openrouter_prefix() -> None:
    assert FAKE.get("openrouter/x/cheap") is FAKE.get("x/cheap")
    assert FAKE.get("x/absent") is None


def test_price_for_fails_closed() -> None:
    with pytest.raises(MissingPriceError, match="no OpenRouter price"):
        price_for(FAKE, "x/absent")


def test_token_cost_is_input_plus_output() -> None:
    price = Price("x", input_usd=1e-6, output_usd=1e-5)
    assert token_cost_usd(1000, 100, price) == pytest.approx(1000 * 1e-6 + 100 * 1e-5)


def test_models_payload_parsing_skips_unusable_entries() -> None:
    payload: dict[str, Any] = {
        "data": [
            {"id": "a/b", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"id": "c/d", "pricing": {"prompt": "free"}},
            {"id": "e/f"},
            "junk",
        ]
    }
    prices = _parse_models_payload(payload)
    assert set(prices) == {"a/b"}
    assert prices["a/b"].output_usd == pytest.approx(2e-6)


def test_models_payload_without_prices_is_an_error() -> None:
    with pytest.raises(CostError):
        _parse_models_payload({"data": []})
    with pytest.raises(CostError):
        _parse_models_payload({"nope": 1})


def test_price_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "prices.json"
    _write_prices(FAKE, path)
    again = _read_prices(path)
    assert again is not None
    assert again.fetched_at == FAKE.fetched_at
    assert again.get("x/dear") == FAKE.get("x/dear")


def test_broken_price_cache_reads_as_missing(tmp_path: Path) -> None:
    path = tmp_path / "prices.json"
    path.write_text("{not json", encoding="utf-8")
    assert _read_prices(path) is None


# --------------------------------------------------------------------------------------
# counted tokens and the estimate
# --------------------------------------------------------------------------------------


def test_count_tokens_is_additive() -> None:
    assert count_tokens([]) == 0
    one = count_tokens(["hola mundo"])
    assert one > 0
    assert count_tokens(["hola mundo", "hola mundo"]) == 2 * one


def test_estimate_arithmetic_with_a_fake_price_table(monkeypatch: pytest.MonkeyPatch) -> None:
    unit = Unit(config(), "clean", "pilot")
    monkeypatch.setattr(
        "sesgo._cost.unit_input_tokens", lambda _unit, _language="es": (200, 10_000)
    )
    estimate = estimate_unit(unit, FAKE)
    assert estimate.samples == 200
    assert estimate.input_tokens == 10_000
    assert estimate.output_tokens == 200 * DEFAULT_OUTPUT_TOKENS["off"]
    assert estimate.output_source == "default"
    assert estimate.usd == pytest.approx(10_000 * 1e-7 + 4_000 * 2e-7)
    assert estimate.key == "pilot/cheap__clean"


def test_estimate_uses_the_reasoning_default_for_a_low_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = Unit(config(reasoning="low"), "clean", "pilot")
    monkeypatch.setattr("sesgo._cost.unit_input_tokens", lambda _unit, _language="es": (10, 100))
    assert estimate_unit(unit, FAKE).output_tokens == 10 * DEFAULT_OUTPUT_TOKENS["low"]


def test_estimate_prefers_a_measured_output_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    unit = Unit(config(), "clean", "full")
    monkeypatch.setattr("sesgo._cost.unit_input_tokens", lambda _unit, _language="es": (100, 1_000))
    ledger: list[LedgerEntry] = [
        {
            "unit": "pilot/cheap__clean",
            "config": "cheap",
            "prompt_style": "clean",
            "status": "ok",
            "samples": 200,
            "output_tokens": 1_400,
        }
    ]
    estimate = estimate_unit(unit, FAKE, ledger)
    assert estimate.output_source == "measured"
    assert estimate.output_tokens == 700  # 1400 / 200 = 7 tokens per sample


def test_estimate_fails_closed_without_a_price(monkeypatch: pytest.MonkeyPatch) -> None:
    unit = Unit(config(model="openrouter/x/unknown"), "clean", "pilot")
    monkeypatch.setattr("sesgo._cost.unit_input_tokens", lambda _unit, _language="es": (1, 1))
    with pytest.raises(MissingPriceError):
        estimate_unit(unit, FAKE)


def test_measured_output_falls_back_to_another_style() -> None:
    ledger: list[LedgerEntry] = [
        {"config": "a", "prompt_style": "paper", "status": "ok", "samples": 10, "output_tokens": 50}
    ]
    assert measured_output_tokens(ledger, "a", "clean") == pytest.approx(5.0)
    assert measured_output_tokens(ledger, "b", "clean") is None


def test_measured_output_ignores_failed_lines() -> None:
    ledger: list[LedgerEntry] = [
        {
            "config": "a",
            "prompt_style": "clean",
            "status": "failed",
            "samples": 10,
            "output_tokens": 50,
        }
    ]
    assert measured_output_tokens(ledger, "a", "clean") is None


# --------------------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------------------


def test_entry_spend_prefers_the_larger_of_the_two_costs() -> None:
    assert entry_spend_usd({"cost_tokens_usd": 1.0, "cost_credits_usd": 2.0}) == 2.0
    assert entry_spend_usd({"cost_tokens_usd": 3.0, "cost_credits_usd": 1.0}) == 3.0
    assert entry_spend_usd({"cost_tokens_usd": 1.5, "cost_credits_usd": None}) == 1.5
    assert entry_spend_usd({}) == 0.0


def test_ledger_total_sums_the_billed_amounts() -> None:
    entries: list[LedgerEntry] = [
        {"cost_tokens_usd": 1.0, "cost_credits_usd": 2.0},
        {"cost_tokens_usd": 0.5, "cost_credits_usd": None},
    ]
    assert ledger_spent_usd(entries) == pytest.approx(2.5)


def test_unit_is_done_only_for_a_successful_line() -> None:
    entries: list[LedgerEntry] = [
        {"unit": "pilot/a__clean", "status": "failed"},
        {"unit": "pilot/b__clean", "status": "ok"},
    ]
    assert unit_is_done(entries, "pilot/b__clean") is True
    assert unit_is_done(entries, "pilot/a__clean") is False
    assert unit_is_done(entries, "pilot/c__clean") is False


def test_ledger_append_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    assert read_ledger(path) == []
    first: LedgerEntry = {"unit": "pilot/a__clean", "status": "ok", "cost_tokens_usd": 0.5}
    append_ledger(first, path)
    append_ledger({"unit": "pilot/b__clean", "status": "ok", "cost_tokens_usd": 0.25}, path)
    entries = read_ledger(path)
    assert [entry["unit"] for entry in entries] == ["pilot/a__clean", "pilot/b__clean"]
    assert ledger_spent_usd(entries) == pytest.approx(0.75)


def test_broken_ledger_line_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"unit": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(CostError, match=":2:"):
        read_ledger(path)


# --------------------------------------------------------------------------------------
# in-flight marker (V9)
# --------------------------------------------------------------------------------------


def test_inflight_marker_round_trip_and_clear(tmp_path: Path) -> None:
    path = tmp_path / "inflight.json"
    assert read_inflight(path) is None
    marker: Inflight = {
        "timestamp": "2026-09-21T10:00:00+00:00",
        "unit": "full/m__clean",
        "stage": "full",
        "config": "m",
        "credits_before": 20.5,
    }
    write_inflight(marker, path)
    assert read_inflight(path) == marker
    clear_inflight(path)
    assert read_inflight(path) is None
    clear_inflight(path)  # removing a missing marker is not an error


def test_unreadable_inflight_marker_is_treated_as_absent(tmp_path: Path) -> None:
    path = tmp_path / "inflight.json"
    path.write_text("not json", encoding="utf-8")
    assert read_inflight(path) is None


def test_interrupted_entry_books_the_credits_delta() -> None:
    marker: Inflight = {
        "timestamp": "2026-09-21T10:00:00+00:00",
        "unit": "full/m__clean",
        "stage": "full",
        "config": "m",
        "prompt_style": "clean",
        "credits_before": 20.0,
    }
    entry = interrupted_entry(marker, credits_now=20.043)
    assert entry["status"] == "interrupted"
    assert entry["unit"] == "full/m__clean"
    assert entry["cost_tokens_usd"] == 0.0
    assert entry["cost_credits_usd"] == pytest.approx(0.043)
    # The spend reaches the budget gate, and the unit is still not done.
    assert entry_spend_usd(entry) == pytest.approx(0.043)
    assert unit_is_done([entry], "full/m__clean") is False


def test_interrupted_entry_never_books_a_negative_delta() -> None:
    marker: Inflight = {"unit": "full/m__clean", "credits_before": 20.0}
    assert interrupted_entry(marker, credits_now=19.9)["cost_credits_usd"] == 0.0


# --------------------------------------------------------------------------------------
# budget and credits
# --------------------------------------------------------------------------------------


def credits(remaining: float, usage: float = 0.0) -> Credits:
    return Credits(total_credits=remaining + usage, total_usage=usage, read_at=0.0)


def test_explicit_budget_wins_and_never_touches_the_network() -> None:
    assert resolve_budget(0.0001, spent=5.0) == 0.0001


def test_default_budget_is_capped_by_the_credits_minus_the_reserve() -> None:
    reading = credits(remaining=8.30, usage=16.70)
    # 8.30 - 8.0 pins the documented reserve of USD 0.30, which is never spent.
    assert resolve_budget(None, spent=0.0, credits=reading) == pytest.approx(8.0)
    # Money already spent raises the budget by the same amount it lowered the credits,
    # so the gate `spent + estimate > budget` stays equivalent to `estimate <= credits`.
    assert resolve_budget(None, spent=1.0, credits=reading) == pytest.approx(9.0)


def test_default_budget_never_passes_the_project_cap() -> None:
    reading = credits(remaining=1000.0)
    assert resolve_budget(None, spent=0.0, credits=reading) == BUDGET_CAP_USD


def test_default_budget_fails_closed_when_the_balance_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a balance the run must stop, never fall back to the cap: that would let
    # the gate open a unit the account cannot pay for.
    def unreachable(*_a: object, **_k: object) -> Credits:
        raise CostError("credits endpoint unreachable")

    monkeypatch.setattr("sesgo._cost.fetch_credits", unreachable)
    with pytest.raises(CostError):
        resolve_budget(None, spent=0.0)


def test_poll_credits_returns_as_soon_as_it_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter([credits(10.0, 1.0), credits(10.0, 1.0), credits(10.0, 3.0)])
    monkeypatch.setattr("sesgo._cost.fetch_credits", lambda *_a, **_k: next(readings))
    slept: list[float] = []
    after, settled = poll_credits(
        credits(10.0, 1.0), min_delta=1.0, timeout_s=60.0, interval_s=1.0, sleep=slept.append
    )
    assert settled is True
    assert after.total_usage == 3.0
    assert slept == [1.0, 1.0]


def test_poll_credits_gives_up_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sesgo._cost.fetch_credits", lambda *_a, **_k: credits(10.0, 1.0))
    after, settled = poll_credits(
        credits(10.0, 1.0), min_delta=1.0, timeout_s=0.0, interval_s=0.0, sleep=lambda _s: None
    )
    assert settled is False
    assert after.total_usage == 1.0
