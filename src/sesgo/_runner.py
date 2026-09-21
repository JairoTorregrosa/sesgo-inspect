"""`sesgo-run`: estimate, smoke, pilot, full and the ledger (spec 06).

The runner is built for an unattended `tmux` session:

* no TTY is assumed (`display="plain"`), every progress line carries a UTC timestamp
  and is flushed, so `tee` to a file loses nothing;
* one unit of work = (config, prompt style, stage). A unit runs through
  `inspect_ai.eval_set`, so retry and resume are native and a finished unit is skipped;
* a failed unit is recorded and the run continues with the next one;
* a budget gate runs *before* each unit, so a unit that would pass the cap never starts.

Exit codes: `0` done, `1` a usage or configuration error, `2` at least one unit failed,
`3` the budget gate stopped the run, `4` the V4 reasoning check failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from inspect_ai import eval_set  # pyright: ignore[reportUnknownVariableType]
from inspect_ai.log import EvalLog

from sesgo._cost import (
    BUDGET_CAP_USD,
    CREDITS_RESERVE_USD,
    TOKENIZER,
    CostError,
    Credits,
    Estimate,
    Inflight,
    LedgerEntry,
    MissingPriceError,
    append_ledger,
    clear_inflight,
    entry_spend_usd,
    estimate_unit,
    fetch_credits,
    inflight_path,
    interrupted_entry,
    ledger_path,
    ledger_spent_usd,
    load_prices,
    poll_credits,
    read_inflight,
    read_ledger,
    resolve_budget,
    unit_is_done,
    utc_now,
    write_inflight,
)
from sesgo._data import repo_root
from sesgo._panel import (
    STAGES,
    Panel,
    PanelError,
    Stage,
    Unit,
    load_panel,
    select_units,
)
from sesgo._task import TASK_VERSION, sesgo

__all__ = ["main", "run_stage"]

EXIT_OK: Final[int] = 0
EXIT_USAGE: Final[int] = 1
EXIT_UNIT_FAILED: Final[int] = 2
EXIT_BUDGET: Final[int] = 3
EXIT_CHECK_FAILED: Final[int] = 4

CREDITS_POLL_TIMEOUT_S: Final[float] = 60.0
"""Upper bound on the idle time after a unit while the credits endpoint settles."""

CREDITS_POLL_INTERVAL_S: Final[float] = 6.0

_CREDITS_MIN_DELTA_FRACTION: Final[float] = 0.25
"""A settled reading must show at least this fraction of the list-price cost. Measured:
OpenRouter charged 0.41 of list for `meta-llama/llama-3.1-8b-instruct` (a cheaper
routed provider), so a threshold of 0.5 never settled for that model."""

_CREDITS_MIN_DELTA_FLOOR: Final[float] = 1e-6

_RETRY_ATTEMPTS: Final[int] = 3
_FAIL_ON_ERROR: Final[float] = 0.02
_RETRY_ON_ERROR: Final[int] = 3


def log(message: str) -> None:
    """Print one progress line with a UTC timestamp, flushed for `tee` in tmux."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}Z] {message}", flush=True)


def logs_root(path: Path | None = None) -> Path:
    """Resolve the `logs/` directory; `None` means `<repo root>/logs`."""
    return path if path is not None else repo_root() / "logs"


# --------------------------------------------------------------------------------------
# estimate
# --------------------------------------------------------------------------------------


def _fmt_usd(value: float) -> str:
    return f"{value:9.4f}"


def print_estimates(estimates: Sequence[Estimate], stage: Stage) -> float:
    """Print the estimate table of a stage and return its total in USD."""
    header = (
        f"{'config':26} {'style':6} {'reas':5} {'samples':>7} {'in tok':>9} "
        f"{'out tok':>9} {'out src':>8} {'USD':>9} {'cum USD':>9}"
    )
    print(
        f"stage: {stage}   input tokens counted with tiktoken {TOKENIZER} "
        "(exact for OpenAI models, an approximation for the others)"
    )
    print(header)
    print("-" * len(header))
    total = 0.0
    for estimate in estimates:
        total += estimate.usd
        config = estimate.unit.config
        print(
            f"{config.name:26.26} {estimate.unit.prompt_style:6} {config.reasoning:5} "
            f"{estimate.samples:7d} {estimate.input_tokens:9d} {estimate.output_tokens:9d} "
            f"{estimate.output_source:>8} {_fmt_usd(estimate.usd)} {_fmt_usd(total)}"
        )
    print("-" * len(header))
    print(f"{'TOTAL':26} {'':6} {'':5} {'':7} {'':9} {'':9} {'':8} {_fmt_usd(total)}")
    return total


def _estimates_for(
    panel: Panel,
    stage: Stage,
    only: Sequence[str] | None,
    reasoning: str,
    refresh_prices: bool,
) -> tuple[list[Estimate], list[tuple[Unit, str]]]:
    """Estimate every selected unit, cheapest first.

    A unit whose model has no price is dropped, not run: the estimate fails closed
    (spec 06), and one unpriced model must not block the rest of the panel.

    Returns:
        `(estimates, unpriced)` where `unpriced` pairs a unit with the reason.

    Raises:
        PanelError: If the selection is empty.
    """
    units = select_units(panel, stage, only, reasoning)
    if not units:
        raise PanelError("no units selected; check --only and --reasoning")
    prices = load_prices(refresh=refresh_prices)
    ledger = read_ledger()
    estimates: list[Estimate] = []
    unpriced: list[tuple[Unit, str]] = []
    for unit in units:
        try:
            estimates.append(estimate_unit(unit, prices, ledger))
        except MissingPriceError as exc:
            unpriced.append((unit, str(exc)))
    # Cheapest first (D14): a mistake costs the least, and the budget gate stops the
    # run at the first unit that does not fit, leaving the cheap ones already done.
    estimates.sort(key=lambda estimate: (estimate.usd, estimate.unit.key))
    return estimates, unpriced


def command_estimate(args: argparse.Namespace) -> int:
    """Run `sesgo-run estimate`."""
    panel = load_panel(args.panel)
    stage: Stage = args.stage
    estimates, unpriced = _estimates_for(
        panel, stage, args.only, args.reasoning, args.refresh_prices
    )
    total = print_estimates(estimates, stage)
    for unit, reason in unpriced:
        print(f"NOT PRICED, will not run: {unit.key}: {reason}")
    spent = ledger_spent_usd(read_ledger())
    print(f"\nledger spent so far: USD {spent:.4f}")
    print(f"estimated for this selection: USD {total:.4f}")
    return EXIT_OK


# --------------------------------------------------------------------------------------
# running a unit
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitUsage:
    """Token usage of one finished unit, summed over every model it touched."""

    samples: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int


def usage_of(logs: Sequence[EvalLog]) -> UnitUsage:
    """Sum `EvalLog.stats.model_usage` over the logs `eval_set` returned for a unit."""
    samples = 0
    input_tokens = output_tokens = reasoning_tokens = total_tokens = 0
    for entry in logs:
        results = entry.results
        if results is not None:
            samples += results.completed_samples
        for usage in (entry.stats.model_usage or {}).values():
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            reasoning_tokens += usage.reasoning_tokens or 0
            total_tokens += usage.total_tokens
    return UnitUsage(
        samples=samples,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )


def _mentions_temperature(text: str) -> bool:
    return "temperature" in text.lower()


def _log_error_text(logs: Sequence[EvalLog]) -> str:
    parts: list[str] = []
    for entry in logs:
        if entry.error is not None:
            parts.append(entry.error.message)
        for sample in entry.samples or []:
            if sample.error is not None:
                parts.append(sample.error.message)
    return " | ".join(parts)[:2000]


def _eval_unit(unit: Unit, log_dir: Path, drop_temperature: bool) -> tuple[bool, list[EvalLog]]:
    config = unit.config
    task = sesgo(
        prompt_style=unit.prompt_style,
        limit_per_category=unit.limit_per_category,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    if drop_temperature:
        # GenerateConfig.merge skips None, so clearing must happen on the task itself.
        task.config.temperature = None
    extra: dict[str, Any] = dict(config.generate)
    success, logs = eval_set(
        tasks=task,
        log_dir=str(log_dir),
        model=config.model,
        model_args=dict(config.model_args),
        display="plain",
        log_format="eval",
        fail_on_error=_FAIL_ON_ERROR,
        retry_on_error=_RETRY_ON_ERROR,
        retry_attempts=_RETRY_ATTEMPTS,
        max_connections=config.max_connections,
        timeout=config.timeout,
        attempt_timeout=config.attempt_timeout,
        score=True,
        **extra,
    )
    return success, list(logs)


def prepaid_usd(ledger: Sequence[LedgerEntry], unit_key: str) -> float:
    """Money already booked for `unit_key` by interrupted attempts.

    `cost_tokens_usd` of a finished unit is computed from the *whole* log, and after a
    resume that log also holds the samples the killed attempt already paid for. Without
    this subtraction the same requests would be billed twice: once as `interrupted` and
    again inside the `ok` line.
    """
    return sum(
        entry_spend_usd(entry)
        for entry in ledger
        if entry.get("unit") == unit_key and entry.get("status") == "interrupted"
    )


def run_unit(
    unit: Unit, estimate: Estimate, root: Path, ledger: Sequence[LedgerEntry] = ()
) -> LedgerEntry:
    """Run one unit and build its ledger line.

    Args:
        unit: The unit of work.
        estimate: Its estimate, used for the credits-poll threshold and the record.
        root: The `logs/` directory.
        ledger: Ledger lines read before this unit, used to avoid billing the samples
            of an interrupted attempt twice.

    Returns:
        The ledger entry. Status is `ok` or `failed`; the caller appends it.
    """
    config = unit.config
    log_dir = unit.log_dir(root)
    log_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"unit {unit.key}: start, model={config.model}, samples~{estimate.samples}, "
        f"estimate USD {estimate.usd:.4f}, logs {log_dir}"
    )

    before: Credits | None = None
    try:
        before = fetch_credits()
    except CostError as exc:
        log(f"unit {unit.key}: credits reading unavailable before the unit ({exc})")

    if before is not None:
        # Drop the marker *before the first call*: from here on, money can be spent that
        # no ledger line would record if this process were killed (V9).
        marker: Inflight = {
            "timestamp": utc_now(),
            "unit": unit.key,
            "stage": unit.stage,
            "config": config.name,
            "model": config.model,
            "prompt_style": unit.prompt_style,
            "credits_before": before.total_usage,
            "estimate_usd": round(estimate.usd, 8),
        }
        write_inflight(marker)

    note: str | None = config.note
    try:
        success, logs = _eval_unit(unit, log_dir, drop_temperature=False)
        if not success and _mentions_temperature(_log_error_text(logs)):
            log(f"unit {unit.key}: provider rejected temperature; retrying without it")
            note = "retried without temperature: the provider rejected it"
            success, logs = _eval_unit(unit, log_dir, drop_temperature=True)
        error_text = "" if success else _log_error_text(logs)
    except Exception as exc:  # noqa: BLE001 - a failed unit must not stop the run
        success, logs = False, []
        error_text = f"{type(exc).__name__}: {exc}"[:2000]

    usage = usage_of(logs)
    cost_tokens_gross = (
        usage.input_tokens * estimate.price.input_usd
        + usage.output_tokens * estimate.price.output_usd
    )
    prepaid = prepaid_usd(ledger, unit.key)
    cost_tokens = max(cost_tokens_gross - prepaid, 0.0)
    if prepaid > 0:
        log(
            f"unit {unit.key}: USD {prepaid:.6f} of the token cost was already booked by "
            f"an interrupted attempt; this line bills USD {cost_tokens:.6f} of "
            f"USD {cost_tokens_gross:.6f}"
        )

    cost_credits: float | None = None
    settled = False
    after: Credits | None = None
    if before is not None:
        # Gross, never net: after a resume the net token cost can be 0, and a threshold
        # of 0 would call the very first (still lagging) reading "settled" and book a
        # spend far below the real one.
        threshold = max(cost_tokens_gross * _CREDITS_MIN_DELTA_FRACTION, _CREDITS_MIN_DELTA_FLOOR)
        try:
            after, settled = poll_credits(
                before,
                min_delta=threshold,
                timeout_s=CREDITS_POLL_TIMEOUT_S,
                interval_s=CREDITS_POLL_INTERVAL_S,
            )
            cost_credits = max(after.total_usage - before.total_usage, 0.0)
        except CostError as exc:
            log(f"unit {unit.key}: credits reading unavailable after the unit ({exc})")

    entry: LedgerEntry = {
        "timestamp": utc_now(),
        "unit": unit.key,
        "stage": unit.stage,
        "config": config.name,
        "model": config.model,
        "group": config.group,
        "reasoning": config.reasoning,
        "prompt_style": unit.prompt_style,
        "status": "ok" if success else "failed",
        "samples": usage.samples,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "total_tokens": usage.total_tokens,
        "cost_tokens_usd": round(cost_tokens, 8),
        "cost_tokens_gross_usd": round(cost_tokens_gross, 8),
        "prepaid_usd": round(prepaid, 8),
        "cost_credits_usd": None if cost_credits is None else round(cost_credits, 8),
        "credits_before": None if before is None else before.total_usage,
        "credits_after": None if after is None else after.total_usage,
        "credits_settled": settled,
        "price_input_per_mtok": estimate.price.input_per_mtok,
        "price_output_per_mtok": estimate.price.output_per_mtok,
        "estimate_usd": round(estimate.usd, 8),
        # File name only: the ledger is committed and must not carry a local path.
        "log_file": Path(logs[0].location).name if logs else "",
        "task_version": TASK_VERSION,
    }
    if note:
        entry["note"] = note
    if not success:
        entry["error"] = error_text or "eval_set reported failure"
    # The unit is accounted for in this entry, so the marker has done its job.
    clear_inflight()
    return entry


RECONCILE_SETTLE_TIMEOUT_S: Final[float] = 36.0
"""How long to let `total_usage` keep growing before booking an interrupted attempt."""

RECONCILE_SETTLE_INTERVAL_S: Final[float] = 6.0


def _settled_credits(
    timeout_s: float = RECONCILE_SETTLE_TIMEOUT_S,
    interval_s: float = RECONCILE_SETTLE_INTERVAL_S,
) -> Credits:
    """Read `total_usage` until two consecutive readings agree, or the timeout expires.

    OpenRouter settles usage seconds to minutes after the requests. Booking an
    interrupted attempt from the first reading would under-report it, and an
    under-reported spend is the dangerous direction: the budget gate would then open a
    unit the account cannot pay for. On timeout the last reading is returned.

    Raises:
        CostError: If the endpoint cannot be read at all.
    """
    latest = fetch_credits()
    deadline = time.time() + max(timeout_s, 0.0)
    while time.time() < deadline:
        time.sleep(interval_s)
        again = fetch_credits()
        if again.total_usage <= latest.total_usage:
            return again
        latest = again
    return latest


def reconcile_inflight() -> LedgerEntry | None:
    """Book the spend of an attempt that was killed before it could write a line (V9).

    A leftover `results/inflight.json` means the previous invocation died between the
    first API call and the ledger append. The money is gone from the account either way,
    so it is booked here from the credits delta, before the budget gate reads the
    ledger. Runs are sequential and single-caller (README), so the delta is attributable.

    Returns:
        The appended entry, or `None` when there was nothing to reconcile.
    """
    marker = read_inflight()
    if marker is None:
        return None
    try:
        now = _settled_credits()
    except CostError as exc:
        log(
            f"leftover in-flight marker for {marker.get('unit')} but the credits endpoint "
            f"is unreachable ({exc}); keeping {inflight_path()} for the next run"
        )
        return None
    entry = interrupted_entry(marker, now.total_usage)
    append_ledger(entry)
    clear_inflight()
    log(
        f"reconciled an interrupted attempt of {entry.get('unit')} "
        f"(marker {marker.get('timestamp')}): credits delta USD "
        f"{float(entry.get('cost_credits_usd') or 0.0):.6f} booked as status=interrupted"
    )
    return entry


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------


def run_stage(
    stage: Stage,
    only: Sequence[str] | None = None,
    reasoning: str = "all",
    budget: float | None = None,
    panel_file: Path | None = None,
    root: Path | None = None,
    refresh_prices: bool = False,
    dry_run: bool = False,
) -> int:
    """Run every selected unit of a stage, cheapest first, under a budget gate.

    Args:
        stage: `"smoke"`, `"pilot"` or `"full"`.
        only: Config names to run, or `None` for all.
        reasoning: `"off"`, `"low"` or `"all"`.
        budget: Budget in USD, or `None` for the live default
            `min(15, spent + credits - 0.30)`.
        panel_file: Path of `models.yaml`.
        root: The `logs/` directory.
        refresh_prices: Refetch the OpenRouter price list.
        dry_run: Print the plan and stop before the first call.

    Returns:
        An exit code.
    """
    panel = load_panel(panel_file)
    estimates, unpriced = _estimates_for(panel, stage, only, reasoning, refresh_prices)
    directory = logs_root(root)

    if not dry_run:
        reconcile_inflight()
    ledger = read_ledger()
    spent = ledger_spent_usd(ledger)
    credits: Credits | None = None
    if budget is None:
        credits = fetch_credits()
    limit = resolve_budget(budget, spent, credits=credits)

    total = print_estimates(estimates, stage)
    log(f"ledger spent USD {spent:.4f}; budget USD {limit:.4f}; selection estimate USD {total:.4f}")
    if credits is not None:
        log(
            f"budget default = min(cap {BUDGET_CAP_USD:.2f}, spent {spent:.4f} + credits "
            f"{credits.remaining:.4f} - reserve {CREDITS_RESERVE_USD:.2f})"
        )
    if dry_run:
        log("dry run: nothing was called")
        return EXIT_OK

    failures = len(unpriced)
    for unit, reason in unpriced:
        log(f"unit {unit.key}: SKIPPED, the estimate failed closed: {reason}")
    for estimate in estimates:
        unit = estimate.unit
        if unit_is_done(ledger, unit.key):
            log(f"unit {unit.key}: skipped, already complete in the ledger")
            continue
        if spent + estimate.usd > limit:
            log(
                f"BUDGET STOP before {unit.key}: spent USD {spent:.4f} + estimate USD "
                f"{estimate.usd:.4f} = USD {spent + estimate.usd:.4f} > budget USD {limit:.4f}. "
                "No call was made for this unit. Raise --budget or add credit."
            )
            return EXIT_BUDGET
        entry = run_unit(unit, estimate, directory, ledger)
        append_ledger(entry)
        ledger.append(entry)
        spend = entry_spend_usd(entry)
        spent += spend
        if entry.get("status") == "ok":
            log(
                f"unit {unit.key}: done, {entry.get('samples')} samples, "
                f"tokens in/out/reasoning "
                f"{entry.get('input_tokens')}/{entry.get('output_tokens')}/"
                f"{entry.get('reasoning_tokens')}, cost_tokens USD "
                f"{entry.get('cost_tokens_usd'):.6f}, cost_credits "
                f"{entry.get('cost_credits_usd')}, settled={entry.get('credits_settled')}, "
                f"running total USD {spent:.4f}"
            )
        else:
            failures += 1
            log(f"unit {unit.key}: FAILED, recorded and continuing: {entry.get('error', '')[:300]}")

    log(f"stage {stage} finished; {failures} failed unit(s); ledger total USD {spent:.4f}")
    if stage == "smoke":
        return check_reasoning(panel) or (EXIT_UNIT_FAILED if failures else EXIT_OK)
    return EXIT_UNIT_FAILED if failures else EXIT_OK


def check_reasoning(panel: Panel) -> int:
    """V4: assert that `reasoning: off` really means ~0 reasoning tokens.

    Reads the smoke ledger lines and prints one PASS/FAIL line per config.

    Returns:
        `0` when every config behaves as declared, `4` otherwise.
    """
    ledger = read_ledger()
    latest: dict[str, LedgerEntry] = {}
    for entry in ledger:
        if entry.get("stage") == "smoke" and entry.get("status") == "ok":
            latest[str(entry.get("config"))] = entry
    print("\nV4 reasoning switch (smoke stage, reasoning tokens from log.stats.model_usage)")
    header = f"{'config':26} {'declared':9} {'reasoning tok':>13} {'per sample':>11}  verdict"
    print(header)
    print("-" * len(header))
    failed = 0
    skipped = 0
    for config in panel.configs:
        entry = latest.get(config.name)
        if entry is None:
            skipped += 1
            print(
                f"{config.name:26.26} {config.reasoning:9} {'-':>13} {'-':>11}  SKIP (no smoke run)"
            )
            continue
        tokens = int(entry.get("reasoning_tokens") or 0)
        samples = max(int(entry.get("samples") or 0), 1)
        per_sample = tokens / samples
        ok = tokens == 0 if config.reasoning == "off" else tokens > 0
        failed += 0 if ok else 1
        verdict = "PASS" if ok else "FAIL"
        print(
            f"{config.name:26.26} {config.reasoning:9} {tokens:13d} {per_sample:11.1f}  {verdict}"
        )
    print("-" * len(header))
    if failed:
        print(f"V4 FAILED for {failed} config(s): fix models.yaml (model_args/generate) and rerun")
        return EXIT_CHECK_FAILED
    checked = len(panel.configs) - skipped
    tail = f", {skipped} config(s) skipped for lack of a smoke run" if skipped else ""
    print(
        f"V4 PASS on {checked} config(s): every 'off' config produced 0 reasoning tokens "
        f"and every 'low' config more than 0{tail}"
    )
    return EXIT_OK


# --------------------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------------------


def command_ledger(args: argparse.Namespace) -> int:
    """Run `sesgo-run ledger`."""
    if args.add is not None:
        note, amount = args.add
        try:
            usd = float(amount)
        except ValueError:
            print(f"--add: {amount!r} is not a number", file=sys.stderr)
            return EXIT_USAGE
        entry: LedgerEntry = {
            "timestamp": utc_now(),
            "unit": f"manual/{note[:60]}",
            "stage": "manual",
            "status": "manual",
            "cost_tokens_usd": usd,
            "cost_credits_usd": usd,
            "note": note,
        }
        path = append_ledger(entry)
        print(f"added manual ledger line: USD {usd:.4f} ({note}) -> {path}")
        return EXIT_OK

    entries = read_ledger()
    if not entries:
        print(f"ledger is empty ({ledger_path()})")
        return EXIT_OK
    header = (
        f"{'timestamp':20} {'unit':44} {'status':7} {'samples':>7} "
        f"{'tokens USD':>11} {'credits USD':>12} {'billed USD':>11}"
    )
    print(header)
    print("-" * len(header))
    total = 0.0
    for entry in entries:
        spend = entry_spend_usd(entry)
        total += spend
        credits = entry.get("cost_credits_usd")
        credits_text = "-" if credits is None else f"{float(credits):12.6f}"
        print(
            f"{str(entry.get('timestamp', '')):20.20} {str(entry.get('unit', '')):44.44} "
            f"{str(entry.get('status', '')):7} {int(entry.get('samples') or 0):7d} "
            f"{float(entry.get('cost_tokens_usd') or 0.0):11.6f} {credits_text:>12} "
            f"{spend:11.6f}"
        )
    print("-" * len(header))
    print(f"TOTAL billed (max of tokens and credits per line): USD {total:.4f}")
    token_total = sum(float(entry.get("cost_tokens_usd") or 0.0) for entry in entries)
    credits_total = sum(
        float(entry.get("cost_credits_usd") or 0.0)
        for entry in entries
        if entry.get("cost_credits_usd") is not None
    )
    print(f"  sum of cost_tokens_usd:  USD {token_total:.4f}")
    print(f"  sum of cost_credits_usd: USD {credits_total:.4f}")
    if token_total > 0:
        ratio = credits_total / token_total
        print(f"  credits / tokens ratio:  {ratio:.2f} (V7 wants 0.80 - 1.20)")
    try:
        reading = fetch_credits()
        print(
            f"  OpenRouter now: total_usage {reading.total_usage:.6f}, "
            f"remaining USD {reading.remaining:.4f}"
        )
    except CostError as exc:
        print(f"  OpenRouter credits unavailable: {exc}")
    pending = read_inflight()
    if pending is not None:
        print(
            f"  WARNING: {inflight_path()} is left over from an interrupted attempt of "
            f"{pending.get('unit')} ({pending.get('timestamp')}). Its spend is NOT in the "
            "total above yet; the next `sesgo-run smoke|pilot|full` books it as an "
            "`interrupted` line from the credits delta."
        )
    return EXIT_OK


def command_check(args: argparse.Namespace) -> int:
    """Run `sesgo-run check-reasoning` on the recorded smoke runs."""
    return check_reasoning(load_panel(args.panel))


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAMES",
        help="config names from models.yaml, comma-separated; repeatable",
    )
    parser.add_argument(
        "--reasoning",
        choices=["off", "low", "all"],
        default="all",
        help="keep only configs with this reasoning setting (default: all)",
    )
    parser.add_argument("--panel", type=Path, default=None, help="path of models.yaml")
    parser.add_argument("--log-dir", type=Path, default=None, help="logs directory")
    parser.add_argument(
        "--refresh-prices", action="store_true", help="refetch the OpenRouter price list"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sesgo-run",
        description="Run the SESGO monitoring panel and keep the spend ledger.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    estimate = sub.add_parser("estimate", help="print the cost table, call nothing")
    _add_common(estimate)
    estimate.add_argument("--stage", choices=list(STAGES), default="pilot")
    estimate.set_defaults(func=command_estimate)

    for stage in STAGES:
        runner = sub.add_parser(stage, help=f"run the {stage} stage")
        _add_common(runner)
        runner.add_argument(
            "--budget",
            type=float,
            default=None,
            help="USD cap for this run (default: min(15, spent + OpenRouter credits - 0.30))",
        )
        runner.add_argument(
            "--dry-run", action="store_true", help="print the plan and call nothing"
        )
        runner.set_defaults(func=_stage_command(stage))

    ledger = sub.add_parser("ledger", help="print the spend ledger")
    ledger.add_argument(
        "--add",
        nargs=2,
        metavar=("NOTE", "USD"),
        help="add a manual spend line, for real calls made outside the runner",
    )
    ledger.set_defaults(func=command_ledger)

    check = sub.add_parser("check-reasoning", help="re-print the V4 verdict from the ledger")
    check.add_argument("--panel", type=Path, default=None, help="path of models.yaml")
    check.set_defaults(func=command_check)
    return parser


def _stage_command(stage: Stage) -> Any:
    def command(args: argparse.Namespace) -> int:
        return run_stage(
            stage,
            only=args.only,
            reasoning=args.reasoning,
            budget=args.budget,
            panel_file=args.panel,
            root=args.log_dir,
            refresh_prices=args.refresh_prices,
            dry_run=args.dry_run,
        )

    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of `sesgo-run`. Returns an exit code."""
    args = _parser().parse_args(list(argv) if argv is not None else None)
    handler: Any = args.func
    try:
        return int(handler(args))
    except (PanelError, CostError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
