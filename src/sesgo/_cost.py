"""Prices, token counts, budget and the spend ledger (spec 06).

Three numbers must never be confused:

* **estimate** - counted input tokens times the OpenRouter list price, plus an output
  guess. It is what the budget gate uses *before* a unit runs.
* **`cost_tokens_usd`** - the same arithmetic on the token counts the provider actually
  reported in `EvalLog.stats.model_usage`. Exact in tokens, list-price in money.
* **`cost_credits_usd`** - the delta of `total_usage` from OpenRouter's credits endpoint
  around the unit. This is real money, but the endpoint lags by seconds to minutes, so
  the runner polls for it and keeps both numbers.

The budget gate uses the larger of the two per ledger line, so a lagging credits reading
can never make us believe we spent less than we did.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

import httpx
import tiktoken
from dotenv import load_dotenv

from sesgo._data import repo_root
from sesgo._dataset import sesgo_dataset
from sesgo._panel import Unit
from sesgo._prompts import PAPER_SYSTEM_MESSAGE, format_clean_prompt, format_paper_prompt
from sesgo._types import Language, PromptStyle

__all__ = [
    "BUDGET_CAP_USD",
    "CREDITS_RESERVE_USD",
    "DEFAULT_OUTPUT_TOKENS",
    "TOKENIZER",
    "CostError",
    "Credits",
    "Estimate",
    "Inflight",
    "LedgerEntry",
    "MissingPriceError",
    "Price",
    "PriceTable",
    "append_ledger",
    "clear_inflight",
    "count_tokens",
    "entry_spend_usd",
    "estimate_unit",
    "fetch_credits",
    "inflight_path",
    "interrupted_entry",
    "ledger_path",
    "ledger_spent_usd",
    "load_prices",
    "poll_credits",
    "price_for",
    "prices_path",
    "read_inflight",
    "read_ledger",
    "resolve_budget",
    "results_dir",
    "token_cost_usd",
    "unit_input_tokens",
    "unit_is_done",
    "write_inflight",
]

MODELS_URL: Final[str] = "https://openrouter.ai/api/v1/models"
CREDITS_URL: Final[str] = "https://openrouter.ai/api/v1/credits"

TOKENIZER: Final[str] = "o200k_base"
"""One tokenizer for every model. Providers tokenize differently, so a counted input
token is an approximation for anything that is not an OpenAI model; the estimate table
says so. It is exact in the sense that it counts real formatted prompts, not guesses."""

DEFAULT_OUTPUT_TOKENS: Final[Mapping[str, int]] = {"off": 20, "low": 600}
"""Output tokens per sample used before any pilot measurement exists (spec 06)."""

BUDGET_CAP_USD: Final[float] = 15.0
"""Hard project cap (D14)."""

CREDITS_RESERVE_USD: Final[float] = 0.30
"""Never spend the account down to zero: keep this much OpenRouter credit unused."""

PRICES_MAX_AGE_S: Final[float] = 24 * 3600.0

_HTTP_TIMEOUT: Final[float] = 30.0

_CHAT_OVERHEAD_TOKENS: Final[int] = 8
"""Per-message chat-template overhead (role markers and separators). Measured against
real OpenRouter usage: counted + overhead lands within a few percent of reported."""


class CostError(RuntimeError):
    """A cost, price or budget operation failed in a way that must stop the run."""


class MissingPriceError(CostError):
    """No price is known for a model. The estimate fails closed (spec 06)."""


# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------


def results_dir(root: Path | None = None) -> Path:
    """Return `results/`, creating it if needed.

    `SESGO_RESULTS_DIR` overrides the location, the same way `SESGO_DATA_DIR` overrides
    the data directory. Use it to exercise the runner without touching the committed
    ledger.

    Args:
        root: Repository root, or `None` to detect it.

    Returns:
        The directory.
    """
    env = os.environ.get("SESGO_RESULTS_DIR")
    path = (
        Path(env).expanduser() if env else (root if root is not None else repo_root()) / "results"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def prices_path(root: Path | None = None) -> Path:
    """Path of the price cache, `results/prices.json`."""
    return results_dir(root) / "prices.json"


def ledger_path(root: Path | None = None) -> Path:
    """Path of the spend ledger, `results/ledger.jsonl`."""
    return results_dir(root) / "ledger.jsonl"


def inflight_path(root: Path | None = None) -> Path:
    """Path of the in-flight marker, `results/inflight.json`. Not committed."""
    return results_dir(root) / "inflight.json"


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with seconds."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------------------
# prices
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    """List price of one OpenRouter model, in USD per token."""

    model_id: str
    input_usd: float
    output_usd: float

    @property
    def input_per_mtok(self) -> float:
        """Input price in USD per million tokens."""
        return self.input_usd * 1e6

    @property
    def output_per_mtok(self) -> float:
        """Output price in USD per million tokens."""
        return self.output_usd * 1e6


@dataclass(frozen=True)
class PriceTable:
    """Prices of every OpenRouter model, with the time they were fetched."""

    fetched_at: str
    prices: Mapping[str, Price]

    def get(self, model_id: str) -> Price | None:
        """Look a price up, tolerating the `openrouter/` prefix. `None` when unknown."""
        return self.prices.get(model_id.removeprefix("openrouter/"))


def _parse_models_payload(payload: object) -> dict[str, Price]:
    if not isinstance(payload, Mapping):
        raise CostError(f"{MODELS_URL}: expected a JSON object, got {type(payload).__name__}")
    data = cast(Mapping[str, Any], payload).get("data")
    if not isinstance(data, list):
        raise CostError(f"{MODELS_URL}: no 'data' list in the response")
    prices: dict[str, Price] = {}
    for item in cast(list[Any], data):
        if not isinstance(item, Mapping):
            continue
        entry = cast(Mapping[str, Any], item)
        model_id = entry.get("id")
        pricing = entry.get("pricing")
        if not isinstance(model_id, str) or not isinstance(pricing, Mapping):
            continue
        fields = cast(Mapping[str, Any], pricing)
        try:
            input_usd = float(cast(str, fields["prompt"]))
            output_usd = float(cast(str, fields["completion"]))
        except (KeyError, TypeError, ValueError):
            continue
        prices[model_id] = Price(model_id=model_id, input_usd=input_usd, output_usd=output_usd)
    if not prices:
        raise CostError(f"{MODELS_URL}: no usable prices in the response")
    return prices


def fetch_prices(timeout: float = _HTTP_TIMEOUT) -> PriceTable:
    """Fetch the OpenRouter price list. No API key is needed.

    Args:
        timeout: HTTP timeout in seconds.

    Returns:
        The price table.

    Raises:
        CostError: If the endpoint is unreachable or the payload is unusable.
    """
    try:
        response = httpx.get(MODELS_URL, timeout=timeout)
        response.raise_for_status()
        payload = cast(object, response.json())
    except httpx.HTTPError as exc:
        raise CostError(f"could not fetch {MODELS_URL}: {exc}") from exc
    return PriceTable(fetched_at=utc_now(), prices=_parse_models_payload(payload))


def _write_prices(table: PriceTable, path: Path) -> None:
    document = {
        "fetched_at": table.fetched_at,
        "source": MODELS_URL,
        "prices": {
            model_id: {"prompt": price.input_usd, "completion": price.output_usd}
            for model_id, price in sorted(table.prices.items())
        },
    }
    path.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def _read_prices(path: Path) -> PriceTable | None:
    if not path.exists():
        return None
    try:
        document = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return None
    if not isinstance(document, Mapping):
        return None
    mapping = cast(Mapping[str, Any], document)
    raw = mapping.get("prices")
    fetched_at = mapping.get("fetched_at")
    if not isinstance(raw, Mapping) or not isinstance(fetched_at, str):
        return None
    prices: dict[str, Price] = {}
    for model_id, value in cast(Mapping[str, Any], raw).items():
        if not isinstance(value, Mapping):
            continue
        fields = cast(Mapping[str, Any], value)
        try:
            prices[model_id] = Price(
                model_id=model_id,
                input_usd=float(cast(float, fields["prompt"])),
                output_usd=float(cast(float, fields["completion"])),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return PriceTable(fetched_at=fetched_at, prices=prices) if prices else None


def _age_seconds(fetched_at: str) -> float:
    try:
        stamp = datetime.fromisoformat(fetched_at)
    except ValueError:
        return float("inf")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return (datetime.now(UTC) - stamp).total_seconds()


def load_prices(
    path: Path | None = None,
    refresh: bool = False,
    max_age_s: float = PRICES_MAX_AGE_S,
    offline: bool = False,
) -> PriceTable:
    """Load prices from the cache, refreshing them when they are old or missing.

    Args:
        path: Cache path, or `None` for `results/prices.json`.
        refresh: Fetch even when the cache is fresh.
        max_age_s: Maximum age of a cached table before it is refetched.
        offline: Never fetch; fail if there is no cache.

    Returns:
        The price table.

    Raises:
        CostError: If prices can be neither read nor fetched.
    """
    target = path if path is not None else prices_path()
    cached = _read_prices(target)
    if not refresh and cached is not None and _age_seconds(cached.fetched_at) <= max_age_s:
        return cached
    if offline:
        if cached is not None:
            return cached
        raise CostError(f"no cached prices at {target} and offline was requested")
    try:
        table = fetch_prices()
    except CostError:
        if cached is not None:
            return cached
        raise
    _write_prices(table, target)
    return table


def price_for(table: PriceTable, model_id: str) -> Price:
    """Look a price up or fail closed.

    Args:
        table: The price table.
        model_id: An OpenRouter model id.

    Returns:
        The price.

    Raises:
        MissingPriceError: If the model has no price. The unit must not run (spec 06).
    """
    price = table.get(model_id)
    if price is None:
        raise MissingPriceError(
            f"no OpenRouter price for {model_id!r}; the estimate fails closed so the unit "
            "does not run. Check the model id or set `price_model` in models.yaml"
        )
    return price


def token_cost_usd(input_tokens: float, output_tokens: float, price: Price) -> float:
    """USD for a token count at list price. `output_tokens` includes reasoning tokens."""
    return input_tokens * price.input_usd + output_tokens * price.output_usd


# --------------------------------------------------------------------------------------
# counted input tokens
# --------------------------------------------------------------------------------------

_encoding_cache: dict[str, tiktoken.Encoding] = {}
_input_token_cache: dict[tuple[PromptStyle, int | None, Language], tuple[int, int]] = {}


def count_tokens(texts: Sequence[str], tokenizer: str = TOKENIZER) -> int:
    """Total tokens of many texts, counted with one `tiktoken` encoding."""
    if tokenizer not in _encoding_cache:
        _encoding_cache[tokenizer] = tiktoken.get_encoding(tokenizer)
    encoding = _encoding_cache[tokenizer]
    return sum(len(piece) for piece in encoding.encode_batch(list(texts)))


def unit_input_tokens(unit: Unit, language: Language = "es") -> tuple[int, int]:
    """Count the real input tokens of every prompt of a unit.

    Every prompt of the stage is formatted with the real templates (`_prompts.py`) and
    tokenized. The system message of the paper style is counted once per sample, and a
    small per-message chat overhead is added, because the provider bills the rendered
    chat request and not the bare text.

    Args:
        unit: The unit of work.
        language: Dataset language.

    Returns:
        `(samples, input_tokens)`.

    Raises:
        FileNotFoundError: If the dataset has not been built.
    """
    key = (unit.prompt_style, unit.limit_per_category, language)
    if key in _input_token_cache:
        return _input_token_cache[key]
    dataset = sesgo_dataset(
        categories=None,
        language=language,
        dataset_version="paper",
        shuffle=unit.prompt_style == "clean",
        limit_per_category=unit.limit_per_category,
    )
    prompts: list[str] = []
    for sample in dataset:
        metadata = sample.metadata or {}
        context = cast(str, metadata["context"])
        question = cast(str, metadata["question"])
        choices = list(sample.choices or [])
        if unit.prompt_style == "paper":
            prompts.append(PAPER_SYSTEM_MESSAGE)
            prompts.append(format_paper_prompt(context, question, choices))
        else:
            prompts.append(format_clean_prompt(context, question, choices, language))
    samples = len(dataset)
    total = count_tokens(prompts) + _CHAT_OVERHEAD_TOKENS * len(prompts)
    _input_token_cache[key] = (samples, total)
    return samples, total


# --------------------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------------------

LedgerStatus = Literal["ok", "failed", "manual"]


class LedgerEntry(TypedDict, total=False):
    """One line of `results/ledger.jsonl`. Never contains prompt text (spec 06)."""

    timestamp: str
    unit: str
    stage: str
    config: str
    model: str
    group: str
    reasoning: str
    prompt_style: str
    status: str
    samples: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cost_tokens_usd: float
    cost_tokens_gross_usd: float
    prepaid_usd: float
    cost_credits_usd: float | None
    credits_before: float | None
    credits_after: float | None
    credits_settled: bool
    price_input_per_mtok: float
    price_output_per_mtok: float
    estimate_usd: float
    log_file: str
    task_version: str
    error: str
    note: str


def read_ledger(path: Path | None = None) -> list[LedgerEntry]:
    """Read the ledger.

    Args:
        path: Ledger path, or `None` for `results/ledger.jsonl`.

    Returns:
        Every line, oldest first. A missing file is an empty ledger.

    Raises:
        CostError: If a line is not valid JSON.
    """
    target = path if path is not None else ledger_path()
    if not target.exists():
        return []
    entries: list[LedgerEntry] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = cast(object, json.loads(line))
        except json.JSONDecodeError as exc:
            raise CostError(f"{target}:{number}: not valid JSON: {exc}") from exc
        if isinstance(value, Mapping):
            entries.append(cast(LedgerEntry, value))
    return entries


def append_ledger(entry: LedgerEntry, path: Path | None = None) -> Path:
    """Append one line to the ledger and return the path written to."""
    target = path if path is not None else ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def entry_spend_usd(entry: LedgerEntry) -> float:
    """Money attributed to one ledger line.

    The credits delta is the truth, but it lags, so a settled credits reading wins and
    otherwise the larger of the two numbers is used. Never the smaller: an
    under-reported spend would let the budget gate open a unit we cannot pay for.
    """
    tokens = float(entry.get("cost_tokens_usd") or 0.0)
    credits = entry.get("cost_credits_usd")
    if credits is None:
        return tokens
    return max(tokens, float(credits))


def ledger_spent_usd(entries: Iterable[LedgerEntry]) -> float:
    """Total USD recorded in the ledger."""
    return sum(entry_spend_usd(entry) for entry in entries)


def unit_is_done(entries: Iterable[LedgerEntry], unit_key: str) -> bool:
    """Whether a line with this `Unit.key` already finished with status `ok`."""
    return any(entry.get("unit") == unit_key and entry.get("status") == "ok" for entry in entries)


# --------------------------------------------------------------------------------------
# in-flight marker (V9): the spend of a killed attempt must not vanish
# --------------------------------------------------------------------------------------


class Inflight(TypedDict, total=False):
    """`results/inflight.json`: the unit that is running right now.

    A ledger line is only written when a unit finishes, so a unit that is killed
    mid-run (`tmux kill-session`, SIGINT, a crash) spends real money that no line ever
    records. The runner therefore drops this marker *before the first call* and removes
    it when the unit ends; the next invocation finds a leftover marker, reads the
    credits endpoint again and books the difference as an `interrupted` line.
    """

    timestamp: str
    unit: str
    stage: str
    config: str
    model: str
    prompt_style: str
    credits_before: float
    estimate_usd: float


def write_inflight(marker: Inflight, path: Path | None = None) -> Path:
    """Record that a unit is about to start spending, and return the path written to."""
    target = path if path is not None else inflight_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(marker, ensure_ascii=False, sort_keys=True) + "\n", "utf-8")
    return target


def read_inflight(path: Path | None = None) -> Inflight | None:
    """Read a leftover in-flight marker, or `None` when there is none or it is unreadable."""
    target = path if path is not None else inflight_path()
    if not target.exists():
        return None
    try:
        value = cast(object, json.loads(target.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return None
    return cast(Inflight, value) if isinstance(value, Mapping) else None


def clear_inflight(path: Path | None = None) -> None:
    """Remove the in-flight marker."""
    target = path if path is not None else inflight_path()
    target.unlink(missing_ok=True)


def interrupted_entry(marker: Inflight, credits_now: float) -> LedgerEntry:
    """Build the ledger line that books the spend of a killed attempt.

    Args:
        marker: The leftover marker written before the killed attempt.
        credits_now: `total_usage` read on the next invocation.

    Returns:
        A line with status `interrupted`. `cost_tokens_usd` is 0 because the killed
        process never wrote a log we can count tokens from; the credits delta is the
        only evidence, and it is what the budget gate will charge.
    """
    delta = max(credits_now - float(marker.get("credits_before") or 0.0), 0.0)
    entry: LedgerEntry = {
        "timestamp": utc_now(),
        "unit": str(marker.get("unit", "unknown")),
        "stage": str(marker.get("stage", "unknown")),
        "config": str(marker.get("config", "")),
        "model": str(marker.get("model", "")),
        "prompt_style": str(marker.get("prompt_style", "")),
        "status": "interrupted",
        "samples": 0,
        "cost_tokens_usd": 0.0,
        "cost_credits_usd": round(delta, 8),
        "credits_before": float(marker.get("credits_before") or 0.0),
        "credits_after": credits_now,
        "credits_settled": True,
        "note": (
            "attempt interrupted (killed or crashed) and never wrote a ledger line; "
            f"spend booked on the next invocation from the credits delta, marker of "
            f"{marker.get('timestamp', '?')}"
        ),
    }
    return entry


def measured_output_tokens(
    entries: Sequence[LedgerEntry], config_name: str, prompt_style: str
) -> float | None:
    """Mean output tokens per sample measured for a config in an earlier stage.

    Returns `None` when the config has never run. A line of the same prompt style wins;
    otherwise any successful line of the config is used.
    """
    for same_style in (True, False):
        for entry in reversed(entries):
            if entry.get("status") != "ok" or entry.get("config") != config_name:
                continue
            if same_style and entry.get("prompt_style") != prompt_style:
                continue
            samples = int(entry.get("samples") or 0)
            output = int(entry.get("output_tokens") or 0)
            if samples > 0 and output > 0:
                return output / samples
    return None


# --------------------------------------------------------------------------------------
# estimate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """Predicted cost of one unit before it runs."""

    unit: Unit
    samples: int
    input_tokens: int
    output_tokens: int
    usd: float
    price: Price
    output_source: Literal["measured", "default"]

    @property
    def key(self) -> str:
        """Unit key."""
        return self.unit.key


def estimate_unit(
    unit: Unit,
    prices: PriceTable,
    ledger: Sequence[LedgerEntry] = (),
    language: Language = "es",
) -> Estimate:
    """Estimate the cost of one unit.

    Args:
        unit: The unit of work.
        prices: Price table.
        ledger: Ledger lines, used for a measured output-token mean.
        language: Dataset language.

    Returns:
        The estimate.

    Raises:
        MissingPriceError: If the model has no price.
        FileNotFoundError: If the dataset has not been built.
    """
    price = price_for(prices, unit.config.price_model)
    samples, input_tokens = unit_input_tokens(unit, language)
    measured = measured_output_tokens(ledger, unit.config.name, unit.prompt_style)
    per_sample = measured if measured is not None else DEFAULT_OUTPUT_TOKENS[unit.config.reasoning]
    output_tokens = int(round(per_sample * samples))
    return Estimate(
        unit=unit,
        samples=samples,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=token_cost_usd(input_tokens, output_tokens, price),
        price=price,
        output_source="measured" if measured is not None else "default",
    )


# --------------------------------------------------------------------------------------
# credits and budget
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Credits:
    """A reading of the OpenRouter credits endpoint."""

    total_credits: float
    total_usage: float
    read_at: float

    @property
    def remaining(self) -> float:
        """Credit left on the account."""
        return self.total_credits - self.total_usage


def _api_key() -> str:
    load_dotenv(repo_root() / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise CostError(
            "OPENROUTER_API_KEY is not set; put it in .env (it is never printed or logged)"
        )
    return key


def fetch_credits(timeout: float = _HTTP_TIMEOUT) -> Credits:
    """Read `total_credits` and `total_usage` from OpenRouter.

    The API key is read from the environment or `.env` and never printed.

    Args:
        timeout: HTTP timeout in seconds.

    Returns:
        The reading.

    Raises:
        CostError: If the endpoint is unreachable or the payload is unusable.
    """
    try:
        response = httpx.get(
            CREDITS_URL,
            headers={"Authorization": f"Bearer {_api_key()}"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = cast(object, response.json())
    except httpx.HTTPError as exc:
        raise CostError(f"could not read {CREDITS_URL}: {type(exc).__name__}") from exc
    if not isinstance(payload, Mapping):
        raise CostError(f"{CREDITS_URL}: unexpected payload")
    data = cast(Mapping[str, Any], payload).get("data")
    if not isinstance(data, Mapping):
        raise CostError(f"{CREDITS_URL}: no 'data' object in the response")
    fields = cast(Mapping[str, Any], data)
    try:
        return Credits(
            total_credits=float(cast(float, fields["total_credits"])),
            total_usage=float(cast(float, fields["total_usage"])),
            read_at=time.time(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CostError(f"{CREDITS_URL}: missing total_credits/total_usage") from exc


def poll_credits(
    before: Credits,
    min_delta: float,
    timeout_s: float = 90.0,
    interval_s: float = 6.0,
    sleep: Any = time.sleep,
) -> tuple[Credits, bool]:
    """Wait for the credits endpoint to catch up after a unit.

    OpenRouter settles usage asynchronously, so a reading taken right after the last
    response can still show the pre-run value. Poll until the delta reaches `min_delta`
    or the timeout expires; the caller keeps the reading either way and records whether
    it settled.

    Args:
        before: The reading taken before the unit.
        min_delta: Delta that counts as settled. Use a fraction of the token cost.
        timeout_s: Give up after this long.
        interval_s: Seconds between polls.
        sleep: Sleep function, injectable for tests.

    Returns:
        `(reading, settled)`.

    Raises:
        CostError: If the endpoint cannot be read at all.
    """
    deadline = time.time() + max(timeout_s, 0.0)
    latest = fetch_credits()
    while True:
        if latest.total_usage - before.total_usage >= min_delta:
            return latest, True
        if time.time() >= deadline:
            return latest, False
        sleep(interval_s)
        latest = fetch_credits()


def resolve_budget(
    explicit: float | None,
    spent: float,
    cap: float = BUDGET_CAP_USD,
    reserve: float = CREDITS_RESERVE_USD,
    credits: Credits | None = None,
) -> float:
    """Compute the budget of a run.

    The default is `min(cap, spent + remaining credits - reserve)`, which makes the
    single gate `spent + estimate > budget` enforce both the project cap (D14) and the
    balance actually on the account, with a reserve that is never spent.

    Args:
        explicit: A `--budget` value, or `None` for the live default.
        spent: Money already recorded in the ledger.
        cap: Project cap.
        reserve: Credit to leave unspent.
        credits: A credits reading, or `None` to fetch one.

    Returns:
        The budget in USD.

    Raises:
        CostError: If the credits endpoint cannot be read. The default budget fails
            closed on purpose: without a balance we cannot promise not to overspend.
    """
    if explicit is not None:
        return explicit
    reading = credits if credits is not None else fetch_credits()
    return min(cap, spent + reading.remaining - reserve)
