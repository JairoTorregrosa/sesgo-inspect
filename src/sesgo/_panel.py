"""The monitoring panel: `models.yaml` loading and validation (spec 06).

A panel entry is one *config*: a model plus the generation settings that change what
the model does (temperature, reasoning). A config crossed with a prompt style and a
stage is a *unit of work*, the thing the runner evaluates, estimates and bills.

To add a model a user adds one entry to `models.yaml`. Every validation error names the
entry and the field, so the message is actionable without reading this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

import yaml

from sesgo._data import repo_root
from sesgo._task import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from sesgo._types import PROMPT_STYLES, PromptStyle

__all__ = [
    "DEFAULT_PANEL_FILE",
    "STAGES",
    "STAGE_LIMITS",
    "Group",
    "Panel",
    "PanelConfig",
    "PanelDefaults",
    "PanelError",
    "Reasoning",
    "Stage",
    "Unit",
    "load_panel",
    "panel_path",
    "select_units",
]

Group = Literal["anchor", "modern"]
Reasoning = Literal["off", "low"]
Stage = Literal["smoke", "pilot", "full"]

GROUPS: Final[tuple[Group, ...]] = ("anchor", "modern")
REASONINGS: Final[tuple[Reasoning, ...]] = ("off", "low")
STAGES: Final[tuple[Stage, ...]] = ("smoke", "pilot", "full")

STAGE_LIMITS: Final[Mapping[Stage, int | None]] = {
    "smoke": 3,
    "pilot": 50,
    "full": None,
}
"""`limit_per_category` of each stage. `full` runs the whole dataset."""

DEFAULT_PANEL_FILE: Final[str] = "models.yaml"

DEFAULT_MAX_CONNECTIONS: Final[int] = 16
"""Parallel requests per unit. OpenRouter tolerates 16-32 for these models; the runner
keeps it static (not adaptive) so one unit cannot starve the next one."""

DEFAULT_TIMEOUT: Final[int] = 600
"""Seconds for a whole request including retries. A hung request must not stall a run."""

DEFAULT_ATTEMPT_TIMEOUT: Final[int] = 180
"""Seconds for one attempt. Reasoning models answer this dataset in under 30 s."""

_MODEL_PREFIX: Final[str] = "openrouter/"

_ENTRY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "model",
        "group",
        "reasoning",
        "prompt_styles",
        "model_args",
        "generate",
        "temperature",
        "max_tokens",
        "max_connections",
        "timeout",
        "attempt_timeout",
        "price_model",
        "note",
    }
)

_DEFAULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "temperature",
        "max_tokens",
        "max_connections",
        "timeout",
        "attempt_timeout",
    }
)

_NAME_CHARS: Final[frozenset[str]] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
)
"""A config name becomes a directory name, so it must not need quoting or escaping."""


class PanelError(ValueError):
    """`models.yaml` is invalid. The message names the entry and the field."""


@dataclass(frozen=True)
class PanelDefaults:
    """Values an entry inherits when it does not set them itself."""

    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    max_connections: int = DEFAULT_MAX_CONNECTIONS
    timeout: int = DEFAULT_TIMEOUT
    attempt_timeout: int = DEFAULT_ATTEMPT_TIMEOUT


@dataclass(frozen=True)
class PanelConfig:
    """One panel entry, with every default already resolved."""

    name: str
    model: str
    group: Group
    reasoning: Reasoning
    prompt_styles: tuple[PromptStyle, ...]
    temperature: float
    max_tokens: int | None
    max_connections: int
    timeout: int
    attempt_timeout: int
    price_model: str
    model_args: Mapping[str, Any] = field(default_factory=dict[str, Any])
    generate: Mapping[str, Any] = field(default_factory=dict[str, Any])
    note: str | None = None


@dataclass(frozen=True)
class Unit:
    """One unit of work: a config, a prompt style and a stage."""

    config: PanelConfig
    prompt_style: PromptStyle
    stage: Stage

    @property
    def name(self) -> str:
        """Directory-safe `{config}__{style}` name."""
        return f"{self.config.name}__{self.prompt_style}"

    @property
    def key(self) -> str:
        """Stable identity of the unit across runs, used as the ledger key."""
        return f"{self.stage}/{self.name}"

    @property
    def limit_per_category(self) -> int | None:
        """`limit_per_category` of the stage."""
        return STAGE_LIMITS[self.stage]

    def log_dir(self, logs_root: Path) -> Path:
        """Log directory of the unit: `logs/{stage}/{config}__{style}/`. Not created.

        The smoke stage adds a `panel/` level, because `logs/smoke/` also holds the
        one-off smoke logs of the parser verification and the two must not mix.
        """
        if self.stage == "smoke":
            return logs_root / "smoke" / "panel" / self.name
        return logs_root / self.stage / self.name


@dataclass(frozen=True)
class Panel:
    """The whole `models.yaml`."""

    defaults: PanelDefaults
    configs: tuple[PanelConfig, ...]
    path: Path | None = None

    def names(self) -> tuple[str, ...]:
        """Names of every config, in file order."""
        return tuple(config.name for config in self.configs)

    def by_name(self, name: str) -> PanelConfig:
        """Look one config up by its `name` field.

        Raises:
            PanelError: If no entry has that name.
        """
        for config in self.configs:
            if config.name == name:
                return config
        raise PanelError(f"unknown config {name!r}; known: {list(self.names())}")

    def select(
        self, only: Sequence[str] | None = None, reasoning: str = "all"
    ) -> tuple[PanelConfig, ...]:
        """Filter the panel by name and by reasoning setting.

        Args:
            only: Config names to keep, or `None` for all. A name that matches no entry
                is an error, because a typo must not silently run nothing.
            reasoning: `"off"`, `"low"` or `"all"`.

        Returns:
            The selected configs in file order.

        Raises:
            PanelError: If a name is unknown or `reasoning` is invalid.
        """
        if reasoning not in (*REASONINGS, "all"):
            raise PanelError(f"--reasoning must be off, low or all, got {reasoning!r}")
        chosen = self.configs
        if only is not None:
            wanted = [name.strip() for item in only for name in str(item).split(",")]
            wanted = [name for name in wanted if name]
            unknown = [name for name in wanted if name not in self.names()]
            if unknown:
                raise PanelError(f"unknown config names {unknown}; known: {list(self.names())}")
            chosen = tuple(config for config in chosen if config.name in wanted)
        if reasoning != "all":
            chosen = tuple(config for config in chosen if config.reasoning == reasoning)
        return chosen


def select_units(
    panel: Panel,
    stage: Stage,
    only: Sequence[str] | None = None,
    reasoning: str = "all",
) -> tuple[Unit, ...]:
    """Expand the selected configs into units of work.

    Args:
        panel: The loaded panel.
        stage: `"smoke"`, `"pilot"` or `"full"`.
        only: Config names to keep, or `None` for all.
        reasoning: `"off"`, `"low"` or `"all"`.

    Returns:
        One unit per (config, prompt style), in file order.

    Raises:
        PanelError: If the stage or a filter value is invalid.
    """
    if stage not in STAGES:
        raise PanelError(f"unknown stage {stage!r}; valid: {list(STAGES)}")
    return tuple(
        Unit(config=config, prompt_style=style, stage=stage)
        for config in panel.select(only, reasoning)
        for style in config.prompt_styles
    )


def panel_path(path: Path | None = None) -> Path:
    """Resolve the path of `models.yaml`; `None` means `<repo root>/models.yaml`."""
    return path if path is not None else repo_root() / DEFAULT_PANEL_FILE


def _as_mapping(value: object, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PanelError(f"{where}: expected a mapping, got {type(value).__name__}")
    mapping = cast(Mapping[object, Any], value)
    bad = [key for key in mapping if not isinstance(key, str)]
    if bad:
        raise PanelError(f"{where}: keys must be strings, got {bad!r}")
    return cast(Mapping[str, Any], mapping)


def _as_positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PanelError(f"{where}: expected an integer, got {value!r}")
    if value < 1:
        raise PanelError(f"{where}: must be >= 1, got {value}")
    return value


def _as_max_tokens(value: object, where: str) -> int | None:
    if value is None:
        return None
    return _as_positive_int(value, where)


def _as_temperature(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PanelError(f"{where}: expected a number, got {value!r}")
    number = float(value)
    if not 0.0 <= number <= 2.0:
        raise PanelError(f"{where}: must be in [0, 2], got {number}")
    return number


def _check_keys(mapping: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise PanelError(f"{where}: unknown keys {unknown}; allowed: {sorted(allowed)}")


def _defaults_from(raw: object) -> PanelDefaults:
    mapping = _as_mapping(raw, "defaults")
    _check_keys(mapping, _DEFAULT_KEYS, "defaults")
    base = PanelDefaults()
    return PanelDefaults(
        temperature=(
            _as_temperature(mapping["temperature"], "defaults.temperature")
            if "temperature" in mapping
            else base.temperature
        ),
        max_tokens=(
            _as_max_tokens(mapping["max_tokens"], "defaults.max_tokens")
            if "max_tokens" in mapping
            else base.max_tokens
        ),
        max_connections=(
            _as_positive_int(mapping["max_connections"], "defaults.max_connections")
            if "max_connections" in mapping
            else base.max_connections
        ),
        timeout=(
            _as_positive_int(mapping["timeout"], "defaults.timeout")
            if "timeout" in mapping
            else base.timeout
        ),
        attempt_timeout=(
            _as_positive_int(mapping["attempt_timeout"], "defaults.attempt_timeout")
            if "attempt_timeout" in mapping
            else base.attempt_timeout
        ),
    )


def _styles_from(raw: object, where: str) -> tuple[PromptStyle, ...]:
    if raw is None:
        return ("clean",)
    items = [raw] if isinstance(raw, str) else raw
    if not isinstance(items, Sequence) or isinstance(items, str | bytes):
        raise PanelError(f"{where}: expected a list of prompt styles, got {raw!r}")
    names = [str(item) for item in cast(Sequence[object], items)]
    if not names:
        raise PanelError(f"{where}: must not be empty; valid: {list(PROMPT_STYLES)}")
    unknown = sorted({name for name in names if name not in PROMPT_STYLES})
    if unknown:
        raise PanelError(f"{where}: unknown prompt styles {unknown}; valid: {list(PROMPT_STYLES)}")
    seen: list[PromptStyle] = []
    for style in PROMPT_STYLES:
        if style in names and style not in seen:
            seen.append(style)
    return tuple(seen)


def _config_from(raw: object, index: int, defaults: PanelDefaults) -> PanelConfig:
    where = f"configs[{index}]"
    mapping = _as_mapping(raw, where)
    name_raw = mapping.get("name")
    if not isinstance(name_raw, str) or not name_raw:
        raise PanelError(f"{where}.name: required, must be a non-empty string, got {name_raw!r}")
    where = f"configs[{index}] ({name_raw})"
    _check_keys(mapping, _ENTRY_KEYS, where)
    bad_chars = sorted(set(name_raw) - _NAME_CHARS)
    if bad_chars:
        raise PanelError(
            f"{where}.name: {name_raw!r} contains {bad_chars!r}; the name becomes a "
            "directory name, so use letters, digits, '.', '-' and '_' only"
        )

    model = mapping.get("model")
    if not isinstance(model, str) or "/" not in model:
        raise PanelError(
            f"{where}.model: required, must be a provider-qualified model id such as "
            f"'openrouter/openai/gpt-4o-mini', got {model!r}"
        )

    group = mapping.get("group")
    if group not in GROUPS:
        raise PanelError(f"{where}.group: must be one of {list(GROUPS)}, got {group!r}")

    reasoning = mapping.get("reasoning")
    if reasoning not in REASONINGS:
        raise PanelError(f"{where}.reasoning: must be one of {list(REASONINGS)}, got {reasoning!r}")

    model_args = _as_mapping(mapping.get("model_args"), f"{where}.model_args")
    generate = dict(_as_mapping(mapping.get("generate"), f"{where}.generate"))

    # `max_tokens` must reach the task, not the eval call: GenerateConfig.merge skips
    # None, so an eval-level override can raise the task value but never clear it.
    max_tokens_raw = mapping.get("max_tokens", generate.pop("max_tokens", defaults.max_tokens))
    temperature_raw = mapping.get("temperature", generate.pop("temperature", defaults.temperature))

    price_model = mapping.get("price_model")
    if price_model is not None and not isinstance(price_model, str):
        raise PanelError(f"{where}.price_model: expected a string, got {price_model!r}")
    note = mapping.get("note")
    if note is not None and not isinstance(note, str):
        raise PanelError(f"{where}.note: expected a string, got {note!r}")

    return PanelConfig(
        name=name_raw,
        model=model,
        group=group,
        reasoning=reasoning,
        prompt_styles=_styles_from(mapping.get("prompt_styles"), f"{where}.prompt_styles"),
        temperature=_as_temperature(temperature_raw, f"{where}.temperature"),
        max_tokens=_as_max_tokens(max_tokens_raw, f"{where}.max_tokens"),
        max_connections=_as_positive_int(
            mapping.get("max_connections", defaults.max_connections),
            f"{where}.max_connections",
        ),
        timeout=_as_positive_int(mapping.get("timeout", defaults.timeout), f"{where}.timeout"),
        attempt_timeout=_as_positive_int(
            mapping.get("attempt_timeout", defaults.attempt_timeout), f"{where}.attempt_timeout"
        ),
        price_model=price_model if price_model else model.removeprefix(_MODEL_PREFIX),
        model_args=model_args,
        generate=generate,
        note=note,
    )


def parse_panel(document: object, path: Path | None = None) -> Panel:
    """Validate an already-parsed `models.yaml` document (the result of `yaml.safe_load`).

    Raises:
        PanelError: If any entry or default is invalid.
    """
    mapping = _as_mapping(document, str(path) if path else "models.yaml")
    _check_keys(mapping, frozenset({"defaults", "configs"}), "models.yaml")
    defaults = _defaults_from(mapping.get("defaults"))
    raw_configs = mapping.get("configs")
    if not isinstance(raw_configs, list) or not raw_configs:
        raise PanelError("configs: required, must be a non-empty list of entries")
    configs = tuple(
        _config_from(entry, index, defaults)
        for index, entry in enumerate(cast(list[object], raw_configs))
    )
    seen: dict[str, int] = {}
    for index, config in enumerate(configs):
        if config.name in seen:
            raise PanelError(
                f"configs[{index}].name: duplicate name {config.name!r}, first used at "
                f"configs[{seen[config.name]}]; names identify log directories"
            )
        seen[config.name] = index
    return Panel(defaults=defaults, configs=configs, path=path)


def load_panel(path: Path | None = None) -> Panel:
    """Load and validate `models.yaml`; `None` means `<repo root>/models.yaml`.

    Raises:
        FileNotFoundError: If the file does not exist.
        PanelError: If the file is not valid YAML or an entry is invalid.
    """
    resolved = panel_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"panel file not found: {resolved}")
    try:
        document = cast(object, yaml.safe_load(resolved.read_text(encoding="utf-8")))
    except yaml.YAMLError as exc:
        raise PanelError(f"{resolved}: not valid YAML: {exc}") from exc
    return parse_panel(document, resolved)
