"""The SESGO task: parameters, validation and self-describing metadata (spec 05).

The task wires the four earlier specs together: dataset (01), prompt and solver (02),
parser and scorer (03), metrics (04). Every parameter is validated with an informative
`ValueError` before any data is read, and every choice that can change a number is
written into `Task.metadata`, so a log from the monitoring runner describes itself.
"""

from typing import Any, Final, TypedDict

from inspect_ai import Task, task
from inspect_ai.log import HeadlineMetric
from inspect_ai.model import GenerateConfig

from sesgo._data import COMMITS, CONVERTER_VERSION, manifest_entry, validate_version_language
from sesgo._dataset import sesgo_dataset
from sesgo._metrics import ALL_GROUP, metric_key, sesgo_metrics
from sesgo._parser import PARSER_VERSION, REFUSAL_PATTERNS_VERSION
from sesgo._scorer import sesgo_scorer
from sesgo._solver import get_default_solver
from sesgo._types import (
    CATEGORIES,
    PROMPT_STYLES,
    Category,
    DatasetVersion,
    Language,
    PromptStyle,
)

TASK_VERSION: Final[str] = "1-A"
"""Bump the number for a change in scoring, prompt, dataset, parser, refusal patterns or
routing; bump the letter for an interface change (D16)."""

DEFAULT_TEMPERATURE: Final[float] = 0.75
"""Paper temperature (D11)."""

DEFAULT_MAX_TOKENS: Final[int] = 512
"""Room for a short chain of thought plus the answer line. `None` uses the provider
default, which is what a reasoning configuration needs."""

TIE_SIGNS: Final[tuple[int, ...]] = (-1, 0, 1)
"""Accepted values of `tie_sign` (D10)."""

MIN_TEMPERATURE: Final[float] = 0.0
MAX_TEMPERATURE: Final[float] = 2.0

HEADLINE_METRIC: Final[str] = metric_key("ambig", ALL_GROUP, "bias_score")
"""The one number that summarizes a SESGO run: the pooled bias score of the ambiguous
split, the paper's own headline. `inspect view` pulls it to the front of the log header
and the log listing shows it as the score of the run (V6)."""


class TaskMeta(TypedDict):
    """`Task.metadata` of the SESGO task.

    Everything here can change a reported number, so the monitoring history can group
    runs by these fields and refuse to compare runs that do not share them.
    """

    task_version: str
    dataset_version: DatasetVersion
    language: Language
    upstream_commit: str
    converter_version: str
    data_file: str
    data_sha256: str
    categories: list[Category]
    limit_per_category: int | None
    samples: int
    prompt_style: PromptStyle
    shuffle: bool
    parser_version: str
    refusal_patterns_version: str
    tie_sign: int
    temperature: float
    max_tokens: int | None


def resolve_categories(categories: str | list[str] | None) -> tuple[Category, ...]:
    """Normalize the `categories` parameter into a tuple of known categories.

    The Inspect command line turns `-T categories=racismo` into a string and
    `-T categories=racismo,genero` into a list, so both forms are accepted, and a
    comma-separated string is split again for the Python caller. `None` means all four.

    Returns:
        The selected categories, without duplicates, in the canonical order.

    Raises:
        ValueError: If the value is empty or names an unknown category.
    """
    if categories is None:
        return CATEGORIES
    raw = [categories] if isinstance(categories, str) else list(categories)
    names = [part.strip() for item in raw for part in str(item).split(",")]
    if not names or any(not name for name in names):
        raise ValueError(
            f"categories must not be empty, got {categories!r}; valid: {list(CATEGORIES)}"
        )
    unknown = sorted({name for name in names if name not in CATEGORIES})
    if unknown:
        raise ValueError(f"unknown categories {unknown}; valid: {list(CATEGORIES)}")
    return tuple(category for category in CATEGORIES if category in names)


def resolve_shuffle(prompt_style: PromptStyle, shuffle: bool | None) -> bool:
    """Resolve the option-shuffle flag.

    Option order is shuffled per sample (D4), except in the paper style, which must keep
    the upstream order because the upstream prompt numbers the options. `shuffle=None`
    takes the default of the style.

    Raises:
        ValueError: If the paper style is combined with `shuffle=True`.
    """
    if shuffle is None:
        return prompt_style == "clean"
    if shuffle and prompt_style == "paper":
        raise ValueError(
            "prompt_style='paper' requires shuffle=False: the upstream prompt numbers the "
            "options in upstream order and the upstream parser matches them in that order"
        )
    return shuffle


def _validate(
    prompt_style: PromptStyle,
    temperature: float,
    max_tokens: int | None,
    limit_per_category: int | None,
    tie_sign: int,
) -> None:
    """Check the parameters that no other module checks.

    Raises:
        ValueError: If any value is outside its documented range.
    """
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt_style {prompt_style!r}; valid: {list(PROMPT_STYLES)}")
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise ValueError(
            f"temperature must be in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}], got {temperature}"
        )
    if max_tokens is not None and max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1 or None, got {max_tokens}")
    if limit_per_category is not None and limit_per_category < 1:
        raise ValueError(f"limit_per_category must be >= 1 or None, got {limit_per_category}")
    if tie_sign not in TIE_SIGNS:
        raise ValueError(f"tie_sign must be one of {list(TIE_SIGNS)}, got {tie_sign!r}")


@task
def sesgo(
    categories: str | list[str] | None = None,
    language: Language = "es",
    dataset_version: DatasetVersion = "paper",
    prompt_style: PromptStyle = "clean",
    shuffle: bool | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    limit_per_category: int | None = None,
    tie_sign: int = 1,
) -> Task:
    """SESGO: social bias of a language model in Spanish (arXiv:2509.03329).

    Each sample gives a context, a question and three options: the historically
    discriminated group (`target`), another group (`other`) and "not enough information"
    (`unknown`). Ambiguous contexts have `unknown` as the correct answer; disambiguated
    contexts name the right group. The scorer reports accuracy plus the bias metrics of
    the paper, per split and per category.

    Args:
        categories: `racismo`, `genero`, `clasismo`, `xenofobia`; a comma-separated
            string or a list, or `None` for all four.
        language: `"es"` (the paper) or `"en"`.
        dataset_version: `"paper"` (4,156 prompts, D2) or `"head"`.
        prompt_style: `"clean"` (Spanish prompt, `RESPUESTA: X`, D5) or `"paper"`
            (the upstream English system message and option numbering, D12).
        shuffle: Shuffle the three options per sample with a seed derived from the sample
            id (D4). `None` means `True` for the clean style and `False` for the paper one.
        temperature: Sampling temperature, in `[0, 2]`. The paper uses 0.75 (D11).
        max_tokens: Token budget of one response, or `None` for the provider default.
            Use `None` for reasoning models, which spend tokens before answering.
        limit_per_category: Keep only this many samples per category, chosen
            deterministically and stratified over split and polarity. `None` keeps all.
        tie_sign: Sign of the bias score when Ft equals Fo: `1` (D10), `0` (upstream
            `np.sign`, used to reproduce the paper) or `-1`.

    Returns:
        The task.

    Raises:
        ValueError: If a parameter is unknown or outside its range, or if the dataset
            version and language pair does not exist upstream.
        FileNotFoundError: If the data has not been built; run `uv run sesgo-data build`.
    """
    selected = resolve_categories(categories)
    validate_version_language(dataset_version, language)
    _validate(prompt_style, temperature, max_tokens, limit_per_category, tie_sign)
    shuffle_options = resolve_shuffle(prompt_style, shuffle)

    dataset = sesgo_dataset(
        categories=selected,
        language=language,
        dataset_version=dataset_version,
        shuffle=shuffle_options,
        limit_per_category=limit_per_category,
    )
    entry = manifest_entry(dataset_version, language)
    metadata: TaskMeta = {
        "task_version": TASK_VERSION,
        "dataset_version": dataset_version,
        "language": language,
        "upstream_commit": COMMITS[dataset_version],
        "converter_version": CONVERTER_VERSION,
        "data_file": entry["file"],
        "data_sha256": entry["sha256"],
        "categories": list(selected),
        "limit_per_category": limit_per_category,
        "samples": len(dataset),
        "prompt_style": prompt_style,
        "shuffle": shuffle_options,
        "parser_version": PARSER_VERSION,
        "refusal_patterns_version": REFUSAL_PATTERNS_VERSION,
        "tie_sign": tie_sign,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return Task(
        dataset=dataset,
        solver=get_default_solver(prompt_style),
        scorer=sesgo_scorer(prompt_style),
        metrics=[sesgo_metrics(tie_sign)],
        headline_metric=HeadlineMetric(metric=HEADLINE_METRIC),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        version=TASK_VERSION,
        metadata=dict[str, Any](metadata),
    )
