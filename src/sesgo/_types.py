"""Shared types. This module is the contract between the specs in `specs/`."""

from typing import Final, Literal, TypedDict, get_args

Category = Literal["racismo", "genero", "clasismo", "xenofobia"]
Language = Literal["es", "en"]
DatasetVersion = Literal["paper", "head"]
Split = Literal["ambig", "disambig"]
Polarity = Literal["neg", "nonneg"]
Role = Literal["target", "other", "unknown"]
PromptStyle = Literal["clean", "paper"]
ParseStage = Literal["strict", "lenient", "none"]
Outcome = Literal["answered", "invalid_response_format", "refusal", "no_response"]

CATEGORIES: Final[tuple[Category, ...]] = get_args(Category)
LANGUAGES: Final[tuple[Language, ...]] = get_args(Language)
DATASET_VERSIONS: Final[tuple[DatasetVersion, ...]] = get_args(DatasetVersion)
SPLITS: Final[tuple[Split, ...]] = get_args(Split)
ROLES: Final[tuple[Role, ...]] = get_args(Role)
PROMPT_STYLES: Final[tuple[PromptStyle, ...]] = get_args(PromptStyle)

LETTERS: Final[tuple[str, str, str]] = ("A", "B", "C")


class Option(TypedDict):
    """One answer option in upstream order."""

    text: str
    role: Role


class Source(TypedDict):
    """Provenance of a record."""

    commit: str
    file: str
    sheet: str
    row: int


class Record(TypedDict):
    """One line of `data/sesgo-{dataset_version}-{language}.jsonl`."""

    id: str
    category: Category
    language: Language
    dataset_version: DatasetVersion
    context_condition: Split
    question_polarity: Polarity
    context: str
    question: str
    options: list[Option]
    label_role: Role
    bbq: bool
    source: Source


class SampleMeta(TypedDict):
    """`Sample.metadata`. `roles` is aligned with `Sample.choices`."""

    category: Category
    language: Language
    dataset_version: DatasetVersion
    context_condition: Split
    question_polarity: Polarity
    context: str
    question: str
    roles: list[Role]
    label_role: Role
    bbq: bool
    category_split: str


class ScoreValue(TypedDict):
    """`Score.value` of the SESGO scorer.

    Every key is present on every sample. Values are 0.0 or 1.0.

    `ft` and `fo` are per-sample indicators, so every headline metric is a function of means.
    """

    correct: float
    chose_target: float
    chose_other: float
    chose_unknown: float
    ft: float
    fo: float
    unparsed: float
    invalid: float
    refusal: float
    no_response: float
    truncated: float
    parse_strict: float
    parse_lenient: float


SCORE_KEYS: Final[tuple[str, ...]] = tuple(ScoreValue.__annotations__)
