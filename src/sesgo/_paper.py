"""Published SESGO results, as typed constants (arXiv:2509.03329).

These are the reference values that spec 07 check V2 compares our re-scoring against.
Nothing here is computed: every number is transcribed from the paper.

Two things about the paper are corrected on purpose, and only these two:

1. **The captions of Tables 2 and 3 are swapped.** The table the paper captions
   "disambiguated" holds the *ambiguous* bias scores, and the table captioned
   "ambiguous" holds the *disambiguated* ones. The `"ambig"` / `"disambig"` keys below
   are the corrected reading. The pooled rows agree with Table A3 at T = 0.75 under
   this reading and disagree under the printed captions, which is the evidence.
2. Nothing else. Known internal inconsistencies of the paper are recorded in
   `KNOWN_INCONSISTENCIES`, not silently patched.

Table A3 is reported at three temperatures; only T = 0.75 is transcribed, because that
is the SESGO default (D11) and the temperature of check V2.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, get_args

from sesgo._metrics import ALL_GROUP, metric_key
from sesgo._types import CATEGORIES, SPLITS, Category, Split

__all__ = [
    "ANCHOR_PAPER_MODELS",
    "CATEGORY_BIAS_SCORES",
    "KNOWN_INCONSISTENCIES",
    "PAPER_MODELS",
    "PAPER_MODEL_LABELS",
    "PAPER_TEMPERATURE",
    "PAPER_TIE_SIGN",
    "PAPER_TOLERANCE",
    "PaperModel",
    "PooledReference",
    "TABLE_2_3_POOLED",
    "TABLE_A3",
    "paper_expected_metrics",
    "paper_model_for_anchor",
]

PaperModel = Literal[
    "gpt_4o_mini",
    "llama_31_instruct",
    "llama_31_uncensored",
    "deepseek_r1",
    "gemini_20_flash",
    "claude_35_haiku",
]
"""The six models the paper evaluated, in the column order of Tables 2, 3 and A3."""

PAPER_MODELS: Final[tuple[PaperModel, ...]] = get_args(PaperModel)

PAPER_MODEL_LABELS: Final[Mapping[PaperModel, str]] = {
    "gpt_4o_mini": "GPT-4o mini",
    "llama_31_instruct": "Llama 3.1 8B Instruct",
    "llama_31_uncensored": "Llama 3.1 8B Uncensored",
    "deepseek_r1": "DeepSeek R1",
    "gemini_20_flash": "Gemini 2.0 Flash",
    "claude_35_haiku": "Claude 3.5 Haiku",
}
"""Display names as printed in the paper."""

ANCHOR_PAPER_MODELS: Final[Mapping[str, PaperModel]] = {
    "llama-3.1-8b-instruct": "llama_31_instruct",
    "gpt-4o-mini": "gpt_4o_mini",
}
"""Our anchor model names (D13) mapped to the paper column they reproduce.

The paper ran the OpenRouter-era `meta-llama/llama-3.1-8b-instruct` and
`openai/gpt-4o-mini`. These are the only two panel models the paper also evaluated, so
they are the only two whose numbers can be compared directly to a published column.
The other four paper columns (Llama 3.1 8B Uncensored, DeepSeek R1, Gemini 2.0 Flash,
Claude 3.5 Haiku) are not in our panel and exist here only for check V2, which re-scores
the paper's own published per-row answers.
"""

PAPER_TEMPERATURE: Final[float] = 0.75
"""The temperature of the transcribed Table A3 block, and our default (D11)."""

PAPER_TIE_SIGN: Final[int] = 0
"""Tie sign that reproduces the paper.

Upstream uses `np.sign`, which is `0` when `ft == fo`. Our default is `+1` (D10), so
every comparison against these constants must pass `tie_sign=PAPER_TIE_SIGN`.
"""

PAPER_TOLERANCE: Final[float] = 0.001
"""Absolute tolerance of check V2. The paper prints three decimals."""


@dataclass(frozen=True)
class PooledReference:
    """One cell block of Table A3: the pooled numbers of one model and one split.

    Attributes:
        accuracy: Pooled accuracy over every prompt of the split.
        ft_minus_fo: Pooled `Ft - Fo`.
        bias_score: Pooled bias score (paper Eq. 1, with `sigma = 0` on a tie).
    """

    accuracy: float
    ft_minus_fo: float
    bias_score: float


TABLE_A3: Final[Mapping[PaperModel, Mapping[Split, PooledReference]]] = {
    "gpt_4o_mini": {
        "ambig": PooledReference(accuracy=0.806, ft_minus_fo=0.024, bias_score=0.196),
        "disambig": PooledReference(accuracy=0.926, ft_minus_fo=-0.002, bias_score=-0.074),
    },
    "llama_31_instruct": {
        "ambig": PooledReference(accuracy=0.320, ft_minus_fo=0.139, bias_score=0.694),
        "disambig": PooledReference(accuracy=0.858, ft_minus_fo=-0.018, bias_score=-0.143),
    },
    "llama_31_uncensored": {
        "ambig": PooledReference(accuracy=0.384, ft_minus_fo=0.148, bias_score=0.633),
        "disambig": PooledReference(accuracy=0.880, ft_minus_fo=-0.006, bias_score=-0.120),
    },
    "deepseek_r1": {
        "ambig": PooledReference(accuracy=0.899, ft_minus_fo=-0.001, bias_score=-0.101),
        "disambig": PooledReference(accuracy=0.738, ft_minus_fo=0.007, bias_score=0.262),
    },
    "gemini_20_flash": {
        "ambig": PooledReference(accuracy=0.898, ft_minus_fo=0.041, bias_score=0.110),
        "disambig": PooledReference(accuracy=0.890, ft_minus_fo=0.009, bias_score=0.110),
    },
    "claude_35_haiku": {
        "ambig": PooledReference(accuracy=0.772, ft_minus_fo=0.077, bias_score=0.241),
        "disambig": PooledReference(accuracy=0.773, ft_minus_fo=-0.013, bias_score=-0.227),
    },
}
"""Paper Table A3 at T = 0.75, pooled over the four categories."""


CATEGORY_BIAS_SCORES: Final[Mapping[Split, Mapping[PaperModel, Mapping[Category, float]]]] = {
    # Paper Table 2 (printed caption says "disambiguated"; the values are ambiguous).
    "ambig": {
        "gpt_4o_mini": {
            "genero": 0.006,
            "racismo": 0.050,
            "clasismo": 0.069,
            "xenofobia": 0.514,
        },
        "llama_31_instruct": {
            "genero": 0.510,
            "racismo": 0.530,
            "clasismo": 0.602,
            "xenofobia": 0.993,
        },
        "llama_31_uncensored": {
            "genero": 0.390,
            "racismo": 0.524,
            "clasismo": 0.595,
            "xenofobia": 0.907,
        },
        "deepseek_r1": {
            "genero": 0.000,
            "racismo": 0.039,
            "clasismo": 0.087,
            "xenofobia": -0.224,
        },
        "gemini_20_flash": {
            "genero": 0.037,
            "racismo": 0.019,
            "clasismo": 0.017,
            "xenofobia": 0.285,
        },
        "claude_35_haiku": {
            "genero": 0.206,
            "racismo": -0.085,
            "clasismo": 0.200,
            "xenofobia": 0.431,
        },
    },
    # Paper Table 3 (printed caption says "ambiguous"; the values are disambiguated).
    "disambig": {
        "gpt_4o_mini": {
            "genero": 0.061,
            "racismo": -0.070,
            "clasismo": 0.000,
            "xenofobia": -0.076,
        },
        "llama_31_instruct": {
            "genero": 0.105,
            "racismo": -0.191,
            "clasismo": -0.093,
            "xenofobia": -0.146,
        },
        "llama_31_uncensored": {
            "genero": 0.154,
            "racismo": -0.168,
            "clasismo": 0.053,
            "xenofobia": -0.096,
        },
        "deepseek_r1": {
            "genero": 0.202,
            "racismo": 0.273,
            "clasismo": 0.229,
            "xenofobia": -0.303,
        },
        "gemini_20_flash": {
            "genero": 0.000,
            "racismo": 0.114,
            "clasismo": 0.000,
            "xenofobia": 0.066,
        },
        "claude_35_haiku": {
            "genero": 0.000,
            "racismo": -0.321,
            "clasismo": 0.185,
            "xenofobia": -0.172,
        },
    },
}
"""Per-category bias scores of Tables 2 and 3, with the swapped captions corrected."""


TABLE_2_3_POOLED: Final[Mapping[Split, Mapping[PaperModel, float]]] = {
    "ambig": {
        "gpt_4o_mini": 0.196,
        "llama_31_instruct": 0.694,
        "llama_31_uncensored": 0.638,
        "deepseek_r1": -0.101,
        "gemini_20_flash": 0.110,
        "claude_35_haiku": 0.241,
    },
    "disambig": {
        "gpt_4o_mini": -0.074,
        "llama_31_instruct": -0.143,
        "llama_31_uncensored": -0.120,
        "deepseek_r1": 0.262,
        "gemini_20_flash": 0.110,
        "claude_35_haiku": -0.227,
    },
}
"""The `Pooled` row of Tables 2 and 3.

Kept separate from `TABLE_A3` because one cell disagrees with it; see
`KNOWN_INCONSISTENCIES`. Prefer `TABLE_A3` as the reference for pooled values.
"""


KNOWN_INCONSISTENCIES: Final[tuple[str, ...]] = (
    "Llama 3.1 8B Uncensored, ambiguous, pooled bias score: Table A3 (T=0.75) prints "
    "0.633 and the Pooled row of Table 2 prints 0.638. Both are recorded; neither is "
    "forced. Check V2 compares against Table A3.",
    "The captions of Tables 2 and 3 are swapped in the paper. CATEGORY_BIAS_SCORES "
    "stores the corrected reading; the pooled rows match Table A3 only this way.",
    "Sheet XENOFOBIA of the published results workbook has rotated model headers. Use "
    "sheet TODOS instead (spec 07, V2 details).",
)
"""Inconsistencies of the published tables, documented rather than patched."""


def paper_model_for_anchor(model: str) -> PaperModel | None:
    """Return the paper column that an anchor model reproduces.

    Args:
        model: One of our panel model names, for example `"gpt-4o-mini"`. A provider
            prefix such as `"openrouter/meta-llama/llama-3.1-8b-instruct"` is accepted;
            only the last path segment is matched.

    Returns:
        The `PaperModel` key, or `None` if the model is not an anchor.
    """
    return ANCHOR_PAPER_MODELS.get(model.rsplit("/", 1)[-1])


def paper_expected_metrics(model: PaperModel) -> dict[str, float]:
    """Flatten every published value of one model into SESGO metric keys.

    This is the form check V2 compares against: the same keys `summarize` produces, so a
    comparison is a plain dict walk.

    Returns:
        A mapping from `metric_key(split, group, name)` to the published value. The
        `all` group carries `accuracy`, `ft_minus_fo` and `bias_score` from Table A3;
        each category carries `bias_score` from Table 2 or 3.
    """
    out: dict[str, float] = {}
    for split in SPLITS:
        pooled = TABLE_A3[model][split]
        out[metric_key(split, ALL_GROUP, "accuracy")] = pooled.accuracy
        out[metric_key(split, ALL_GROUP, "ft_minus_fo")] = pooled.ft_minus_fo
        out[metric_key(split, ALL_GROUP, "bias_score")] = pooled.bias_score
        for category in CATEGORIES:
            out[metric_key(split, category, "bias_score")] = CATEGORY_BIAS_SCORES[split][model][
                category
            ]
    return out
