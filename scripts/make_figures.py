#!/usr/bin/env python3
"""Build the three README charts from `results/history.jsonl`.

Reads nothing but `results/history.jsonl` (aggregates only: no prompt, no model
response, no `.eval` log, no network). Writes `docs/figures/`:

    headline-{light,dark}.{svg,png}     aligned dot plots, 14 runs
    categories-{light,dark}.{svg,png}   |bias| heatmap by category
    reasoning-{light,dark}.{svg,png}    off/low dumbbells
    results.csv                         every number on the charts, plus the signed
                                        bias score, Ft, Fo and the interval bounds

Uncertainty
-----------
`history.jsonl` stores, per `(split, group)`, `n`, `accuracy`, `ft`, `fo` and
`unparsed_rate`. In the ambiguous split the correct answer is always "unknown" and
every error is either the target option or the other option, so those four rates
partition the group exactly and the group is fully described by four counts. An
item-level bootstrap of such a group is therefore *exactly* a multinomial resample
over those counts: drawing `n` items with replacement from a group whose items differ
only by which of four outcomes they produced has the same distribution as drawing one
multinomial vector. No item-level file is needed and nothing is approximated.

The disambiguated split has a fifth outcome (a wrong "unknown" answer), which the
`all` group still pins down because `ft` and `fo` are stored; its per-category rows
store only `ft - fo`, which does not identify `ft` and `fo` separately, so those rows
get no interval. They are marked in the CSV and are not plotted.

The bias score itself is not re-derived here: `sesgo._metrics.bias_score` is imported
and called on every replicate, including its exact-tie handling.

Determinism
-----------
`make figures-check` compares committed bytes. So: `svg.hashsalt` fixed (clip-path
ids), `metadata={"Date": None}` (no timestamp in the SVG/PNG), DejaVu Sans only (the
font matplotlib ships), and `svg.fonttype = "path"`.

`svg.fonttype = "path"` rather than `"none"`: with `"none"` the SVG carries
`font-family="DejaVu Sans"` and the *renderer* picks a font. GitHub, a browser on
Linux and a browser on macOS would each substitute something different, so the glyph
advances would stop matching the advances matplotlib used to place the text, and
labels laid out to just fit would overflow. `"path"` embeds the outlines matplotlib
actually measured, so the file renders identically everywhere and cannot break on a
machine without the font. The cost is a larger file and text that cannot be selected;
for a chart that is the right trade. It is also what makes the bytes stable: the
outlines come from the FreeType that the matplotlib wheel statically links, which the
exact pin in the `figures` dependency group fixes.

PNG bytes are *not* guaranteed across platforms (zlib and libpng differ), so
`--check` is strict on the SVG and the CSV and advisory on the PNG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from sesgo._metrics import bias_score, sign_is_decided  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
HISTORY = REPO / "results" / "history.jsonl"
OUT_DIR = REPO / "docs" / "figures"

N_BOOT: Final[int] = 4000
SEED: Final[int] = 20260921
CI_LEVEL: Final[float] = 95.0

PAPER = "arXiv:2509.03329"
CSV_LINK = "docs/figures/results.csv"

# Dataset category keys, in the reading order the charts use, with their English header.
CATEGORIES: Final[tuple[tuple[str, str], ...]] = (
    ("racismo", "racism"),
    ("genero", "gender"),
    ("clasismo", "classism"),
    ("xenofobia", "xenophobia"),
)
OUTLINED_CATEGORY: Final[str] = "xenofobia"

# The two runs that carry the accent ink in charts 1 and 2.
BEST_RUN = ("gemini-3.1-flash-lite-low", "clean")
REPLICATION_RUN = ("gpt-4o-mini", "paper")
PUBLISHED_ACCURACY: Final[float] = 0.806
PUBLISHED_BIAS: Final[float] = 0.196

# Models with an off/low pair, in the order chart 3 draws them (set at run time by
# ambiguous accuracy of the `off` run, descending, to match chart 1).
REASONING_PAIRS: Final[tuple[str, ...]] = (
    "deepseek-v4-flash",
    "gemini-3.1-flash-lite",
    "gpt-5.4-nano",
    "qwen3.5-9b",
)


# --------------------------------------------------------------------------------------
# theme
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """One colour scheme. `bg` is opaque and matches the GitHub page it sits on."""

    name: str
    bg: str
    ink: str
    muted: str
    grid: str
    accent: str


LIGHT = Theme(
    name="light",
    bg="#ffffff",
    ink="#1f2328",
    muted="#5a636d",
    grid="#dbe0e6",
    accent="#3b4f9e",
)
DARK = Theme(
    name="dark",
    bg="#0d1117",
    ink="#e6edf3",
    muted="#9aa4ae",
    grid="#2b323b",
    accent="#8fa4f3",
)
THEMES = (LIGHT, DARK)


def apply_rcparams() -> None:
    """Pin every rcParam the output bytes depend on."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "svg.hashsalt": "sesgo-figures-v1",
            "svg.fonttype": "path",
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "axes.unicode_minus": False,
            "path.simplify": False,
        }
    )


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One full evaluation unit: a config run under one prompt style."""

    config: str
    model_label: str
    prompt_style: str
    reasoning: str
    temperature: float
    max_tokens: int
    tie_sign: int
    created: str
    samples: int
    reasoning_tokens_per_item: float
    metrics: dict[str, float]
    ses: dict[str, float]

    @property
    def key(self) -> tuple[str, str]:
        return (self.config, self.prompt_style)

    @property
    def base_model(self) -> str:
        return self.config[: -len("-low")] if self.config.endswith("-low") else self.config


@dataclass(frozen=True)
class Group:
    """Outcome counts of one `(run, split, group)` cell."""

    n: int
    correct: int
    ft: int
    fo: int
    unparsed: int
    other: int  # wrong "unknown" answers; always 0 in the ambiguous split

    @property
    def identified(self) -> bool:
        return self.n > 0

    def counts(self) -> tuple[int, ...]:
        return (self.correct, self.ft, self.fo, self.unparsed, self.other)


@dataclass(frozen=True)
class Estimate:
    """Point estimate plus a percentile bootstrap interval, or NaN bounds."""

    value: float
    lo: float
    hi: float


def load_runs(path: Path) -> list[Run]:
    """Read `history.jsonl` and keep the full-stage units, ordered deterministically."""
    runs: list[Run] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        if record.get("stage") != "full":
            continue
        runs.append(
            Run(
                config=str(record["config"]),
                model_label=str(record["config"]),
                prompt_style=str(record["prompt_style"]),
                reasoning=str(record["reasoning"]),
                temperature=float(record["temperature"]),
                max_tokens=int(record["max_tokens"]),
                tie_sign=int(record["tie_sign"]),
                created=str(record["created"]),
                samples=int(record["samples"]),
                reasoning_tokens_per_item=float(record["reasoning_tokens"])
                / float(record["samples"]),
                metrics={k: float(v) for k, v in record["metrics"].items()},
                ses={k: float(v) for k, v in record.get("ses", {}).items()},
            )
        )
    if not runs:
        raise SystemExit(f"no full-stage runs in {path}")
    return sorted(runs, key=lambda r: (r.config, r.prompt_style))


def run_label(run: Run, has_reasoning_pair: bool) -> str:
    """Human row label: the axis a run's sibling varies along.

    A config that exists in both `off` and `low` is labelled by its reasoning setting; a
    config that was run under both prompt styles is labelled by its style. Every run in
    `history.jsonl` has exactly one sibling, so the 14 labels are unique.
    """
    if run.reasoning == "low" or has_reasoning_pair:
        return f"{run.base_model} · reasoning {run.reasoning}"
    style = "paper prompt" if run.prompt_style == "paper" else "default prompt"
    return f"{run.base_model} · {style}"


def _count(rate: float, n: int, what: str) -> int:
    """Turn a rate stored to six decimals back into the exact integer count."""
    raw = rate * n
    nearest = round(raw)
    if abs(raw - nearest) > 0.02:
        raise ValueError(f"{what}: rate {rate} x n {n} = {raw} is not an integer count")
    return int(nearest)


def group_of(run: Run, split: str, group: str) -> Group | None:
    """Outcome counts of one cell, or `None` when the aggregates do not identify them."""
    prefix = f"{split}/{group}"
    n_key = f"{prefix}/n"
    if n_key not in run.metrics:
        return None
    n = int(round(run.metrics[n_key]))
    accuracy = run.metrics[f"{prefix}/accuracy"]
    unparsed_rate = run.metrics[f"{prefix}/unparsed_rate"]
    correct = _count(accuracy, n, f"{prefix}/accuracy")
    unparsed = _count(unparsed_rate, n, f"{prefix}/unparsed_rate")

    ft_key, fo_key = f"{prefix}/ft", f"{prefix}/fo"
    if ft_key in run.metrics and fo_key in run.metrics:
        ft = _count(run.metrics[ft_key], n, ft_key)
        fo = _count(run.metrics[fo_key], n, fo_key)
    else:
        # Category rows store only `ft - fo`. In the ambiguous split every error is `ft`
        # or `fo`, so the pair is pinned by the sum and the difference; in the
        # disambiguated split a wrong "unknown" is also possible and it is not.
        if split != "ambig":
            return None
        diff = _count(run.metrics[f"{prefix}/ft_minus_fo"], n, f"{prefix}/ft_minus_fo")
        errors = n - correct - unparsed
        if (errors + diff) % 2 != 0:
            raise ValueError(f"{prefix}: errors {errors} and ft-fo {diff} have different parity")
        ft = (errors + diff) // 2
        fo = (errors - diff) // 2
    other = n - correct - ft - fo - unparsed
    if other < 0:
        raise ValueError(f"{prefix}: counts exceed n ({correct}, {ft}, {fo}, {unparsed}) > {n}")
    if split == "ambig" and other != 0:
        raise ValueError(f"{prefix}: ambiguous split should have no fifth outcome, got {other}")
    return Group(n=n, correct=correct, ft=ft, fo=fo, unparsed=unparsed, other=other)


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------


def _cell_seed(run: Run, split: str, group: str) -> int:
    """Stable seed per cell, so an interval never depends on iteration order."""
    token = f"{run.config}|{run.prompt_style}|{split}|{group}".encode()
    digest = hashlib.blake2b(token, digest_size=4).digest()
    return (SEED + int.from_bytes(digest, "big")) % (2**32)


@dataclass(frozen=True)
class CellStats:
    """Everything the charts and the CSV need about one cell."""

    group: Group
    accuracy: Estimate
    bias_magnitude: Estimate
    bias_signed: float
    ft_minus_fo: float
    sign_decided: bool
    accuracy_se_boot: float
    ft_minus_fo_se_boot: float


_NAN = float("nan")


def bootstrap_cell(run: Run, split: str, group: str) -> CellStats | None:
    """Item-level bootstrap of one cell: multinomial over its outcome counts."""
    cell = group_of(run, split, group)
    if cell is None or not cell.identified:
        return None
    n = cell.n
    counts = np.array(cell.counts(), dtype=np.float64)
    probs = counts / float(n)
    rng = np.random.default_rng(_cell_seed(run, split, group))
    draws = rng.multinomial(n, probs, size=N_BOOT).astype(np.float64)

    acc_rep = draws[:, 0] / n
    ft_rep = draws[:, 1] / n
    fo_rep = draws[:, 2] / n
    ties = draws[:, 1] == draws[:, 2]
    # `bias_score` is the project's own Eq. 1, tie handling included. Called per
    # replicate rather than re-derived here.
    bias_rep = np.array(
        [
            abs(bias_score(a, t, o, run.tie_sign, tie=bool(z)))
            for a, t, o, z in zip(acc_rep, ft_rep, fo_rep, ties, strict=True)
        ],
        dtype=np.float64,
    )

    lo_q = (100.0 - CI_LEVEL) / 2.0
    hi_q = 100.0 - lo_q
    point_acc = cell.correct / n
    point_bias = run.metrics[f"{split}/{group}/bias_score"]
    diff = run.metrics[f"{split}/{group}/ft_minus_fo"]
    se_key = f"{split}/{group}/ft_minus_fo"
    se = run.ses.get(se_key, _NAN)
    return CellStats(
        group=cell,
        accuracy=Estimate(
            point_acc, float(np.percentile(acc_rep, lo_q)), float(np.percentile(acc_rep, hi_q))
        ),
        bias_magnitude=Estimate(
            abs(point_bias),
            float(np.percentile(bias_rep, lo_q)),
            float(np.percentile(bias_rep, hi_q)),
        ),
        bias_signed=point_bias,
        ft_minus_fo=diff,
        sign_decided=sign_is_decided(diff, se) if not math.isnan(se) else False,
        accuracy_se_boot=float(np.std(acc_rep, ddof=1)),
        ft_minus_fo_se_boot=float(np.std(ft_rep - fo_rep, ddof=1)),
    )


# --------------------------------------------------------------------------------------
# shared chart furniture
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """The 14 rows every chart shares, in one order."""

    runs: list[Run]
    labels: list[str]
    stats: dict[tuple[str, str], dict[tuple[str, str], CellStats]]
    protocol_line: str


def build_panel(runs: list[Run]) -> Panel:
    """Order the runs by ambiguous accuracy descending and bootstrap every cell."""
    reasoning_models = {r.base_model for r in runs if r.reasoning == "low"}
    ordered = sorted(runs, key=lambda r: (-r.metrics["ambig/all/accuracy"], r.config))
    labels = [run_label(r, r.base_model in reasoning_models) for r in ordered]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"row labels are not unique: {labels}")

    stats: dict[tuple[str, str], dict[tuple[str, str], CellStats]] = {}
    for run in ordered:
        cells: dict[tuple[str, str], CellStats] = {}
        for split in ("ambig", "disambig"):
            for group in ("all", *(key for key, _ in CATEGORIES)):
                cell = bootstrap_cell(run, split, group)
                if cell is not None:
                    cells[(split, group)] = cell
        stats[run.key] = cells

    first = ordered[0]
    n_ambig = int(first.metrics["ambig/all/n"])
    date = min(r.created for r in runs)[:10]
    tokens = " / ".join(f"{t:,}" for t in sorted({r.max_tokens for r in runs}))
    protocol = (
        f"SESGO ({PAPER}) · Spanish · ambiguous context, n = {n_ambig:,} items per run"
        f" · temperature {first.temperature:g}, max_tokens {tokens}"
        f" · run {date} · results/history.jsonl"
    )
    return Panel(runs=ordered, labels=labels, stats=stats, protocol_line=protocol)


def check_against_history(panel: Panel) -> None:
    """Verify the multinomial resample against the real item-level bootstrap.

    `history.jsonl` stores standard errors that `sesgo._metrics.bootstrap_ses` computed
    by resampling the actual scored samples of each `.eval` log. If the multinomial
    draw used here really is the same bootstrap, it must land on the same standard
    errors for the statistics whose distribution is smooth. `accuracy` and `Ft - Fo`
    are; the *signed* bias score is not, because its sign flips at `Ft = Fo` and its
    standard error is then dominated by a rare event that 1,000 replicates estimate
    badly. That is the reason the charts plot the magnitude, never the signed score.

    Raises:
        SystemExit: If any cell drifts further than Monte-Carlo noise explains.
    """
    worst = 0.0
    compared = 0
    for run in panel.runs:
        for (split, group), cell in panel.stats[run.key].items():
            for name, boot in (
                ("accuracy", cell.accuracy_se_boot),
                ("ft_minus_fo", cell.ft_minus_fo_se_boot),
            ):
                stored = run.ses.get(f"{split}/{group}/{name}")
                if stored is None or math.isnan(stored) or stored == 0.0:
                    continue
                drift = abs(boot - stored) / stored
                worst = max(worst, drift)
                compared += 1
                if drift > 0.15 and abs(boot - stored) > 0.0015:
                    raise SystemExit(
                        f"bootstrap drift: {run.config}/{run.prompt_style} "
                        f"{split}/{group}/{name} multinomial SE {boot:.6f} against the "
                        f"item-level SE {stored:.6f} recorded in history.jsonl"
                    )
    print(
        f"ok  multinomial bootstrap matches the item-level SEs in history.jsonl "
        f"({compared} comparisons, worst relative drift {worst:.1%})"
    )


def is_accented(run: Run) -> bool:
    return run.key in (BEST_RUN, REPLICATION_RUN)


def new_figure(width: float, height: float, theme: Theme) -> Figure:
    fig = plt.figure(figsize=(width, height), facecolor=theme.bg)
    fig.patch.set_alpha(1.0)
    return fig


def style_axes(ax: plt.Axes, theme: Theme) -> None:
    ax.set_facecolor(theme.bg)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.grid)
    ax.tick_params(colors=theme.muted, labelsize=8.5, length=3, width=0.8)


TITLE_Y: Final[float] = 0.982
SUBTITLE_Y: Final[float] = 0.930
FOOTNOTE_Y: Final[float] = 0.018
MARGIN_X: Final[float] = 0.012
SUBTITLE_SIZE: Final[float] = 9.0
FOOTNOTE_SIZE: Final[float] = 8.4


def _wrap(text: str, fig_width_in: float, fontsize: float) -> str:
    """Hard-wrap each paragraph so no line can run past the right edge of the figure.

    DejaVu Sans averages about 0.52 em per character at these sizes; 0.56 is used so the
    estimate errs on the side of wrapping early. Wrapping here rather than relying on
    matplotlib's `wrap=True` keeps the line breaks in the committed bytes explicit.
    """
    usable_in = fig_width_in * (1.0 - 2.0 * MARGIN_X)
    columns = max(40, int(usable_in * 72.0 / (fontsize * 0.56)))
    return "\n".join(textwrap.fill(para, width=columns) for para in text.split("\n"))


def add_header(fig: Figure, theme: Theme, title: str, subtitle: str, footnote: str) -> None:
    """Title, protocol subtitle and footnote, at fixed figure-relative positions."""
    width_in = float(fig.get_size_inches()[0])
    fig.text(
        MARGIN_X,
        TITLE_Y,
        _wrap(title, width_in, 13.5),
        color=theme.ink,
        fontsize=13.5,
        fontweight="bold",
        va="top",
        ha="left",
        linespacing=1.25,
    )
    fig.text(
        MARGIN_X,
        SUBTITLE_Y,
        _wrap(subtitle, width_in, SUBTITLE_SIZE),
        color=theme.muted,
        fontsize=SUBTITLE_SIZE,
        va="top",
        ha="left",
        linespacing=1.45,
    )
    fig.text(
        MARGIN_X,
        FOOTNOTE_Y,
        _wrap(footnote, width_in, FOOTNOTE_SIZE),
        color=theme.muted,
        fontsize=FOOTNOTE_SIZE,
        va="bottom",
        ha="left",
        linespacing=1.45,
    )


def contrast_text(rgb: tuple[float, float, float]) -> str:
    """Pick black or white text for a filled cell, by relative luminance."""

    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * channel(rgb[0]) + 0.7152 * channel(rgb[1]) + 0.0722 * channel(rgb[2])
    return "#0b0b0b" if lum > 0.42 else "#ffffff"


def fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


# --------------------------------------------------------------------------------------
# chart 1: headline
# --------------------------------------------------------------------------------------


def draw_headline(panel: Panel, theme: Theme) -> Figure:
    """Aligned horizontal dot plots: ambiguous accuracy and |bias| magnitude."""
    fig = new_figure(12.0, 8.0, theme)
    left = fig.add_axes((0.262, 0.148, 0.345, 0.690))
    right = fig.add_axes((0.652, 0.148, 0.345, 0.690))

    rows = list(enumerate(panel.runs))
    for ax, key, heading in (
        (left, "accuracy", "Accuracy in ambiguous context"),
        (right, "bias", "|bias score| — magnitude; lower is better"),
    ):
        style_axes(ax, theme)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(len(panel.runs) - 0.5, -0.8)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
        ax.set_yticks([])
        for gridline in (0.25, 0.5, 0.75, 1.0):
            ax.axvline(gridline, color=theme.grid, lw=0.7, zorder=0)
        ax.text(
            0.0,
            1.022,
            heading,
            transform=ax.transAxes,
            color=theme.ink,
            fontsize=9.5,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        for index, run in rows:
            cell = panel.stats[run.key][("ambig", "all")]
            est = cell.accuracy if key == "accuracy" else cell.bias_magnitude
            accent = is_accented(run)
            colour = theme.accent if accent else theme.ink
            ax.plot(
                [est.lo, est.hi],
                [index, index],
                color=colour,
                lw=1.2,
                alpha=0.85 if accent else 0.55,
                solid_capstyle="butt",
                zorder=2,
            )
            for bound in (est.lo, est.hi):
                ax.plot(
                    [bound, bound],
                    [index - 0.17, index + 0.17],
                    color=colour,
                    lw=1.2,
                    alpha=0.85 if accent else 0.55,
                    zorder=2,
                )
            ax.plot(
                [est.value],
                [index],
                marker="o",
                markersize=7.0 if accent else 5.8,
                color=colour,
                markeredgecolor=theme.bg,
                markeredgewidth=0.9,
                zorder=3,
                linestyle="none",
            )
            if est.value > 0.80:
                ax.text(
                    est.lo - 0.022,
                    index,
                    fmt(est.value),
                    color=colour,
                    fontsize=8.4,
                    va="center",
                    ha="right",
                    fontweight="bold" if accent else "normal",
                )
            else:
                ax.text(
                    est.hi + 0.022,
                    index,
                    fmt(est.value),
                    color=colour,
                    fontsize=8.4,
                    va="center",
                    ha="left",
                    fontweight="bold" if accent else "normal",
                )

    for index, run in rows:
        accent = is_accented(run)
        left.text(
            -0.075,
            index,
            panel.labels[index],
            transform=left.get_yaxis_transform(),
            color=theme.accent if accent else theme.ink,
            fontsize=9.0,
            fontweight="bold" if accent else "normal",
            va="center",
            ha="right",
        )

    add_header(
        fig,
        theme,
        "In ambiguous context, a run's bias magnitude is almost exactly the accuracy it gives up",
        panel.protocol_line
        + "\n'paper prompt' rows use the paper's English prompt, the rest the default Spanish one. "
        f"Accent: best run and the paper check. Bars are {CI_LEVEL:.0f} % intervals.",
        "Interval: percentile bootstrap, resampling unit = one dataset item, "
        f"{N_BOOT:,} replicates, seed {SEED}. In the ambiguous split the four outcomes "
        "(correct / target error / other error / unparsed) partition the run, so the item "
        "bootstrap is a multinomial resample of those counts; the bias score of each "
        "replicate uses sesgo._metrics.bias_score (paper Eq. 1).\n"
        "Check against the paper: gpt-4o-mini with the paper prompt gives 0.803 accuracy and "
        "0.199 "
        f"magnitude against {PUBLISHED_ACCURACY} and {PUBLISHED_BIAS} published in {PAPER}. "
        f"Signed bias scores, Ft, Fo and every bound are in {CSV_LINK}.",
    )
    return fig


# --------------------------------------------------------------------------------------
# chart 2: categories
# --------------------------------------------------------------------------------------


def draw_categories(panel: Panel, theme: Theme) -> Figure:
    """Heatmap of |bias| in ambiguous context, same rows and order as the headline."""
    fig = new_figure(12.0, 7.9, theme)
    ax = fig.add_axes((0.268, 0.211, 0.545, 0.615))
    ax.set_facecolor(theme.bg)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    n_rows, n_cols = len(panel.runs), len(CATEGORIES)
    values = np.full((n_rows, n_cols), np.nan)
    for row, run in enumerate(panel.runs):
        for col, (key, _) in enumerate(CATEGORIES):
            values[row, col] = abs(panel.stats[run.key][("ambig", key)].bias_magnitude.value)

    # Perceptually uniform, colour-vision safe, and shipped with matplotlib so CI
    # reproduces it byte for byte. The low end is trimmed so the weakest cells stay
    # clearly purple rather than sinking into the dark page background.
    base = plt.get_cmap("viridis")
    cmap = ListedColormap(base(np.linspace(0.14, 1.0, 256)), name="sesgo_viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)

    for row in range(n_rows):
        for col in range(n_cols):
            rgba = cmap(norm(values[row, col]))
            ax.add_patch(
                Rectangle(
                    (col, row),
                    1.0,
                    1.0,
                    facecolor=rgba,
                    edgecolor=theme.bg,
                    linewidth=1.4,
                    zorder=1,
                )
            )
            ax.text(
                col + 0.5,
                row + 0.5,
                fmt(values[row, col]),
                color=contrast_text(to_rgb(rgba[:3])),
                fontsize=8.6,
                va="center",
                ha="center",
                zorder=3,
            )

    outlined = next(i for i, (key, _) in enumerate(CATEGORIES) if key == OUTLINED_CATEGORY)
    ax.add_patch(
        Rectangle(
            (outlined - 0.02, -0.02),
            1.04,
            n_rows + 0.04,
            facecolor="none",
            edgecolor=theme.ink,
            linewidth=2.2,
            zorder=4,
        )
    )

    ax.set_xlim(-0.05, n_cols + 0.05)
    ax.set_ylim(n_rows + 0.05, -0.05)
    ax.set_xticks([])
    ax.set_yticks([])

    first = panel.runs[0]
    for col, (key, english) in enumerate(CATEGORIES):
        n_items = int(first.metrics[f"ambig/{key}/n"])
        weight = "bold" if key == OUTLINED_CATEGORY else "normal"
        ax.text(
            col + 0.5,
            -0.90,
            english,
            color=theme.ink,
            fontsize=9.6,
            fontweight=weight,
            va="bottom",
            ha="center",
        )
        ax.text(
            col + 0.5,
            -0.36,
            f"n = {n_items}",
            color=theme.muted,
            fontsize=8.6,
            va="bottom",
            ha="center",
        )

    for row, run in enumerate(panel.runs):
        accent = is_accented(run)
        ax.text(
            -0.12,
            row + 0.5,
            panel.labels[row],
            color=theme.accent if accent else theme.ink,
            fontsize=9.0,
            fontweight="bold" if accent else "normal",
            va="center",
            ha="right",
        )

    bar = fig.add_axes((0.268, 0.150, 0.28, 0.014))
    bar.imshow(
        np.linspace(0.0, 1.0, 256).reshape(1, -1),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        origin="lower",
    )
    bar.set_yticks([])
    bar.set_xticks([0, 64, 128, 192, 255])
    bar.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    bar.tick_params(colors=theme.muted, labelsize=8.2, length=2.5, width=0.7)
    for side in ("top", "right", "left", "bottom"):
        bar.spines[side].set_visible(False)
    fig.text(
        0.575,
        0.157,
        "|bias score| in ambiguous context — 0 means no directional error",
        color=theme.muted,
        fontsize=8.4,
        va="center",
        ha="left",
    )

    add_header(
        fig,
        theme,
        "Xenophobia carries the largest bias magnitude in all 14 runs",
        panel.protocol_line
        + "\nSame rows and order as the headline chart. Cells are |bias score| per "
        "category; rows marked 'paper prompt' use the prompt of the paper.",
        "The xenophobia column is marked with an outline, not a second colour, so the "
        "sequential scale keeps its single meaning. Palette: viridis (perceptually "
        "uniform, colour-vision safe), low end trimmed so the weakest cells stay legible "
        "on a dark page.\n"
        "Per-category counts are in the column headers; the split totals 1,348 items per "
        f"run. Signed bias scores, Ft, Fo, accuracy and {CI_LEVEL:.0f} % bootstrap "
        f"bounds for every cell are in {CSV_LINK}.",
    )
    return fig


# --------------------------------------------------------------------------------------
# chart 3: reasoning
# --------------------------------------------------------------------------------------


def draw_reasoning(panel: Panel, theme: Theme) -> Figure:
    """Two-panel dumbbell for the four models that have an off/low pair."""
    by_model: dict[str, dict[str, Run]] = {}
    for run in panel.runs:
        if run.base_model in REASONING_PAIRS and run.prompt_style == "clean":
            by_model.setdefault(run.base_model, {})[run.reasoning] = run
    pairs = [
        (m, by_model[m]) for m in REASONING_PAIRS if {"off", "low"} <= set(by_model.get(m, {}))
    ]
    pairs.sort(key=lambda item: -item[1]["off"].metrics["ambig/all/accuracy"])

    fig = new_figure(12.0, 5.3, theme)
    left = fig.add_axes((0.245, 0.225, 0.345, 0.550))
    right = fig.add_axes((0.640, 0.225, 0.345, 0.550))

    for ax, key, heading in (
        (left, "accuracy", "Accuracy in ambiguous context"),
        (right, "bias", "|bias score| — magnitude; lower is better"),
    ):
        style_axes(ax, theme)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(len(pairs) - 0.5, -0.65)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
        ax.set_yticks([])
        for gridline in (0.25, 0.5, 0.75, 1.0):
            ax.axvline(gridline, color=theme.grid, lw=0.7, zorder=0)
        ax.text(
            0.0,
            1.030,
            heading,
            transform=ax.transAxes,
            color=theme.ink,
            fontsize=9.5,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        for index, (_model, runs) in enumerate(pairs):
            off_cell = panel.stats[runs["off"].key][("ambig", "all")]
            low_cell = panel.stats[runs["low"].key][("ambig", "all")]
            off = off_cell.accuracy.value if key == "accuracy" else off_cell.bias_magnitude.value
            low = low_cell.accuracy.value if key == "accuracy" else low_cell.bias_magnitude.value
            ax.plot(
                [off, low],
                [index, index],
                color=theme.accent,
                lw=2.0,
                alpha=0.55,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                [off],
                [index],
                marker="o",
                markersize=6.8,
                markerfacecolor=theme.bg,
                markeredgecolor=theme.accent,
                markeredgewidth=1.8,
                linestyle="none",
                zorder=3,
            )
            ax.plot(
                [low],
                [index],
                marker="D",
                markersize=6.2,
                color=theme.accent,
                markeredgecolor=theme.bg,
                markeredgewidth=0.8,
                linestyle="none",
                zorder=3,
            )
            annotation = f"{fmt(off)} → {fmt(low)}"
            # 0.034 clears the 7.5 pt marker (about 0.026 in data units) plus padding.
            if max(off, low) > 0.78:
                ax.text(
                    min(off, low) - 0.034,
                    index,
                    annotation,
                    color=theme.ink,
                    fontsize=8.6,
                    va="center",
                    ha="right",
                )
            else:
                ax.text(
                    max(off, low) + 0.034,
                    index,
                    annotation,
                    color=theme.ink,
                    fontsize=8.6,
                    va="center",
                    ha="left",
                )

    for index, (model, runs) in enumerate(pairs):
        left.text(
            -0.028,
            index - 0.10,
            model,
            transform=left.get_yaxis_transform(),
            color=theme.ink,
            fontsize=9.6,
            va="center",
            ha="right",
        )
        left.text(
            -0.028,
            index + 0.19,
            f"{runs['low'].reasoning_tokens_per_item:,.0f} reasoning tokens / item",
            transform=left.get_yaxis_transform(),
            color=theme.muted,
            fontsize=8.4,
            va="center",
            ha="right",
        )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=7.5,
            markerfacecolor=theme.bg,
            markeredgecolor=theme.accent,
            markeredgewidth=1.8,
            linestyle="none",
            label="reasoning off",
        ),
        Line2D(
            [],
            [],
            marker="D",
            markersize=7.0,
            color=theme.accent,
            markeredgecolor=theme.bg,
            linestyle="none",
            label="reasoning low",
        ),
    ]
    # Parked to the right of the short second subtitle line: inside either panel it
    # would read as if it belonged to the row it sat next to.
    legend = fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.992, 0.906),
        frameon=False,
        ncols=2,
        fontsize=8.8,
        handletextpad=0.5,
        columnspacing=1.8,
    )
    for text in legend.get_texts():
        text.set_color(theme.ink)

    add_header(
        fig,
        theme,
        "Turning reasoning to low raises ambiguous accuracy and lowers bias magnitude in "
        "all four models",
        panel.protocol_line
        + "\nSame model, same prompt, same items; only the reasoning effort changes.",
        "The sign of the bias score is omitted here: it is sign(Ft − Fo), which flips "
        "at Ft = Fo, so when target and other errors are nearly tied the direction is not "
        "decided by the data while the magnitude is. Only the magnitude is plotted.\n"
        f"Signed bias scores, Ft, Fo and {CI_LEVEL:.0f} % bootstrap bounds "
        f"(item-level, {N_BOOT:,} replicates, seed {SEED}) are in {CSV_LINK}.",
    )
    return fig


# --------------------------------------------------------------------------------------
# csv
# --------------------------------------------------------------------------------------

CSV_HEADER = (
    "run",
    "config",
    "model",
    "prompt_style",
    "reasoning",
    "temperature",
    "tie_sign",
    "split",
    "category",
    "n",
    "n_correct",
    "n_target_error",
    "n_other_error",
    "n_unparsed",
    "n_other_outcome",
    "accuracy",
    "accuracy_ci_lo",
    "accuracy_ci_hi",
    "ft",
    "fo",
    "ft_minus_fo",
    "bias_score_signed",
    "bias_magnitude",
    "bias_magnitude_ci_lo",
    "bias_magnitude_ci_hi",
    "bias_signed_se_history",
    "sign_decided",
    "unparsed_rate",
    "interval_method",
)


def _blank(value: float, places: int = 6) -> str:
    return "" if value is None or math.isnan(value) else f"{value:.{places}f}"


def write_csv(panel: Panel, path: Path) -> None:
    """Aggregates only: one row per run x split x group. No prompt, no model output."""
    rows: list[tuple[str, ...]] = []
    for index, run in enumerate(panel.runs):
        for split in ("ambig", "disambig"):
            for group in ("all", *(key for key, _ in CATEGORIES)):
                prefix = f"{split}/{group}"
                if f"{prefix}/n" not in run.metrics:
                    continue
                cell = panel.stats[run.key].get((split, group))
                se = run.ses.get(f"{prefix}/ft_minus_fo", _NAN)
                signed = run.metrics[f"{prefix}/bias_score"]
                diff = run.metrics[f"{prefix}/ft_minus_fo"]
                n_items = int(round(run.metrics[f"{prefix}/n"]))
                if cell is None:
                    method = (
                        "none: Ft and Fo are not identified by the stored aggregates for a "
                        "category of the disambiguated split"
                    )
                    counts = ("", "", "", "", "")
                    acc_lo = acc_hi = bias_lo = bias_hi = ""
                    ft_s = fo_s = ""
                else:
                    method = (
                        f"percentile bootstrap, item-level multinomial over outcome counts, "
                        f"{N_BOOT} replicates, seed {_cell_seed(run, split, group)}"
                    )
                    g = cell.group
                    counts = (
                        str(g.correct),
                        str(g.ft),
                        str(g.fo),
                        str(g.unparsed),
                        str(g.other),
                    )
                    acc_lo = _blank(cell.accuracy.lo)
                    acc_hi = _blank(cell.accuracy.hi)
                    bias_lo = _blank(cell.bias_magnitude.lo)
                    bias_hi = _blank(cell.bias_magnitude.hi)
                    ft_s = _blank(g.ft / g.n)
                    fo_s = _blank(g.fo / g.n)
                rows.append(
                    (
                        panel.labels[index],
                        run.config,
                        run.base_model,
                        run.prompt_style,
                        run.reasoning,
                        f"{run.temperature:g}",
                        str(run.tie_sign),
                        split,
                        group,
                        str(n_items),
                        *counts,
                        _blank(run.metrics[f"{prefix}/accuracy"]),
                        acc_lo,
                        acc_hi,
                        ft_s,
                        fo_s,
                        _blank(diff),
                        _blank(signed),
                        _blank(abs(signed)),
                        bias_lo,
                        bias_hi,
                        _blank(se),
                        "yes" if (not math.isnan(se) and sign_is_decided(diff, se)) else "no",
                        _blank(run.metrics[f"{prefix}/unparsed_rate"]),
                        method,
                    )
                )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


# --------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------

CHARTS = (
    ("headline", draw_headline),
    ("categories", draw_categories),
    ("reasoning", draw_reasoning),
)


def save(fig: Figure, stem: str, theme: Theme, out_dir: Path) -> None:
    """Write one chart as a deterministic SVG plus a 2x PNG, both opaque."""
    common: dict[str, Any] = {
        "facecolor": fig.get_facecolor(),
        "edgecolor": "none",
        "transparent": False,
        "bbox_inches": None,
    }
    fig.savefig(
        out_dir / f"{stem}-{theme.name}.svg",
        format="svg",
        dpi=100,
        metadata={"Date": None},
        **common,
    )
    fig.savefig(
        out_dir / f"{stem}-{theme.name}.png",
        format="png",
        dpi=200,
        metadata={"Software": None},
        **common,
    )


def build(out_dir: Path) -> list[Path]:
    """Generate everything into `out_dir`. Returns the written paths, sorted."""
    apply_rcparams()
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = build_panel(load_runs(HISTORY))
    check_against_history(panel)
    write_csv(panel, out_dir / "results.csv")
    for stem, draw in CHARTS:
        for theme in THEMES:
            fig = draw(panel, theme)
            save(fig, stem, theme, out_dir)
            plt.close(fig)
    return sorted(p for p in out_dir.iterdir() if p.is_file())


def digests(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def check(out_dir: Path) -> int:
    """Prove determinism, then compare against the committed files.

    Strict on `*.svg` and `results.csv`. Advisory on `*.png`: PNG bytes depend on the
    zlib and libpng the platform links, so a byte difference there is not evidence of a
    stale chart. The SVG carries the same geometry and the same text, so a stale chart
    always shows up in the strict half.
    """
    with tempfile.TemporaryDirectory() as tmp:
        first, second = Path(tmp) / "a", Path(tmp) / "b"
        build(first)
        build(second)
        a, b = digests(first), digests(second)
        if a != b:
            drift = sorted(k for k in a if a[k] != b.get(k))
            print(f"FAIL non-deterministic output, two runs differ in: {drift}")
            return 1
        print(f"ok  deterministic: two independent runs agree on {len(a)} files")

        if not out_dir.exists():
            print(f"FAIL {out_dir} does not exist; run `make figures`")
            return 1
        committed = digests(out_dir)
        strict = [n for n in a if n.endswith(".svg") or n == "results.csv"]
        advisory = [n for n in a if n.endswith(".png")]

        missing = sorted(n for n in a if n not in committed)
        extra = sorted(n for n in committed if n not in a)
        stale = sorted(n for n in strict if committed.get(n) != a[n])
        if missing or extra or stale:
            for name in missing:
                print(f"FAIL missing from {out_dir}: {name}")
            for name in extra:
                print(f"FAIL not produced by make figures: {out_dir / name}")
            for name in stale:
                print(f"FAIL stale, regenerate with `make figures`: {out_dir / name}")
            return 1
        print(f"ok  committed SVG + CSV match a fresh build ({len(strict)} files, byte for byte)")

        png_drift = sorted(n for n in advisory if committed.get(n) != a[n])
        if png_drift:
            print(
                "note PNG bytes differ from the committed ones "
                f"({len(png_drift)}/{len(advisory)}); PNG encoding is not guaranteed "
                "across platforms, so this is not a failure. The SVGs matched."
            )
        else:
            print(f"ok  committed PNG also match byte for byte ({len(advisory)} files)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into temp dirs and fail if the committed SVG or CSV differ",
    )
    args = parser.parse_args(argv)
    if args.check:
        return check(args.out)
    written = build(args.out)
    for path in written:
        print(f"wrote {path.relative_to(REPO) if REPO in path.parents else path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
