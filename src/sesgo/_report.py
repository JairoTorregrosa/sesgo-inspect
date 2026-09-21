"""`sesgo-report`: history records and the Spanish `REPORT.md` (spec 06).

The report reads `.eval` logs, never the model responses: it keeps aggregates only. No
prompt text and no completion ever reaches `results/history.jsonl` or `REPORT.md`, so
both files are safe to commit.

Everything degrades gracefully. A section whose configs did not run says so in one line
instead of failing, because the point of the monitoring package is that a partial run
still produces a readable report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.scorer import SampleScore

from sesgo._cost import (
    LedgerEntry,
    entry_spend_usd,
    read_ledger,
    results_dir,
    utc_now,
)
from sesgo._data import repo_root
from sesgo._metrics import (
    ALL_GROUP,
    SIGN_Z,
    Row,
    bootstrap_ses,
    headline_keys,
    metric_key,
    rows_from_sample_scores,
    sign_is_decided,
    summarize,
)
from sesgo._panel import Panel, PanelError, load_panel
from sesgo._paper import (
    KNOWN_INCONSISTENCIES,
    PAPER_MODEL_LABELS,
    PAPER_TIE_SIGN,
    paper_expected_metrics,
    paper_model_for_anchor,
)
from sesgo._types import CATEGORIES, SPLITS, Category, Split

__all__ = ["HistoryRecord", "build_report", "main", "write_history"]

DEFAULT_LOGS: Final[str] = "logs/full"
DEFAULT_REPORT: Final[str] = "REPORT.md"
HISTORY_NAME: Final[str] = "history.jsonl"

UNPARSED_FLAG: Final[float] = 0.02
"""Above this unparsed rate the numbers of a config are not trustworthy (spec 06)."""

SPLIT_LABELS: Final[Mapping[Split, str]] = {
    "ambig": "ambiguo",
    "disambig": "desambiguado",
}

CATEGORY_LABELS: Final[Mapping[Category, str]] = {
    "racismo": "racismo",
    "genero": "género",
    "clasismo": "clasismo",
    "xenofobia": "xenofobia",
}

_NA: Final[str] = "—"

UNSTABLE_SIGN_MARK: Final[str] = "*"
"""Marker appended to a `bias_score` cell whose sign is not decided by the data."""

UNSTABLE_SIGN_LEGEND: Final[tuple[str, ...]] = (
    f"`{UNSTABLE_SIGN_MARK}` = el **signo** de ese `bias_score` no está decidido por los "
    f"datos: |Ft − Fo| ≤ {SIGN_Z:.2f}·SE(Ft − Fo). La ecuación 1 del paper multiplica una "
    "magnitud (dominada por 1 − exactitud) por σ = signo(Ft − Fo), y ese factor salta de "
    "−1 a +1 justo en Ft = Fo. En las celdas marcadas el remuestreo cae a ambos lados del "
    "salto, así que el `±` mide la anchura del salto (≈ |bias_score|) y **no** la "
    "incertidumbre de la magnitud, que es mucho menor. Léase la magnitud; el signo es "
    "ruido. Con Ft = Fo exactos el signo es además la convención `tie_sign` (D10), no una "
    "medición.",
)


# --------------------------------------------------------------------------------------
# history records
# --------------------------------------------------------------------------------------


@dataclass
class HistoryRecord:
    """One log, reduced to aggregates. This is the input of the future web site."""

    key: str
    run_id: str
    task_id: str
    created: str
    recorded_at: str
    config: str
    model: str
    group: str
    reasoning: str
    prompt_style: str
    stage: str
    task_version: str
    dataset_version: str
    dataset_commit: str
    data_sha256: str
    parser_version: str
    refusal_patterns_version: str
    language: str
    tie_sign: int
    temperature: float | None
    max_tokens: int | None
    samples: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cost_tokens_usd: float | None
    cost_credits_usd: float | None
    log_file: str
    metrics: dict[str, float] = field(default_factory=dict[str, float])
    ses: dict[str, float] = field(default_factory=dict[str, float])

    def to_json(self) -> dict[str, Any]:
        """Serialize the record, turning `nan` into `None` so the line stays valid JSON."""
        payload: dict[str, Any] = {
            key: value for key, value in self.__dict__.items() if key not in ("metrics", "ses")
        }
        payload["metrics"] = {key: _json_float(value) for key, value in self.metrics.items()}
        payload["ses"] = {key: _json_float(value) for key, value in self.ses.items()}
        return payload

    def value(self, split: Split, group: str, name: str) -> float:
        """Read one metric, or `nan` when the run did not produce it."""
        return self.metrics.get(metric_key(split, group, name), float("nan"))

    def se(self, split: Split, group: str, name: str) -> float:
        """Read one bootstrap standard error, or `nan` when the run did not produce it."""
        return self.ses.get(metric_key(split, group, name), float("nan"))


def _json_float(value: float) -> float | None:
    return None if math.isnan(value) or math.isinf(value) else round(value, 6)


def _meta_str(metadata: Mapping[str, Any], key: str, default: str = "") -> str:
    value = metadata.get(key, default)
    return value if isinstance(value, str) else str(value)


def _config_name_from(log: EvalLog) -> tuple[str, str]:
    """Derive `(config, stage)` from the unit log directory `logs/{stage}/{name}__{style}`.

    The config name falls back to the model id when the directory does not follow the
    layout, and the stage to `""` when it is unknown.
    """
    location = Path(log.location or "")
    directory = location.parent.name
    stage = location.parent.parent.name
    if "__" in directory:
        return directory.rsplit("__", 1)[0], stage
    return log.eval.model.rsplit("/", 1)[-1], stage


def _sample_scores(log: EvalLog) -> list[SampleScore]:
    scores: list[SampleScore] = []
    for sample in log.samples or []:
        for scorer_name, score in (sample.scores or {}).items():
            scores.append(
                SampleScore(
                    score=score,
                    sample_id=sample.id,
                    sample_metadata=sample.metadata,
                    scorer=scorer_name,
                )
            )
    return scores


def rows_of(log: EvalLog) -> list[Row]:
    """Rebuild the scored rows of a log, one `Row` per scored sample."""
    return rows_from_sample_scores(_sample_scores(log))


def _ledger_for(
    ledger: Sequence[LedgerEntry], log: EvalLog, config: str, style: str
) -> LedgerEntry | None:
    name = Path(log.location or "").name
    for entry in reversed(ledger):
        if name and entry.get("log_file") == name:
            return entry
    for entry in reversed(ledger):
        # Only a finished unit describes this log's cost: an `interrupted` line (the
        # booked spend of a killed attempt, V9) carries no tokens and must not shadow it.
        if entry.get("status") != "ok":
            continue
        if entry.get("config") == config and entry.get("prompt_style") == style:
            return entry
    return None


def record_from_log(
    log: EvalLog,
    panel: Panel | None = None,
    ledger: Sequence[LedgerEntry] = (),
    n_boot: int = 1000,
) -> HistoryRecord:
    """Reduce one successful `.eval` log to a history record.

    The panel supplies `group` and `reasoning`; the ledger supplies the real cost.
    """
    metadata = log.eval.metadata or {}
    config, stage = _config_name_from(log)
    style = _meta_str(metadata, "prompt_style", "clean")
    tie_sign_raw = metadata.get("tie_sign", 1)
    tie_sign = int(tie_sign_raw) if isinstance(tie_sign_raw, int | float) else 1

    rows = rows_of(log)
    metrics = summarize(rows, tie_sign=tie_sign)
    ses = bootstrap_ses(rows, headline_keys(rows), tie_sign=tie_sign, n_boot=n_boot)

    group = reasoning = "unknown"
    if panel is not None:
        for entry in panel.configs:
            if entry.name == config:
                group, reasoning = entry.group, entry.reasoning
                break

    input_tokens = output_tokens = reasoning_tokens = total_tokens = 0
    for usage in (log.stats.model_usage or {}).values():
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        reasoning_tokens += usage.reasoning_tokens or 0
        total_tokens += usage.total_tokens

    line = _ledger_for(ledger, log, config, style)
    cost_tokens = None if line is None else line.get("cost_tokens_usd")
    cost_credits = None if line is None else line.get("cost_credits_usd")

    temperature = metadata.get("temperature")
    max_tokens = metadata.get("max_tokens")
    return HistoryRecord(
        key=f"{log.eval.run_id}/{log.eval.task_id}",
        run_id=log.eval.run_id,
        task_id=log.eval.task_id,
        created=log.eval.created,
        recorded_at=utc_now(),
        config=config,
        model=log.eval.model,
        group=group,
        reasoning=reasoning,
        prompt_style=style,
        stage=stage,
        task_version=log.eval.task_version
        if isinstance(log.eval.task_version, str)
        else str(log.eval.task_version),
        dataset_version=_meta_str(metadata, "dataset_version"),
        dataset_commit=_meta_str(metadata, "upstream_commit"),
        data_sha256=_meta_str(metadata, "data_sha256"),
        parser_version=_meta_str(metadata, "parser_version"),
        refusal_patterns_version=_meta_str(metadata, "refusal_patterns_version"),
        language=_meta_str(metadata, "language", "es"),
        tie_sign=tie_sign,
        temperature=float(temperature) if isinstance(temperature, int | float) else None,
        max_tokens=int(max_tokens) if isinstance(max_tokens, int) else None,
        samples=len(rows),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        cost_tokens_usd=None if cost_tokens is None else float(cost_tokens),
        cost_credits_usd=None if cost_credits is None else float(cost_credits),
        log_file=Path(log.location or "").name,
        metrics=metrics,
        ses=ses,
    )


def read_history(path: Path) -> list[dict[str, Any]]:
    """Read `history.jsonl`, oldest record first. A missing file is an empty history."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = cast(object, json.loads(line))
        if isinstance(value, Mapping):
            records.append(dict(cast(Mapping[str, Any], value)))
    return records


def write_history(records: Sequence[HistoryRecord], path: Path) -> tuple[int, int]:
    """Append the records that are not in the history yet.

    The key is `run_id/task_id`, so rerunning `sesgo-report` on the same logs adds
    nothing (spec 06 acceptance 4).

    Returns:
        `(added, skipped)`.
    """
    existing = read_history(path)
    known = {str(item.get("key")) for item in existing}
    added = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            if record.key in known:
                continue
            handle.write(json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
            known.add(record.key)
            added += 1
    return added, len(records) - added


def _floats(raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in cast(Mapping[str, Any], raw).items():
        if value is None:
            out[str(key)] = float("nan")
        elif isinstance(value, int | float):
            out[str(key)] = float(value)
    return out


def _text(payload: Mapping[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default


def _integer(payload: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key)
    return int(value) if isinstance(value, int | float) else default


def _optional_number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, int | float) else None


def _record_from_json(payload: Mapping[str, Any]) -> HistoryRecord:
    """Rebuild a record from one `history.jsonl` line, tolerating older schemas.

    Missing fields take neutral defaults so an old line still renders.
    """
    max_tokens = payload.get("max_tokens")
    return HistoryRecord(
        key=_text(payload, "key"),
        run_id=_text(payload, "run_id"),
        task_id=_text(payload, "task_id"),
        created=_text(payload, "created"),
        recorded_at=_text(payload, "recorded_at"),
        config=_text(payload, "config"),
        model=_text(payload, "model"),
        group=_text(payload, "group", "unknown"),
        reasoning=_text(payload, "reasoning", "unknown"),
        prompt_style=_text(payload, "prompt_style", "clean"),
        stage=_text(payload, "stage"),
        task_version=_text(payload, "task_version"),
        dataset_version=_text(payload, "dataset_version"),
        dataset_commit=_text(payload, "dataset_commit"),
        data_sha256=_text(payload, "data_sha256"),
        parser_version=_text(payload, "parser_version"),
        refusal_patterns_version=_text(payload, "refusal_patterns_version"),
        language=_text(payload, "language", "es"),
        tie_sign=_integer(payload, "tie_sign", 1),
        temperature=_optional_number(payload, "temperature"),
        max_tokens=int(max_tokens) if isinstance(max_tokens, int) else None,
        samples=_integer(payload, "samples"),
        input_tokens=_integer(payload, "input_tokens"),
        output_tokens=_integer(payload, "output_tokens"),
        reasoning_tokens=_integer(payload, "reasoning_tokens"),
        total_tokens=_integer(payload, "total_tokens"),
        cost_tokens_usd=_optional_number(payload, "cost_tokens_usd"),
        cost_credits_usd=_optional_number(payload, "cost_credits_usd"),
        log_file=_text(payload, "log_file"),
        metrics=_floats(payload.get("metrics")),
        ses=_floats(payload.get("ses")),
    )


def latest_records(history: Sequence[Mapping[str, Any]]) -> list[HistoryRecord]:
    """Keep the newest record per (config, prompt style, task version).

    Returns them in file order of first appearance.
    """
    newest: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for payload in history:
        key = (
            str(payload.get("config")),
            str(payload.get("prompt_style")),
            str(payload.get("task_version")),
        )
        previous = newest.get(key)
        if previous is None or str(payload.get("created", "")) >= str(previous.get("created", "")):
            newest[key] = payload
    return [_record_from_json(payload) for payload in newest.values()]


# --------------------------------------------------------------------------------------
# REPORT.md
# --------------------------------------------------------------------------------------


def _num(value: float, digits: int = 3) -> str:
    return _NA if math.isnan(value) else f"{value:.{digits}f}"


def _pm(value: float, se: float, digits: int = 3) -> str:
    if math.isnan(value):
        return _NA
    if math.isnan(se):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {se:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _sorted_records(records: Sequence[HistoryRecord]) -> list[HistoryRecord]:
    order = {"anchor": 0, "modern": 1, "unknown": 2}
    return sorted(
        records,
        key=lambda record: (
            order.get(record.group, 3),
            record.config,
            record.prompt_style,
        ),
    )


def _bias_cell(record: HistoryRecord, split: Split, group: str) -> str:
    """Render one `bias_score` cell, marking a sign the data do not decide.

    Returns:
        `"value ± se"`, followed by `UNSTABLE_SIGN_MARK` when `sign_is_decided` is
        false for that cell. A zero bias score has no sign to qualify and is never
        marked.
    """
    value = record.value(split, group, "bias_score")
    text = _pm(value, record.se(split, group, "bias_score"))
    if text == _NA or value == 0.0:
        return text
    decided = sign_is_decided(
        record.value(split, group, "ft_minus_fo"), record.se(split, group, "ft_minus_fo")
    )
    return text if decided else f"{text} {UNSTABLE_SIGN_MARK}"


def _section_summary(records: Sequence[HistoryRecord]) -> list[str]:
    lines = ["## 1. Resumen por split", ""]
    if not records:
        lines += ["No hay ninguna corrida todavía.", ""]
        return lines
    marked = False
    for split in SPLITS:
        present = [
            record for record in records if not math.isnan(record.value(split, ALL_GROUP, "n"))
        ]
        lines.append(f"### Contexto {SPLIT_LABELS[split]}")
        lines.append("")
        if not present:
            lines += [f"Ninguna corrida tiene muestras con contexto {SPLIT_LABELS[split]}.", ""]
            continue
        header = ["modelo", "estilo", "razonam.", "n", "exactitud", "bias_score"]
        header += [f"bias {CATEGORY_LABELS[category]}" for category in CATEGORIES]
        rows: list[list[str]] = []
        for record in _sorted_records(present):
            row = [
                record.config,
                record.prompt_style,
                record.reasoning,
                _num(record.value(split, ALL_GROUP, "n"), 0),
                _pm(
                    record.value(split, ALL_GROUP, "accuracy"),
                    record.se(split, ALL_GROUP, "accuracy"),
                ),
                _bias_cell(record, split, ALL_GROUP),
            ]
            row += [_bias_cell(record, split, category) for category in CATEGORIES]
            rows.append(row)
        lines += _table(header, rows)
        lines.append("")
        if any(UNSTABLE_SIGN_MARK in cell for row in rows for cell in row):
            marked = True
    if marked:
        lines += [*UNSTABLE_SIGN_LEGEND, ""]
    return lines


def _section_anchors(records: Sequence[HistoryRecord]) -> list[str]:
    lines = [
        "## 2. Anclas contra el paper",
        "",
        "`efecto protocolo` = estilo clean − estilo paper (nuestro). "
        "`fidelidad del port` = nuestro estilo paper − valor publicado. "
        f"Las métricas del paper se reproducen con `tie_sign={PAPER_TIE_SIGN}`; "
        "nuestras corridas usan el valor de `tie_sign` de su propio log.",
        "",
    ]
    anchors = [record for record in records if paper_model_for_anchor(record.config) is not None]
    if not anchors:
        lines += ["Todavía no hay corridas de los modelos ancla.", ""]
        return lines
    by_config: dict[str, dict[str, HistoryRecord]] = {}
    for record in anchors:
        by_config.setdefault(record.config, {})[record.prompt_style] = record
    header = [
        "modelo",
        "split",
        "métrica",
        "paper",
        "nuestro paper",
        "nuestro clean",
        "fidelidad del port",
        "efecto protocolo",
    ]
    rows: list[list[str]] = []
    for config, styles in sorted(by_config.items()):
        paper_model = paper_model_for_anchor(config)
        if paper_model is None:
            continue
        expected = paper_expected_metrics(paper_model)
        ours_paper = styles.get("paper")
        ours_clean = styles.get("clean")
        for split in SPLITS:
            for name in ("accuracy", "ft_minus_fo", "bias_score"):
                key = metric_key(split, ALL_GROUP, name)
                if key not in expected:
                    continue
                published = expected[key]
                paper_value = (
                    ours_paper.value(split, ALL_GROUP, name) if ours_paper is not None else math.nan
                )
                clean_value = (
                    ours_clean.value(split, ALL_GROUP, name) if ours_clean is not None else math.nan
                )
                rows.append(
                    [
                        f"{config} ({PAPER_MODEL_LABELS[paper_model]})",
                        SPLIT_LABELS[split],
                        name,
                        _num(published),
                        _num(paper_value),
                        _num(clean_value),
                        _num(paper_value - published),
                        _num(clean_value - paper_value),
                    ]
                )
    lines += _table(header, rows)
    lines.append("")
    missing = [
        config
        for config, styles in sorted(by_config.items())
        if "paper" not in styles or "clean" not in styles
    ]
    if missing:
        lines += [
            "Faltan estilos para: " + ", ".join(missing) + ". Las columnas vacías se "
            "llenan cuando esas unidades corran.",
            "",
        ]
    return lines


def _section_reasoning(records: Sequence[HistoryRecord]) -> list[str]:
    lines = ["## 3. Razonamiento apagado contra bajo", ""]
    pairs: dict[str, dict[str, HistoryRecord]] = {}
    for record in records:
        if record.group != "modern" or record.prompt_style != "clean":
            continue
        base = record.config.removesuffix("-low")
        pairs.setdefault(base, {})[record.reasoning] = record
    complete = {base: value for base, value in pairs.items() if len(value) == 2}
    if not complete:
        lines += [
            "Todavía no hay ningún modelo con las dos variantes (off y low) corridas.",
            "",
        ]
        if pairs:
            lines += [
                "Corridas sueltas: "
                + ", ".join(
                    f"{base} ({'/'.join(sorted(value))})" for base, value in sorted(pairs.items())
                )
                + ".",
                "",
            ]
        return lines
    header = [
        "modelo",
        "split",
        "exactitud off",
        "exactitud low",
        "Δ exactitud",
        "bias off",
        "bias low",
        "Δ bias",
        "tokens de razonamiento / muestra",
    ]
    rows: list[list[str]] = []
    unstable: list[str] = []
    for base, value in sorted(complete.items()):
        off, low = value["off"], value["low"]
        per_sample = low.reasoning_tokens / low.samples if low.samples else math.nan
        for split in SPLITS:
            if UNSTABLE_SIGN_MARK in _bias_cell(off, split, ALL_GROUP) or (
                UNSTABLE_SIGN_MARK in _bias_cell(low, split, ALL_GROUP)
            ):
                unstable.append(f"{base}/{SPLIT_LABELS[split]}")
            rows.append(
                [
                    base,
                    SPLIT_LABELS[split],
                    _num(off.value(split, ALL_GROUP, "accuracy")),
                    _num(low.value(split, ALL_GROUP, "accuracy")),
                    _num(
                        low.value(split, ALL_GROUP, "accuracy")
                        - off.value(split, ALL_GROUP, "accuracy")
                    ),
                    _num(off.value(split, ALL_GROUP, "bias_score")),
                    _num(low.value(split, ALL_GROUP, "bias_score")),
                    _num(
                        low.value(split, ALL_GROUP, "bias_score")
                        - off.value(split, ALL_GROUP, "bias_score")
                    ),
                    _num(per_sample, 1),
                ]
            )
    lines += _table(header, rows)
    lines.append("")
    if unstable:
        lines += [
            "`Δ bias` resta dos números con signo. En "
            + ", ".join(f"`{item}`" for item in sorted(set(unstable)))
            + " al menos uno de los dos `bias_score` lleva la marca `"
            + UNSTABLE_SIGN_MARK
            + "` de la sección 1, así que ese Δ puede ser sólo un cambio de signo entre "
            "dos magnitudes parecidas y no un cambio real del sesgo. Compare ahí "
            "|bias off| con |bias low|.",
            "",
        ]
    return lines


def _section_rates(records: Sequence[HistoryRecord]) -> list[str]:
    lines = [
        "## 4. Tasas de respuesta y cotas",
        "",
        "`unparsed` incluye formato inválido, rechazo y respuesta vacía; todas cuentan en "
        "el denominador y ninguna cuenta en Ft ni en Fo (D8). Las cotas "
        "`bias_score_lo`/`bias_score_hi` son el peor y el mejor caso si todas las "
        "respuestas sin parsear hubieran sido erróneas contra un grupo o contra el otro.",
        "",
    ]
    if not records:
        lines += ["Sin corridas.", ""]
        return lines
    header = [
        "modelo",
        "estilo",
        "split",
        "unparsed",
        "inválido",
        "rechazo",
        "sin respuesta",
        "truncado",
        "parser estricto",
        "bias [lo, hi]",
        "aviso",
    ]
    rows: list[list[str]] = []
    flags: list[str] = []
    for record in _sorted_records(records):
        for split in SPLITS:
            if math.isnan(record.value(split, ALL_GROUP, "n")):
                continue
            unparsed = record.value(split, ALL_GROUP, "unparsed_rate")
            low = record.value(split, ALL_GROUP, "bias_score_lo")
            high = record.value(split, ALL_GROUP, "bias_score_hi")
            warning = ""
            if not math.isnan(unparsed) and unparsed > UNPARSED_FLAG:
                warning = f"unparsed > {UNPARSED_FLAG:.0%}"
                flags.append(
                    f"{record.config}/{record.prompt_style}/{SPLIT_LABELS[split]}: "
                    f"unparsed {unparsed:.3f}"
                )
            rows.append(
                [
                    record.config,
                    record.prompt_style,
                    SPLIT_LABELS[split],
                    _num(unparsed),
                    _num(record.value(split, ALL_GROUP, "invalid_rate")),
                    _num(record.value(split, ALL_GROUP, "refusal_rate")),
                    _num(record.value(split, ALL_GROUP, "no_response_rate")),
                    _num(record.value(split, ALL_GROUP, "truncation_rate")),
                    _num(record.value(split, ALL_GROUP, "parse_strict_rate")),
                    f"[{_num(low)}, {_num(high)}]",
                    warning,
                ]
            )
    lines += _table(header, rows)
    lines.append("")
    lines += _bound_overlap_flags(records)
    if flags:
        lines += ["Avisos: " + "; ".join(flags) + ".", ""]
    return lines


def _bound_overlap_flags(records: Sequence[HistoryRecord]) -> list[str]:
    messages: list[str] = []
    for split in SPLITS:
        ranked = [
            record
            for record in records
            if record.prompt_style == "clean"
            and not math.isnan(record.value(split, ALL_GROUP, "bias_score"))
        ]
        current: Split = split
        ranked.sort(key=lambda record: record.value(current, ALL_GROUP, "bias_score"))
        for first, second in zip(ranked, ranked[1:], strict=False):
            gap = second.value(split, ALL_GROUP, "bias_score") - first.value(
                split, ALL_GROUP, "bias_score"
            )
            width = first.value(split, ALL_GROUP, "bias_score_hi") - first.value(
                split, ALL_GROUP, "bias_score_lo"
            )
            if not math.isnan(width) and not math.isnan(gap) and width > gap:
                messages.append(
                    f"{SPLIT_LABELS[split]}: la banda de `{first.config}` ({width:.3f}) es más "
                    f"ancha que su distancia a `{second.config}` ({gap:.3f}); el orden entre "
                    "los dos no está decidido por los datos."
                )
    if messages:
        return ["Orden entre modelos:", "", *[f"- {message}" for message in messages], ""]
    return []


def _section_cost(ledger: Sequence[LedgerEntry]) -> list[str]:
    lines = ["## 5. Costo real", ""]
    if not ledger:
        lines += ["El ledger está vacío.", ""]
        return lines
    header = [
        "unidad",
        "muestras",
        "tokens in/out/razonam.",
        "USD por tokens",
        "USD por créditos",
        "USD cobrado",
    ]
    rows: list[list[str]] = []
    total = 0.0
    for entry in ledger:
        spend = entry_spend_usd(entry)
        total += spend
        credits = entry.get("cost_credits_usd")
        rows.append(
            [
                str(entry.get("unit", "")),
                str(entry.get("samples", 0)),
                f"{entry.get('input_tokens', 0)}/{entry.get('output_tokens', 0)}/"
                f"{entry.get('reasoning_tokens', 0)}",
                f"{float(entry.get('cost_tokens_usd') or 0.0):.6f}",
                _NA if credits is None else f"{float(credits):.6f}",
                f"{spend:.6f}",
            ]
        )
    lines += _table(header, rows)
    lines += [
        "",
        f"**Total cobrado: USD {total:.4f}.** Por línea se toma el mayor entre el costo "
        "por tokens (a precio de lista de OpenRouter) y el delta de créditos, porque el "
        "endpoint de créditos se actualiza con retraso y nunca debemos subestimar el gasto.",
        "",
    ]
    return lines


def _section_design(records: Sequence[HistoryRecord]) -> list[str]:
    versions = sorted(
        {(record.task_version, record.dataset_version, record.parser_version) for record in records}
    )
    lines = [
        "## 6. Diferencias de diseño con el paper",
        "",
        "- **Prompt limpio en español** (D5) con línea final `RESPUESTA: X`, frente al "
        "prompt del paper, que es un system message en inglés con opciones numeradas "
        "(D12). El efecto medido está en la sección 2, columna *efecto protocolo*.",
        "- **Orden de opciones barajado** por muestra con semilla derivada del id (D4). "
        "El estilo paper no baraja, porque el prompt upstream numera las opciones.",
        "- **Parser determinista de dos etapas** (D6), sin extractor por LLM. La tasa de "
        "etapa estricta está en la sección 4.",
        "- **Enrutamiento D8**: vacío, rechazo y formato inválido se cuentan como "
        "incorrectos y permanecen en el denominador, pero nunca en Ft ni en Fo.",
        "- **`tie_sign = +1`** cuando Ft = Fo (D10). El paper usa `np.sign`, es decir 0; "
        "por eso las comparaciones contra el paper se recalculan con `tie_sign=0`.",
        "- **Razonamiento**: los modelos modernos se corren con el razonamiento apagado y "
        "con `reasoning_effort=low`; el paper no tenía esta variable.",
        "",
        "Inconsistencias conocidas del paper (documentadas, no forzadas):",
        "",
    ]
    lines += [f"- {item}" for item in KNOWN_INCONSISTENCIES]
    lines.append("")
    if versions:
        lines += [
            "Versiones presentes en este reporte: "
            + ", ".join(
                f"task {task}, datos {dataset}, parser {parser}"
                for task, dataset, parser in versions
            )
            + ".",
            "",
            "Problema conocido del parser 3 (corregido en el parser 4, para corridas nuevas): "
            "un rechazo que cita el texto de una sola opción se contaba como respuesta. La "
            "auditoría "
            "V10 lo encontró en 1 de 49,872 muestras reales (`llama-3.1-8b-instruct`, clean, "
            "`clasismo-es-0008`); efecto sobre bias_score: +0.00005.",
            "",
        ]
    return lines


def _section_commands(logs_dir: Path) -> list[str]:
    return [
        "## 7. Comandos",
        "",
        "```bash",
        f"uv run inspect view --log-dir {logs_dir}   # abrir las trazas",
        "uv run sesgo-run estimate --stage full --reasoning all",
        "uv run sesgo-run smoke                     # V4: 3 muestras por categoría, todo el panel",
        "uv run sesgo-run pilot --only gpt-4o-mini",
        "uv run sesgo-run ledger",
        f"uv run sesgo-report --logs {logs_dir}",
        "```",
        "",
        "Para añadir un modelo basta una entrada en `models.yaml`:",
        "",
        "```yaml",
        "  - name: mi-modelo",
        "    model: openrouter/proveedor/mi-modelo",
        "    group: modern",
        '    reasoning: "off"',
        "    model_args: {reasoning_enabled: false}",
        "```",
        "",
        "Después: `uv run sesgo-run smoke --only mi-modelo` (comprueba que el "
        "razonamiento queda realmente en 0 tokens), `uv run sesgo-run pilot --only "
        "mi-modelo`, `uv run sesgo-run full --only mi-modelo` y `uv run sesgo-report`.",
        "",
    ]


def build_report(
    records: Sequence[HistoryRecord],
    ledger: Sequence[LedgerEntry],
    logs_dir: Path,
    notes: Sequence[str] = (),
) -> str:
    """Render `REPORT.md` in Spanish.

    Args:
        records: The newest record per (config, style, task version).
        ledger: Ledger lines.
        logs_dir: Directory the logs were read from, quoted in the commands section.
        notes: Extra notes, for example a config that cannot switch reasoning off.

    Returns:
        The Markdown text. It contains no prompt and no model response.
    """
    configs = sorted({record.config for record in records})
    lines = [
        "# SESGO — reporte de monitoreo",
        "",
        f"Generado el {utc_now()} a partir de `{logs_dir}`. "
        f"{len(records)} unidad(es), {len(configs)} configuración(es): "
        + (", ".join(configs) if configs else "ninguna")
        + ".",
        "",
        "Este archivo solo contiene agregados. No incluye ningún prompt ni ninguna "
        "respuesta de modelo; para eso están las trazas (`inspect view`).",
        "",
    ]
    if notes:
        lines += ["Notas del panel:", "", *[f"- {note}" for note in notes], ""]
    lines += _section_summary(records)
    lines += _section_anchors(records)
    lines += _section_reasoning(records)
    lines += _section_rates(records)
    lines += _section_cost(ledger)
    lines += _section_design(records)
    lines += _section_commands(logs_dir)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------


def _load_logs(logs_dir: Path) -> tuple[list[EvalLog], list[str]]:
    infos = list_eval_logs(str(logs_dir), formats=["eval"], recursive=True)
    good: list[EvalLog] = []
    skipped: list[str] = []
    for info in infos:
        log = read_eval_log(info)
        if log.status != "success" or not log.samples:
            skipped.append(f"{Path(info.name).name} (status={log.status})")
            continue
        good.append(log)
    return good, skipped


def _panel_notes(panel: Panel | None) -> list[str]:
    if panel is None:
        return []
    return [f"`{config.name}`: {config.note}" for config in panel.configs if config.note]


def run_report(
    logs_dir: Path,
    history_file: Path,
    out_file: Path,
    panel_file: Path | None = None,
    n_boot: int = 1000,
) -> int:
    """Build history records from the `.eval` logs and write the report. Exit code."""
    panel: Panel | None = None
    try:
        panel = load_panel(panel_file)
    except (FileNotFoundError, PanelError) as exc:
        print(f"warning: models.yaml not usable ({exc}); group and reasoning stay unknown")

    logs, skipped = _load_logs(logs_dir)
    print(f"read {len(logs)} successful log(s) from {logs_dir}")
    for name in skipped:
        print(f"  skipped {name}")

    ledger = read_ledger()
    records = [record_from_log(log, panel, ledger, n_boot=n_boot) for log in logs]
    added, duplicate = write_history(records, history_file)
    print(f"history: +{added} new, {duplicate} already present -> {history_file}")

    report = build_report(
        latest_records(read_history(history_file)), ledger, logs_dir, _panel_notes(panel)
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report, encoding="utf-8")
    print(f"report written to {out_file}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sesgo-report",
        description="Build results/history.jsonl and a Spanish REPORT.md from .eval logs.",
    )
    parser.add_argument("--logs", type=Path, default=None, help=f"log directory ({DEFAULT_LOGS})")
    parser.add_argument("--history", type=Path, default=None, help="path of history.jsonl")
    parser.add_argument("--out", type=Path, default=None, help=f"output file ({DEFAULT_REPORT})")
    parser.add_argument("--panel", type=Path, default=None, help="path of models.yaml")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of `sesgo-report`. Returns an exit code."""
    args = _parser().parse_args(list(argv) if argv is not None else None)
    root = repo_root()
    logs_dir = cast(Path | None, args.logs) or root / DEFAULT_LOGS
    history = cast(Path | None, args.history) or results_dir(root) / HISTORY_NAME
    out_file = cast(Path | None, args.out) or root / DEFAULT_REPORT
    if not logs_dir.exists():
        print(f"error: log directory {logs_dir} does not exist", file=sys.stderr)
        return 1
    return run_report(logs_dir, history, out_file, cast(Path | None, args.panel), int(args.n_boot))


if __name__ == "__main__":
    raise SystemExit(main())
