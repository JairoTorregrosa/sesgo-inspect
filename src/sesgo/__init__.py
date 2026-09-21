"""SESGO bias evaluation for Inspect AI.

Runs the SESGO benchmark (arXiv:2509.03329, 4,156 Spanish prompts) as an Inspect task,
as infrastructure for continuous monitoring of new models. See `SPEC.md` and `specs/`.

Importing this package never reads the dataset; only calling `sesgo()` does.
"""

from sesgo._dataset import load_records, record_to_sample
from sesgo._metrics import bias_score, sesgo_metrics, summarize
from sesgo._oracle import oracle_solver
from sesgo._scorer import sesgo_scorer
from sesgo._solver import get_default_solver, sesgo_prompt
from sesgo._task import TASK_VERSION, sesgo

__all__ = [
    "TASK_VERSION",
    "bias_score",
    "get_default_solver",
    "load_records",
    "oracle_solver",
    "record_to_sample",
    "sesgo",
    "sesgo_metrics",
    "sesgo_prompt",
    "sesgo_scorer",
    "summarize",
]
