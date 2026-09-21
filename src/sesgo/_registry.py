"""Inspect AI registry entry point. Imports only.

`pyproject.toml` points the `inspect_ai` entry point at this module, so importing it is
what makes `sesgo/sesgo`, `sesgo/sesgo_prompt`, `sesgo/oracle_solver`, `sesgo/sesgo_scorer`
and `sesgo/sesgo_metrics` resolvable by name on the command line.
"""

# ruff: noqa: F401

from sesgo._metrics import sesgo_metrics
from sesgo._oracle import oracle_solver
from sesgo._scorer import sesgo_scorer
from sesgo._solver import sesgo_prompt
from sesgo._task import sesgo

__all__ = ["oracle_solver", "sesgo", "sesgo_metrics", "sesgo_prompt", "sesgo_scorer"]
