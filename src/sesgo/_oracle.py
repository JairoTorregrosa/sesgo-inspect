"""Oracle solver: a perfect (or deterministically wrong) answerer without a model call.

The oracle is a test utility and part of the public API (spec 05). It answers from the
sample metadata, so a run of the whole dataset costs nothing and has a known result:
verification check V3 (spec 07) uses it to prove that the prompt, the parser, the scorer
and the metrics agree end to end.

The chain also builds the SESGO prompt, so `inspect eval ... --solver sesgo/oracle_solver`
produces a log whose prompts look exactly like a real run.
"""

from collections.abc import Sequence
from typing import Final, cast, get_args

from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver

from sesgo._solver import sesgo_prompt
from sesgo._types import LETTERS, PROMPT_STYLES, Polarity, PromptStyle, Role

_POLARITIES: Final[tuple[Polarity, ...]] = get_args(Polarity)

_ROLES_BY_POLARITY: Final[dict[Polarity, Role]] = {"neg": "target", "nonneg": "other"}
"""Ambiguous `wrong=True` rule: the choice that always counts in Ft, never in Fo."""

_OPPOSITE: Final[dict[Role, Role]] = {"target": "other", "other": "target"}
"""Disambiguated `wrong=True` rule: the other non-`unknown` role."""


def oracle_role(label_role: Role, polarity: Polarity, wrong: bool) -> Role:
    """Return the role the oracle answers with. Pure.

    With `wrong=False` the oracle answers `label_role`, so the run is a perfect score.

    With `wrong=True` the answer is deterministic and never `unknown`:

    - Ambiguous samples (`label_role == "unknown"`) get `target` on a negative question
      and `other` on a non-negative one. Both count in Ft and never in Fo, so the run has
      accuracy 0, Ft 1, Fo 0 and `bias_score = +sqrt(2)` in every ambiguous group.
    - Disambiguated samples get the opposite non-`unknown` role. Whether that lands in Ft
      or in Fo depends on the polarity of the sample, so a disambiguated group has
      accuracy 0, `Ft + Fo = 1` and a bias score whose sign follows the dataset.

    Args:
        label_role: Role of the correct option.
        polarity: `"neg"` or `"nonneg"`.
        wrong: Answer wrongly on purpose.

    Returns:
        The role to answer with.

    Raises:
        ValueError: If the role or the polarity is unknown.
    """
    if polarity not in _POLARITIES:
        raise ValueError(f"Unknown polarity {polarity!r}, expected one of {_POLARITIES}.")
    if label_role not in _OPPOSITE and label_role != "unknown":
        raise ValueError(f"Unknown label_role {label_role!r}.")
    if not wrong:
        return label_role
    if label_role == "unknown":
        return _ROLES_BY_POLARITY[polarity]
    return _OPPOSITE[label_role]


def oracle_index(roles: Sequence[Role], label_role: Role, polarity: Polarity, wrong: bool) -> int:
    """Return the index into `roles` of the option the oracle answers with. Pure.

    Raises:
        ValueError: If `roles` does not hold exactly one option per role, or if the role
            or the polarity is unknown.
    """
    if len(roles) != 3 or {*roles} != {"target", "other", "unknown"}:
        raise ValueError(f"SESGO needs one option per role, got {list(roles)!r}.")
    return roles.index(oracle_role(label_role, polarity, wrong))


def oracle_answer(index: int, choices: Sequence[str], prompt_style: PromptStyle) -> str:
    """Render the oracle answer in the format the scorer of `prompt_style` parses. Pure.

    Returns:
        `"RESPUESTA: X"` for the clean style, `"Option N: <text>"` for the paper style.
        Both are parsed at stage `strict`.

    Raises:
        ValueError: If the style is unknown, the index is out of range, or there are not
            exactly three options.
    """
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt_style {prompt_style!r}, expected one of {PROMPT_STYLES}.")
    if len(choices) != 3:
        raise ValueError(f"SESGO needs exactly 3 choices, got {len(choices)}.")
    if not 0 <= index < 3:
        raise ValueError(f"Choice index {index!r} is out of range.")
    if prompt_style == "paper":
        return f"Option {index}: {choices[index]}"
    return f"RESPUESTA: {LETTERS[index]}"


@solver
def oracle_answer_solver(wrong: bool = False, prompt_style: PromptStyle = "clean") -> Solver:
    """Create the solver that writes the oracle answer into `state.output`.

    It is the second step of `oracle_solver` and expects the prompt to be built already.
    No model is called.

    Args:
        wrong: Answer wrongly on purpose, with the deterministic rule of `oracle_role`.
        prompt_style: `"clean"` or `"paper"`; it selects the answer format.

    Returns:
        The solver.

    Raises:
        ValueError: If `prompt_style` is unknown.
    """
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt_style {prompt_style!r}, expected one of {PROMPT_STYLES}.")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        choices = [choice.value for choice in state.choices]
        roles = cast(list[Role], state.metadata["roles"])
        label_role = cast(Role, state.metadata["label_role"])
        polarity = cast(Polarity, state.metadata["question_polarity"])
        index = oracle_index(roles, label_role, polarity, wrong)
        text = oracle_answer(index, choices, prompt_style)
        state.output = ModelOutput.from_content(model=str(state.model), content=text)
        state.messages.append(state.output.message)
        return state

    return solve


@solver
def oracle_solver(wrong: bool = False, prompt_style: PromptStyle = "clean") -> Solver:
    """Create the oracle solver chain: SESGO prompt, then the known answer.

    No model is called, so a full dataset run is free. Pass `-S wrong=true` or
    `-S prompt_style=paper` on the command line to reach the other modes.

    Args:
        wrong: Answer wrongly on purpose, with the deterministic rule of `oracle_role`.
        prompt_style: `"clean"` (D5) or `"paper"` (D12); it selects both the prompt and
            the answer format. It must match the `prompt_style` of the task, because the
            task's scorer parses the answer with the parser of its own style.

    Returns:
        The solver chain.

    Raises:
        ValueError: If `prompt_style` is unknown.
    """
    return chain(sesgo_prompt(prompt_style), oracle_answer_solver(wrong, prompt_style))
