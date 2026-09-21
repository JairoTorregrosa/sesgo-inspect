"""Solver that writes the SESGO prompt into the task state (spec 02).

The solver only builds the prompt. It never calls the model and never parses a
response; parsing lives in the scorer (spec 03).
"""

from typing import cast

from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from sesgo._prompts import PAPER_SYSTEM_MESSAGE, format_clean_prompt, format_paper_prompt
from sesgo._types import PROMPT_STYLES, Language, PromptStyle


@solver
def sesgo_prompt(prompt_style: PromptStyle = "clean") -> Solver:
    """Create the SESGO prompt solver.

    The solver reads `context`, `question` and (clean style) `language` from
    `state.metadata` and the option texts from `state.choices` in presentation
    order. It replaces `state.user_prompt.text`. The paper style also inserts the
    upstream system message at position 0.

    Args:
        prompt_style: `"clean"` (D5) or `"paper"` (D12).

    Returns:
        The solver.

    Raises:
        ValueError: If `prompt_style` is unknown.
    """
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt_style {prompt_style!r}, expected one of {PROMPT_STYLES}.")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        choices = [choice.value for choice in state.choices]
        if len(choices) != 3:
            raise ValueError(f"SESGO needs exactly 3 choices, got {len(choices)}.")
        context = cast(str, state.metadata["context"])
        question = cast(str, state.metadata["question"])
        if prompt_style == "paper":
            state.user_prompt.text = format_paper_prompt(context, question, choices)
            state.messages = [
                ChatMessageSystem(content=PAPER_SYSTEM_MESSAGE),
                *state.messages,
            ]
        else:
            language = cast(Language, state.metadata["language"])
            state.user_prompt.text = format_clean_prompt(context, question, choices, language)
        return state

    return solve


def get_default_solver(prompt_style: PromptStyle = "clean") -> list[Solver]:
    """Default solver chain of the SESGO task: `[sesgo_prompt(style), generate()]`."""
    return [sesgo_prompt(prompt_style), generate()]
