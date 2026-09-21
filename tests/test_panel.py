"""Panel validation and unit expansion (spec 06). Pure logic, no network, no data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from sesgo._panel import (
    Panel,
    PanelError,
    Unit,
    load_panel,
    panel_path,
    parse_panel,
    select_units,
)


def entry(**changes: Any) -> dict[str, Any]:
    """A valid minimal entry, with some fields replaced."""
    base = {"name": "a", "model": "openrouter/x/a", "group": "anchor", "reasoning": "off"}
    return {**base, **changes}


MINIMAL: dict[str, Any] = {"configs": [entry()]}


def panel_of(document: dict[str, Any]) -> Panel:
    return parse_panel(document)


def test_minimal_entry_gets_every_default() -> None:
    config = panel_of(MINIMAL).configs[0]
    assert config.prompt_styles == ("clean",)
    assert config.temperature == 0.75
    assert config.max_tokens == 512
    assert config.max_connections == 16
    assert config.timeout == 600
    assert config.attempt_timeout == 180
    assert config.model_args == {}
    assert config.generate == {}
    assert config.note is None
    # Without `price_model` the price is looked up under the bare OpenRouter id.
    assert config.price_model == "x/a"


def test_price_model_can_be_overridden() -> None:
    document = {
        "configs": [
            {
                "name": "a",
                "model": "openrouter/x/a",
                "group": "modern",
                "reasoning": "off",
                "price_model": "x/other",
            }
        ]
    }
    assert panel_of(document).configs[0].price_model == "x/other"


def test_defaults_are_inherited_and_overridden_per_entry() -> None:
    document = {
        "defaults": {"temperature": 0.1, "max_connections": 32, "max_tokens": 256},
        "configs": [
            {"name": "a", "model": "o/x/a", "group": "anchor", "reasoning": "off"},
            {
                "name": "b",
                "model": "o/x/b",
                "group": "modern",
                "reasoning": "low",
                "temperature": 1.0,
                "max_tokens": None,
            },
        ],
    }
    first, second = panel_of(document).configs
    assert (first.temperature, first.max_connections, first.max_tokens) == (0.1, 32, 256)
    assert (second.temperature, second.max_connections, second.max_tokens) == (1.0, 32, None)


def test_max_tokens_and_temperature_are_lifted_out_of_generate() -> None:
    # GenerateConfig.merge skips None, so max_tokens must reach the task, not the eval.
    document = {
        "configs": [
            {
                "name": "a",
                "model": "o/x/a",
                "group": "modern",
                "reasoning": "low",
                "generate": {"reasoning_effort": "low", "max_tokens": 4096, "temperature": 0.5},
            }
        ]
    }
    config = panel_of(document).configs[0]
    assert config.max_tokens == 4096
    assert config.temperature == 0.5
    assert config.generate == {"reasoning_effort": "low"}


@pytest.mark.parametrize(
    ("document", "needle"),
    [
        # one entry: the message names the offending field
        ({"configs": [{"model": "openrouter/x/a", "group": "anchor", "reasoning": "off"}]}, "name"),
        ({"configs": [entry(name="a/b")]}, "directory name"),
        ({"configs": [entry(model="noslash")]}, "model"),
        ({"configs": [entry(group="nope")]}, "group"),
        ({"configs": [entry(reasoning="high")]}, "reasoning"),
        ({"configs": [entry(oops=1)]}, "unknown keys"),
        ({"configs": [entry(prompt_styles=["nope"])]}, "prompt styles"),
        ({"configs": [entry(temperature=9)]}, "temperature"),
        ({"configs": [entry(max_tokens=0)]}, "max_tokens"),
        ({"configs": [entry(max_connections=-1)]}, "max_connections"),
        # several entries: the message names the entry by index and by name too
        (
            {"configs": [entry(), entry(name="bad", reasoning="sideways")]},
            "configs[1] (bad).reasoning",
        ),
        # the document around the entries
        ({"configs": [entry(), entry()]}, "duplicate name"),
        ({"configs": []}, "configs"),
        ({"models": [], "configs": [entry()]}, "unknown keys"),
        ({"defaults": {"reasoning": "off"}, "configs": [entry()]}, "defaults"),
    ],
)
def test_invalid_models_yaml_names_the_entry_and_the_field(
    document: dict[str, Any], needle: str
) -> None:
    with pytest.raises(PanelError) as info:
        panel_of(document)
    assert needle in str(info.value)


def test_prompt_styles_are_deduplicated_and_ordered() -> None:
    document = {
        "configs": [
            {
                "name": "a",
                "model": "o/x/a",
                "group": "anchor",
                "reasoning": "off",
                "prompt_styles": ["paper", "clean", "paper"],
            }
        ]
    }
    assert panel_of(document).configs[0].prompt_styles == ("clean", "paper")


# --------------------------------------------------------------------------------------
# selection and units
# --------------------------------------------------------------------------------------


def two_config_panel() -> Panel:
    return panel_of(
        {
            "configs": [
                {
                    "name": "anchor-1",
                    "model": "o/x/a",
                    "group": "anchor",
                    "reasoning": "off",
                    "prompt_styles": ["clean", "paper"],
                },
                {"name": "modern-low", "model": "o/x/b", "group": "modern", "reasoning": "low"},
            ]
        }
    )


def test_select_filters_by_name_and_reasoning() -> None:
    panel = two_config_panel()
    assert [config.name for config in panel.select()] == ["anchor-1", "modern-low"]
    assert [config.name for config in panel.select(["modern-low"])] == ["modern-low"]
    assert [config.name for config in panel.select(reasoning="off")] == ["anchor-1"]
    assert [config.name for config in panel.select(["anchor-1"], "low")] == []


def test_select_accepts_comma_separated_and_repeated_names() -> None:
    panel = two_config_panel()
    names = [config.name for config in panel.select(["anchor-1,modern-low"])]
    assert names == ["anchor-1", "modern-low"]


def test_select_rejects_an_unknown_name() -> None:
    with pytest.raises(PanelError, match="unknown config names"):
        two_config_panel().select(["typo"])


def test_select_rejects_an_unknown_reasoning() -> None:
    with pytest.raises(PanelError, match="off, low or all"):
        two_config_panel().select(reasoning="medium")


def test_by_name_raises_for_an_unknown_config() -> None:
    panel = two_config_panel()
    assert panel.by_name("anchor-1").group == "anchor"
    with pytest.raises(PanelError, match="unknown config"):
        panel.by_name("nope")


def test_units_cross_configs_with_prompt_styles() -> None:
    panel = two_config_panel()
    units = select_units(panel, "pilot")
    assert [unit.name for unit in units] == [
        "anchor-1__clean",
        "anchor-1__paper",
        "modern-low__clean",
    ]
    assert [unit.key for unit in units][0] == "pilot/anchor-1__clean"
    # The stage decides how many samples, and therefore how much money, a unit costs.
    assert all(unit.limit_per_category == 50 for unit in units)
    assert all(unit.limit_per_category == 3 for unit in select_units(panel, "smoke"))
    assert all(unit.limit_per_category is None for unit in select_units(panel, "full"))


def test_unit_log_dir_layout() -> None:
    panel = two_config_panel()
    config = panel.by_name("anchor-1")
    root = Path("/logs")
    assert Unit(config, "clean", "full").log_dir(root) == root / "full" / "anchor-1__clean"
    assert Unit(config, "paper", "pilot").log_dir(root) == root / "pilot" / "anchor-1__paper"
    # The smoke stage keeps a panel/ level so it does not mix with the parser smoke logs.
    smoke_dir = root / "smoke" / "panel" / "anchor-1__clean"
    assert Unit(config, "clean", "smoke").log_dir(root) == smoke_dir


def test_unknown_stage_is_an_error() -> None:
    with pytest.raises(PanelError, match="unknown stage"):
        select_units(two_config_panel(), "huge")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# the file on disk
# --------------------------------------------------------------------------------------


def test_load_panel_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    panel = load_panel(path)
    assert panel.names() == ("a",)
    assert panel.path == path


def test_load_panel_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_panel(tmp_path / "absent.yaml")


def test_load_panel_reports_broken_yaml(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("configs: [\n", encoding="utf-8")
    with pytest.raises(PanelError, match="not valid YAML"):
        load_panel(path)


D13_MODERN: Final[tuple[str, ...]] = (
    "gpt-5.4-nano",
    "gemini-3.1-flash-lite",
    "deepseek-v4-flash",
    "qwen3.5-9b",
    "gpt-5.4-nano-low",
    "gemini-3.1-flash-lite-low",
    "deepseek-v4-flash-low",
    "qwen3.5-9b-low",
)
"""The modern half of the D13 panel of the report. Adding a model to `models.yaml` is a
supported user action (README, V12), so this test asserts that the D13 panel is *present*
and well formed, never that the file holds nothing else."""


def test_repo_models_yaml_is_valid_and_complete() -> None:
    panel = load_panel(panel_path())
    names = panel.names()
    assert "llama-3.1-8b-instruct" in names
    assert "gpt-4o-mini" in names
    # D13: the two anchors run both prompt styles, the four modern models run off and low.
    for anchor in ("llama-3.1-8b-instruct", "gpt-4o-mini"):
        assert panel.by_name(anchor).prompt_styles == ("clean", "paper")
        assert panel.by_name(anchor).group == "anchor"
    modern = [panel.by_name(name) for name in D13_MODERN]
    assert all(config.group == "modern" for config in modern)
    assert sum(1 for config in modern if config.reasoning == "off") == 4
    assert sum(1 for config in modern if config.reasoning == "low") == 4
    for config in modern:
        if config.reasoning == "off":
            assert config.model_args.get("reasoning_enabled") is False
        else:
            assert config.generate.get("reasoning_effort") == "low"
            assert config.max_tokens is not None
    assert all(config.temperature == 0.75 for config in panel.configs)
