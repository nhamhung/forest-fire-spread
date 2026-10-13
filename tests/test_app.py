"""Streamlit app: loads, presets work, each ignition and management option runs."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def test_loads_and_runs_default():
    at = app()
    assert not at.exception
    assert any("trees burned" in m.label for m in at.metric)


@pytest.mark.parametrize("preset", range(4))
def test_presets(preset):
    at = app()
    at.button(key=f"preset_{preset}").click().run()
    assert not at.exception


@pytest.mark.parametrize("ignition", ["left_edge", "centre", "lightning"])
def test_ignitions(ignition):
    at = app()
    at.radio(key="ignition").set_value(ignition).run()
    assert not at.exception


@pytest.mark.parametrize("strategy", ["firebreaks", "thinning", "controlled_burns"])
def test_management_with_comparison(strategy):
    at = app()
    at.selectbox(key="strategy").set_value(strategy).run()
    at.checkbox(key="compare").check().run()
    assert not at.exception
    assert len([m for m in at.metric if "trees burned" in m.label]) == 2


def test_firebreak_preset_beats_no_management():
    at = app()
    at.button(key="preset_3").click().run()                 # drought + firebreaks, compared
    burned = [float(m.value.rstrip("%")) for m in at.metric if "trees burned" in m.label]
    assert burned[0] < burned[1]
