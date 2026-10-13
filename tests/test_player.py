"""App simulation wrapper and player payload."""

import base64
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import player  # noqa: E402


def test_simulate_matches_the_library_fire():
    s = player.Scenario(size=80, density=0.62, q_normal=1.0, dryness=0.0)
    r = player.simulate(s)
    from forest_fire.experiments import fire_metrics
    from forest_fire.grid import make_forest
    m = fire_metrics(make_forest(80, 0.62, np.random.default_rng(42)), q=1.0, seed=42)
    assert r["fraction_burned"] == m["fraction_burned"] and r["reached_edge"] == m["spanned"]


def test_same_forest_with_and_without_management():
    s = player.Scenario(size=80, density=0.7, strategy="firebreaks", budget=0.1, ignition="centre")
    treated, untreated = player.simulate(s), player.simulate(s, "none")
    assert np.array_equal(treated["forest"], untreated["forest"]) and treated["removed"] > 0.05
    assert treated["strike"] == untreated["strike"]


def test_payload_round_trip():
    s = replace(player.Scenario(size=60), burn_steps=3)
    r = player.simulate(s)
    html = player.build_html([(r, "a", "</script>")], s)
    p = json.loads(re.search(r"const P = (.*?);\n", html).group(1).replace("<\\/", "</"))
    arrival = np.frombuffer(base64.b64decode(p["runs"][0]["arrival_b64"]), "<i4").reshape(60, 60)
    assert np.array_equal(arrival, r["arrival"]) and p["t_max"] == r["duration"] + 3
    assert p["runs"][0]["tag"] == "</script>"
