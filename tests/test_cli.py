"""The CLI's trajectory round trip: `collect` writes what `coverage` reads.

`coverage`, `drift`, `mine` and `author --against` all take a pickled list
of trajectories, and `collect` is the only thing in the repository that
writes one.
"""
from __future__ import annotations

import pickle

import pytest
import yaml

from scenarios import views as _views
from scenarios.cli import main


@pytest.fixture(autouse=True)
def _restore_providers():
    # Resolving a view registers the optional domain adapters globally.
    before = list(_views._PROVIDERS)
    yield
    _views._PROVIDERS[:] = before


def test_collect_writes_trajectories_that_coverage_reads(tmp_path, capsys):
    out = tmp_path / "rollouts.pkl"
    assert main(["collect", "CartPole-v1", "--out", str(out),
                 "--episodes", "3", "--max-steps", "20"]) == 0
    trajectories = pickle.loads(out.read_bytes())
    assert len(trajectories) == 3
    for t in trajectories:
        assert len(t["features"]) == len(t["actions"]) == len(t["dones"])
        assert len(t["features"][0]) == 4

    scenario = tmp_path / "right_side.yaml"
    scenario.write_text(yaml.safe_dump({
        "name": "right_side",
        "description": "Stop drifting to the right.",
        "env_id": "CartPole-v1",
        "match_mode": "proportion",
        "match_threshold": 0.3,
        "groups": [{"name": "right", "conditions": [
            {"dim": 0, "op": ">", "value": 0.0, "target": "state"}]}],
    }))
    assert main(["coverage", str(scenario), "--against", str(out)]) == 0
    assert "Trajectories:       3" in capsys.readouterr().out
