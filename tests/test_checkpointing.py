"""Resumable checkpoints (full learner-state round-trip)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from agents import PPOAgent
from utils.checkpointing import (
    CHECKPOINT_FORMAT,
    baseline_checkpoint,
    build_checkpoint,
    find_latest_checkpoint,
    is_readable_checkpoint,
    is_resumable,
    save_checkpoint_atomic,
    load_for_resume,
    resolve_resume,
)


class _RMS:
    """Stand-in with the same duck-type as RunningRMS / RunningMeanStd."""

    def __init__(self, mean=0.0, var=1.0, count=1e-4):
        self.mean, self.var, self.count = mean, var, count


def _agent(**kw):
    return PPOAgent(observation_dim=6, action_dim=3, action_type="discrete",
                    device="cpu", **kw)


def _rollout(n=64, obs_dim=6, n_actions=3, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "actions": rng.integers(0, n_actions, n),
        "log_probs": rng.standard_normal(n).astype(np.float32),
        "values": rng.standard_normal(n).astype(np.float32),
        "advantages": rng.standard_normal(n).astype(np.float32),
        "returns": rng.standard_normal(n).astype(np.float32),
    }


class TestPayload:
    def test_contains_everything_needed_to_continue(self):
        p = build_checkpoint(_agent(), 1234, best_eval=0.5, ret_rms=_RMS())
        for key in ("network", "optimizer", "global_step", "best_eval",
                    "ret_rms", "rng", "sde_step", "format"):
            assert key in p, key
        assert p["format"] == CHECKPOINT_FORMAT

    def test_records_that_env_state_is_not_restored(self):
        assert build_checkpoint(_agent(), 0)["env_state_restored"] is False

    def test_extra_fields_pass_through(self):
        p = build_checkpoint(_agent(), 0, extra={"note": "hi"})
        assert p["note"] == "hi"


class TestResumability:
    def test_format_1_is_not_resumable(self):
        assert not is_resumable({"network": {}})

    def test_payload_without_optimizer_is_not_resumable(self):
        assert not is_resumable({"format": 2, "global_step": 1})

    def test_new_payload_is_resumable(self):
        assert is_resumable(build_checkpoint(_agent(), 0))

    def test_loading_a_network_only_checkpoint_raises(self, tmp_path):
        p = tmp_path / "old.pt"
        torch.save({"network": _agent().network.state_dict()}, p)
        with pytest.raises(ValueError, match="cannot be resumed"):
            load_for_resume(p, _agent())


class TestRoundTrip:
    def test_weights_restored_exactly(self, tmp_path):
        src = _agent()
        src.update(_rollout())
        p = tmp_path / "c.pt"
        torch.save(build_checkpoint(src, 500), p)

        dst = _agent()
        load_for_resume(p, dst)
        for a, b in zip(src.network.parameters(), dst.network.parameters()):
            assert torch.equal(a, b)

    def test_optimizer_moments_restored(self, tmp_path):
        src = _agent()
        src.update(_rollout())
        p = tmp_path / "c.pt"
        torch.save(build_checkpoint(src, 500), p)

        dst = _agent()
        state = load_for_resume(p, dst)
        assert state.restored_optimizer
        s_src = list(src.optimizer.state_dict()["state"].values())[0]
        s_dst = list(dst.optimizer.state_dict()["state"].values())[0]
        assert torch.equal(s_src["exp_avg"], s_dst["exp_avg"])
        assert torch.equal(s_src["exp_avg_sq"], s_dst["exp_avg_sq"])
        assert s_src["step"] == s_dst["step"]

    def test_fresh_agent_has_no_moments(self):
        """Guards the test above from passing trivially."""
        assert _agent().optimizer.state_dict()["state"] == {}

    def test_step_and_best_eval_restored(self, tmp_path):
        p = tmp_path / "c.pt"
        torch.save(build_checkpoint(_agent(), 4242, best_eval=3.5), p)
        state = load_for_resume(p, _agent())
        assert state.global_step == 4242
        assert state.best_eval == 3.5

    def test_normaliser_restored(self, tmp_path):
        p = tmp_path / "c.pt"
        torch.save(build_checkpoint(_agent(), 0,
                                    ret_rms=_RMS(2.5, 7.25, 999.0)), p)
        rms = _RMS()
        load_for_resume(p, _agent(), ret_rms=rms)
        assert rms.mean == pytest.approx(2.5)
        assert rms.var == pytest.approx(7.25)
        assert rms.count == pytest.approx(999.0)

    def test_rng_streams_restored(self, tmp_path):
        p = tmp_path / "c.pt"
        torch.save(build_checkpoint(_agent(), 0), p)
        expected = (random.random(), float(np.random.rand()),
                    float(torch.rand(1)))

        random.random(); np.random.rand(); torch.rand(1)   # advance them
        state = load_for_resume(p, _agent())
        assert state.restored_rng
        got = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        assert got == pytest.approx(expected)


class TestDiscovery:
    def _touch(self, d, step):
        p = Path(d) / f"checkpoint_step{step}.pt"
        torch.save(build_checkpoint(_agent(), step), p)
        return p

    def test_picks_highest_step_not_newest_mtime(self, tmp_path):
        self._touch(tmp_path, 1000)
        older = self._touch(tmp_path, 200)
        import os, time
        os.utime(older, (time.time() + 500, time.time() + 500))  # newer mtime
        assert find_latest_checkpoint(tmp_path).name == "checkpoint_step1000.pt"

    def test_none_when_empty(self, tmp_path):
        assert find_latest_checkpoint(tmp_path) is None

    def test_none_for_missing_directory(self, tmp_path):
        assert find_latest_checkpoint(tmp_path / "nope") is None

    def test_ignores_non_numeric_names(self, tmp_path):
        (tmp_path / "checkpoint_stepBEST.pt").write_bytes(b"x")
        assert find_latest_checkpoint(tmp_path) is None


class TestResolveResume:
    @pytest.mark.parametrize("value", [False, None, "false", "never", "no", ""])
    def test_disabled_values(self, value, tmp_path):
        assert resolve_resume(value, tmp_path) is None

    @pytest.mark.parametrize("value", [True, "true", "auto", "latest"])
    def test_auto_finds_latest(self, value, tmp_path):
        torch.save(build_checkpoint(_agent(), 10),
                   tmp_path / "checkpoint_step10.pt")
        assert resolve_resume(value, tmp_path).name == "checkpoint_step10.pt"

    def test_auto_with_no_checkpoints_is_none(self, tmp_path):
        assert resolve_resume("auto", tmp_path) is None

    def test_explicit_path(self, tmp_path):
        p = tmp_path / "mine.pt"
        torch.save(build_checkpoint(_agent(), 1), p)
        assert resolve_resume(str(p), tmp_path) == p

    def test_missing_explicit_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_resume(str(tmp_path / "ghost.pt"), tmp_path)


class TestCrashSafety:
    """A machine dying mid-save must not poison checkpoint discovery."""

    def _good(self, d, step):
        p = Path(d) / f"checkpoint_step{step}.pt"
        save_checkpoint_atomic(build_checkpoint(_agent(), step), p)
        return p

    def test_atomic_save_leaves_no_temp_files(self, tmp_path):
        self._good(tmp_path, 100)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_atomic_save_round_trips(self, tmp_path):
        p = self._good(tmp_path, 100)
        assert load_for_resume(p, _agent()).global_step == 100

    def test_zero_byte_checkpoint_is_not_readable(self, tmp_path):
        p = tmp_path / "checkpoint_step999.pt"
        p.touch()
        assert not is_readable_checkpoint(p)

    def test_truncated_checkpoint_is_not_readable(self, tmp_path):
        p = self._good(tmp_path, 100)
        data = p.read_bytes()
        p.write_bytes(data[: len(data) // 3])
        assert not is_readable_checkpoint(p)

    def test_discovery_skips_zero_byte_head(self, tmp_path):
        """Exactly the 2026-09-16 crash: newest file created, never written."""
        self._good(tmp_path, 3112960)
        (tmp_path / "checkpoint_step3178496.pt").touch()
        found = find_latest_checkpoint(tmp_path)
        assert found.name == "checkpoint_step3112960.pt"

    def test_discovery_skips_multiple_bad_heads(self, tmp_path):
        self._good(tmp_path, 100)
        for step in (200, 300):
            (tmp_path / f"checkpoint_step{step}.pt").write_bytes(b"junk")
        assert find_latest_checkpoint(tmp_path).name == "checkpoint_step100.pt"

    def test_none_when_every_candidate_is_corrupt(self, tmp_path):
        (tmp_path / "checkpoint_step10.pt").touch()
        assert find_latest_checkpoint(tmp_path) is None

    def test_verify_false_returns_highest_step_unchecked(self, tmp_path):
        self._good(tmp_path, 100)
        (tmp_path / "checkpoint_step200.pt").touch()
        got = find_latest_checkpoint(tmp_path, verify=False)
        assert got.name == "checkpoint_step200.pt"

    def test_format_1_head_is_skipped(self, tmp_path):
        self._good(tmp_path, 100)
        torch.save({"network": _agent().network.state_dict()},
                   tmp_path / "checkpoint_step200.pt")
        assert find_latest_checkpoint(tmp_path).name == "checkpoint_step100.pt"


class TestBaselineCheckpoint:
    def test_final_checkpoint_by_default(self):
        assert baseline_checkpoint("fourrooms", 43) == Path(
            "experiments/checkpoints/fourrooms_ppo_seed43/final_model.pt")
        assert baseline_checkpoint("cartpole", 42).name == "final_model.pt"

    def test_collapsed_run_uses_its_best_checkpoint(self):
        assert baseline_checkpoint("fourrooms", 42).name == "best_model.pt"
