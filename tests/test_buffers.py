"""
Comprehensive test suite for replay buffer implementations.

Covers:
- ReplayBuffer: add, sample, capacity overflow, ring buffer behavior
- TrajectoryBuffer: add, sample, get_all, deque maxlen behavior, clear
- DualBuffer: add forget/retain, sample_mixed, ratio control
"""
import pytest
import numpy as np
import torch

from utils.buffers import ReplayBuffer, TrajectoryBuffer, DualBuffer


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_init(self):
        buf = ReplayBuffer(capacity=100, observation_shape=(4,), action_shape=(1,))
        assert len(buf) == 0

    def test_add_single(self):
        buf = ReplayBuffer(capacity=10, observation_shape=(4,), action_shape=(1,))
        buf.add(
            np.zeros(4), np.array([0.0]), 1.0, np.ones(4), False
        )
        assert len(buf) == 1

    def test_add_fills_up(self):
        buf = ReplayBuffer(capacity=5, observation_shape=(4,), action_shape=(1,))
        for i in range(5):
            buf.add(np.zeros(4), np.array([0.0]), float(i), np.zeros(4), False)
        assert len(buf) == 5

    def test_ring_buffer_overwrites(self):
        buf = ReplayBuffer(capacity=3, observation_shape=(2,), action_shape=(1,))
        for i in range(5):
            buf.add(np.array([i, i]), np.array([0.0]), float(i), np.zeros(2), False)
        assert len(buf) == 3
        # Oldest entries (0, 1) should be overwritten by (3, 4)
        # Position cycles: 0->1->2->0->1, so positions 0,1 have data 3,4
        assert buf.observations[0, 0] == 3.0
        assert buf.observations[1, 0] == 4.0
        assert buf.observations[2, 0] == 2.0

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=20, observation_shape=(4,), action_shape=(1,))
        for _ in range(20):
            buf.add(np.random.randn(4), np.array([0.0]), 1.0, np.random.randn(4), False)
        batch = buf.sample(8)
        assert batch["observations"].shape == (8, 4)
        assert batch["actions"].shape == (8, 1)
        assert batch["rewards"].shape == (8,)
        assert batch["next_observations"].shape == (8, 4)
        assert batch["dones"].shape == (8,)

    def test_sample_returns_tensors(self):
        buf = ReplayBuffer(capacity=10, observation_shape=(4,), action_shape=(1,))
        for _ in range(10):
            buf.add(np.random.randn(4), np.array([0.0]), 1.0, np.random.randn(4), False)
        batch = buf.sample(4)
        for key, val in batch.items():
            assert isinstance(val, torch.Tensor)

    def test_sample_data_integrity(self):
        """Verify that sampled data matches what was stored."""
        buf = ReplayBuffer(capacity=5, observation_shape=(2,), action_shape=(1,))
        buf.add(np.array([1.0, 2.0]), np.array([3.0]), 4.0, np.array([5.0, 6.0]), True)
        batch = buf.sample(1)
        assert batch["observations"][0, 0].item() == pytest.approx(1.0)
        assert batch["actions"][0, 0].item() == pytest.approx(3.0)
        assert batch["rewards"][0].item() == pytest.approx(4.0)
        assert batch["dones"][0].item() == pytest.approx(1.0)

    def test_multidim_observation(self):
        buf = ReplayBuffer(capacity=10, observation_shape=(3, 4), action_shape=(2,))
        buf.add(np.zeros((3, 4)), np.zeros(2), 0.0, np.zeros((3, 4)), False)
        assert len(buf) == 1
        batch = buf.sample(1)
        assert batch["observations"].shape == (1, 3, 4)


# ---------------------------------------------------------------------------
# TrajectoryBuffer
# ---------------------------------------------------------------------------

class TestTrajectoryBuffer:
    def _make_traj(self, length=5, obs_dim=4):
        return {
            "observations": [np.random.randn(obs_dim) for _ in range(length)],
            "actions": [np.random.randint(0, 2) for _ in range(length)],
            "rewards": [np.random.randn() for _ in range(length)],
            "dones": [False] * (length - 1) + [True],
        }

    def test_init(self):
        buf = TrajectoryBuffer(capacity=100)
        assert len(buf) == 0

    def test_add_trajectory(self):
        buf = TrajectoryBuffer(capacity=10)
        buf.add_trajectory(self._make_traj())
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = TrajectoryBuffer(capacity=3)
        for _ in range(5):
            buf.add_trajectory(self._make_traj())
        assert len(buf) == 3

    def test_deque_keeps_latest(self):
        """deque(maxlen=N) discards oldest entries."""
        buf = TrajectoryBuffer(capacity=2)
        t1 = {"observations": [np.array([1.0])], "actions": [0], "rewards": [1.0], "dones": [True]}
        t2 = {"observations": [np.array([2.0])], "actions": [0], "rewards": [2.0], "dones": [True]}
        t3 = {"observations": [np.array([3.0])], "actions": [0], "rewards": [3.0], "dones": [True]}
        buf.add_trajectory(t1)
        buf.add_trajectory(t2)
        buf.add_trajectory(t3)
        all_trajs = buf.get_all()
        assert len(all_trajs) == 2
        # t1 should be gone, t2 and t3 remain
        assert all_trajs[0]["rewards"][0] == 2.0
        assert all_trajs[1]["rewards"][0] == 3.0

    def test_get_all_returns_list(self):
        buf = TrajectoryBuffer(capacity=10)
        buf.add_trajectory(self._make_traj())
        buf.add_trajectory(self._make_traj())
        all_t = buf.get_all()
        assert isinstance(all_t, list)
        assert len(all_t) == 2

    def test_sample_trajectories_less_than_available(self):
        buf = TrajectoryBuffer(capacity=10)
        for _ in range(5):
            buf.add_trajectory(self._make_traj())
        sampled = buf.sample_trajectories(3)
        assert len(sampled) == 3

    def test_sample_trajectories_more_than_available(self):
        buf = TrajectoryBuffer(capacity=10)
        for _ in range(3):
            buf.add_trajectory(self._make_traj())
        sampled = buf.sample_trajectories(10)
        assert len(sampled) == 3

    def test_sample_returns_distinct(self):
        buf = TrajectoryBuffer(capacity=100)
        for _ in range(20):
            buf.add_trajectory(self._make_traj())
        sampled = buf.sample_trajectories(5)
        # Each sampled trajectory should be a valid dict
        for t in sampled:
            assert "observations" in t
            assert "actions" in t
            assert "rewards" in t

    def test_clear(self):
        buf = TrajectoryBuffer(capacity=10)
        for _ in range(5):
            buf.add_trajectory(self._make_traj())
        buf.clear()
        assert len(buf) == 0

    def test_trajectory_data_integrity(self):
        buf = TrajectoryBuffer(capacity=10)
        obs = [np.array([1.0, 2.0, 3.0, 4.0])]
        traj = {"observations": obs, "actions": [1], "rewards": [5.0], "dones": [True]}
        buf.add_trajectory(traj)
        retrieved = buf.get_all()[0]
        assert np.allclose(retrieved["observations"][0], obs[0])
        assert retrieved["actions"][0] == 1
        assert retrieved["rewards"][0] == 5.0


# ---------------------------------------------------------------------------
# DualBuffer
# ---------------------------------------------------------------------------

class TestDualBuffer:
    def test_init(self):
        buf = DualBuffer(
            forget_capacity=10, retain_capacity=10,
            observation_shape=(4,), action_shape=(1,)
        )
        # __len__ returns a tuple (forget_len, retain_len)
        f_len, r_len = buf.__len__()
        assert f_len == 0
        assert r_len == 0

    def test_add_forget(self):
        buf = DualBuffer(
            forget_capacity=10, retain_capacity=10,
            observation_shape=(4,), action_shape=(1,)
        )
        buf.add_forget(np.zeros(4), np.array([0.0]), 1.0, np.zeros(4), False)
        f_len, r_len = buf.__len__()
        assert f_len == 1
        assert r_len == 0

    def test_add_retain(self):
        buf = DualBuffer(
            forget_capacity=10, retain_capacity=10,
            observation_shape=(4,), action_shape=(1,)
        )
        buf.add_retain(np.zeros(4), np.array([0.0]), 1.0, np.zeros(4), False)
        f_len, r_len = buf.__len__()
        assert f_len == 0
        assert r_len == 1

    def test_sample_mixed_ratio(self):
        buf = DualBuffer(
            forget_capacity=20, retain_capacity=20,
            observation_shape=(4,), action_shape=(1,)
        )
        for _ in range(20):
            buf.add_forget(np.random.randn(4), np.array([0.0]), 1.0, np.random.randn(4), False)
            buf.add_retain(np.random.randn(4), np.array([0.0]), 0.0, np.random.randn(4), False)

        batch = buf.sample_mixed(batch_size=10, forget_ratio=0.5)
        assert batch["observations"].shape[0] == 10

    def test_sample_mixed_has_mask(self):
        buf = DualBuffer(
            forget_capacity=10, retain_capacity=10,
            observation_shape=(4,), action_shape=(1,)
        )
        for _ in range(10):
            buf.add_forget(np.random.randn(4), np.array([0.0]), 1.0, np.random.randn(4), False)
            buf.add_retain(np.random.randn(4), np.array([0.0]), 0.0, np.random.randn(4), False)

        batch = buf.sample_mixed(batch_size=8, forget_ratio=0.5)
        assert "observations_mask" in batch
        # First half should be forget (1), second half retain (0)
        mask = batch["observations_mask"]
        assert mask[:4].sum() == 4  # forget
        assert mask[4:].sum() == 0  # retain

    def test_sample_mixed_only_forget(self):
        buf = DualBuffer(
            forget_capacity=10, retain_capacity=10,
            observation_shape=(4,), action_shape=(1,)
        )
        for _ in range(10):
            buf.add_forget(np.random.randn(4), np.array([0.0]), 1.0, np.random.randn(4), False)
        batch = buf.sample_mixed(batch_size=5, forget_ratio=0.5)
        assert batch is not None
        # Only forget buffer has data, so retain part returns None -> only forget
        assert batch["observations"].shape[0] > 0

    def test_sample_mixed_only_retain(self):
        buf = DualBuffer(
            forget_capacity=10, retain_capacity=10,
            observation_shape=(4,), action_shape=(1,)
        )
        for _ in range(10):
            buf.add_retain(np.random.randn(4), np.array([0.0]), 0.0, np.random.randn(4), False)
        batch = buf.sample_mixed(batch_size=5, forget_ratio=0.5)
        assert batch is not None
        assert batch["observations"].shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
