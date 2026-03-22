"""
Comprehensive test suite for utility functions and helpers.

Covers:
- set_seed: reproducibility of random, numpy, torch
- get_device: auto, cpu, cuda
- create_directories: directory creation
- setup_logger: console/file logging
- save_checkpoint / load_checkpoint: round-trip
- RunningMeanStd: online mean/std computation
"""
import pytest
import random
import numpy as np
import torch
import tempfile
import os
import shutil
import logging

from agents import PPOAgent
from utils.helpers import (
    set_seed,
    get_device,
    create_directories,
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    RunningMeanStd,
)


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

class TestSetSeed:
    def test_python_random_reproducible(self):
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_reproducible(self):
        set_seed(42)
        a = np.random.randn(10).tolist()
        set_seed(42)
        b = np.random.randn(10).tolist()
        assert a == b

    def test_torch_reproducible(self):
        set_seed(42)
        a = torch.randn(10).tolist()
        set_seed(42)
        b = torch.randn(10).tolist()
        assert a == b

    def test_different_seeds_differ(self):
        set_seed(42)
        a = np.random.randn(10)
        set_seed(99)
        b = np.random.randn(10)
        assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_cpu(self):
        d = get_device("cpu")
        assert d == torch.device("cpu")

    def test_auto(self):
        d = get_device("auto")
        assert d.type in ["cpu", "cuda"]

    def test_explicit_cuda_string(self):
        """Should accept 'cuda' even if not available (just creates device obj)."""
        d = get_device("cuda")
        assert d == torch.device("cuda")


# ---------------------------------------------------------------------------
# create_directories
# ---------------------------------------------------------------------------

class TestCreateDirectories:
    def test_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "output_dir": os.path.join(tmpdir, "outputs"),
                "log_dir": os.path.join(tmpdir, "logs"),
                "training": {"checkpoint_dir": os.path.join(tmpdir, "ckpt")},
            }
            create_directories(config)
            assert os.path.isdir(config["output_dir"])
            assert os.path.isdir(config["log_dir"])
            assert os.path.isdir(config["training"]["checkpoint_dir"])

    def test_idempotent(self):
        """Calling twice should not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "output_dir": os.path.join(tmpdir, "out"),
                "log_dir": os.path.join(tmpdir, "log"),
                "training": {"checkpoint_dir": os.path.join(tmpdir, "ck")},
            }
            create_directories(config)
            create_directories(config)  # should not raise


# ---------------------------------------------------------------------------
# setup_logger
# ---------------------------------------------------------------------------

class TestSetupLogger:
    def test_returns_logger(self):
        logger = setup_logger("test_logger_1")
        assert isinstance(logger, logging.Logger)
        # Cleanup handlers
        logger.handlers.clear()

    def test_log_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            logger = setup_logger("test_logger_2", log_file=log_path)
            logger.info("test message")
            # Flush
            for h in logger.handlers:
                h.flush()
            assert os.path.exists(log_path)
            with open(log_path) as f:
                content = f.read()
            assert "test message" in content
            logger.handlers.clear()

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "sub", "dir", "test.log")
            logger = setup_logger("test_logger_3", log_file=log_path)
            logger.info("hello")
            for h in logger.handlers:
                h.flush()
            assert os.path.exists(log_path)
            logger.handlers.clear()


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_load_roundtrip(self):
        agent = PPOAgent(observation_dim=4, action_dim=2, device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(
                agent, agent.optimizer,
                timestep=1000,
                metrics={"mean_reward": 42.0},
                path=path,
            )

            agent2 = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
            ts, metrics = load_checkpoint(path, agent2, agent2.optimizer)
            assert ts == 1000
            assert metrics["mean_reward"] == pytest.approx(42.0)

            # Weights should match
            for (_, p1), (_, p2) in zip(
                agent.network.named_parameters(),
                agent2.network.named_parameters()
            ):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)

    def test_load_without_optimizer(self):
        agent = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(agent, agent.optimizer, 500, {"r": 1.0}, path)
            agent2 = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
            ts, metrics = load_checkpoint(path, agent2, optimizer=None)
            assert ts == 500
        finally:
            os.unlink(path)

    def test_checkpoint_contains_all_keys(self):
        agent = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(agent, agent.optimizer, 100, {"x": 1.0}, path)
            ckpt = torch.load(path, map_location="cpu")
            assert "timestep" in ckpt
            assert "agent_state_dict" in ckpt
            assert "optimizer_state_dict" in ckpt
            assert "metrics" in ckpt
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------

class TestRunningMeanStd:
    def test_initial_values(self):
        rms = RunningMeanStd(shape=(3,))
        assert np.allclose(rms.mean, np.zeros(3))
        assert np.allclose(rms.var, np.ones(3))

    def test_single_batch(self):
        rms = RunningMeanStd(shape=())
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(data.reshape(-1, 1) if len(data.shape) == 1 else data)
        # After one batch the mean should be close to the batch mean
        # (not exact due to initial epsilon count)
        assert abs(rms.mean - 3.0) < 0.1

    def test_multiple_batches_converge(self):
        rms = RunningMeanStd(shape=())
        np.random.seed(42)
        for _ in range(100):
            batch = np.random.randn(32, 1) * 2 + 5  # mean=5, std=2
            rms.update(batch)
        assert abs(rms.mean - 5.0) < 0.2
        assert abs(np.sqrt(rms.var) - 2.0) < 0.3

    def test_multidim(self):
        rms = RunningMeanStd(shape=(3,))
        np.random.seed(42)
        for _ in range(50):
            batch = np.random.randn(16, 3)
            rms.update(batch)
        assert rms.mean.shape == (3,)
        assert rms.var.shape == (3,)
        # Each dimension should have mean ~0, var ~1
        assert np.allclose(rms.mean, np.zeros(3), atol=0.2)
        assert np.allclose(rms.var, np.ones(3), atol=0.3)

    def test_count_increases(self):
        rms = RunningMeanStd(shape=())
        initial_count = rms.count
        rms.update(np.array([[1.0], [2.0]]))
        assert rms.count > initial_count

    def test_single_value(self):
        rms = RunningMeanStd(shape=())
        rms.update(np.array([[42.0]]))
        # Mean should be close to 42
        assert abs(rms.mean - 42.0) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
