"""Tests for MockMathEnvironment."""

import numpy as np
import pytest

from genesis_v2.env.mock import MockMathEnvironment


class TestMockMathEnvironment:
    def test_observe_shape(self):
        env = MockMathEnvironment(n_cells=64, rng=np.random.default_rng(0))
        obs = env.observe()
        assert obs.shape == (64,)
        assert obs.dtype == np.float32

    def test_interact_returns_observation(self):
        env = MockMathEnvironment(n_cells=32, rng=np.random.default_rng(0))
        action = np.zeros(32, dtype=np.float32)
        fb = env.interact(action)
        assert fb.shape == (32,)
        assert fb.dtype == np.float32

    def test_true_distribution_shape(self):
        env = MockMathEnvironment(n_cells=16, rng=np.random.default_rng(0))
        truth = env.true_distribution(np.zeros(16, dtype=np.float32))
        assert truth.shape == (16,)

    def test_multi_rule_rotation(self):
        env = MockMathEnvironment(n_cells=16, rng=np.random.default_rng(0))
        action = np.zeros(16, dtype=np.float32)
        for _ in range(150):
            env.interact(action)
        # Rotates at tick 50, 100, 150 → 3 rotations
        assert env._current_rule_idx == 3

    def test_values_in_range(self):
        env = MockMathEnvironment(n_cells=32, rng=np.random.default_rng(0))
        for _ in range(50):
            obs = env.observe()
            assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
            env.interact(np.zeros(32, dtype=np.float32))

    def test_close(self):
        env = MockMathEnvironment(n_cells=16, rng=np.random.default_rng(0))
        env.close()  # Should not raise
