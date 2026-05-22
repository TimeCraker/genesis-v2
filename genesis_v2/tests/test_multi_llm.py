"""Tests for MultiLLMEnvironment, BudgetManager integration, and FrozenEmbeddingAtlas."""

import numpy as np
import pytest

from genesis_v2.env.budget import BudgetManager
from genesis_v2.env.embed import FrozenEmbeddingAtlas
from genesis_v2.env.mock import MockMathEnvironment
from genesis_v2.genome.graph import D_ACTION


class TestFrozenEmbeddingAtlas:
    def test_creation(self):
        atlas = FrozenEmbeddingAtlas(d_action=256, d_embed=512, seed=42)
        assert atlas.P.shape == (256, 512)

    def test_embed_action_shape(self):
        atlas = FrozenEmbeddingAtlas()
        action = np.random.default_rng(0).standard_normal(D_ACTION).astype(np.float32)
        emb = atlas.embed_action(action)
        assert emb.shape == (atlas.d_embed,)

    def test_decode_feedback_shape(self):
        atlas = FrozenEmbeddingAtlas()
        emb = np.random.default_rng(0).standard_normal(atlas.d_embed).astype(np.float32)
        fb = atlas.decode_feedback(emb)
        assert fb.shape == (D_ACTION,)

    def test_roundtrip_lossy(self):
        atlas = FrozenEmbeddingAtlas(d_action=64, d_embed=64, seed=42)
        action = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        emb = atlas.embed_action(action)
        fb = atlas.decode_feedback(emb)
        # Should be correlated but not identical (lossy projection)
        corr = np.corrcoef(action, fb)[0, 1]
        assert abs(corr) > 0.1  # at least some correlation

    def test_deterministic(self):
        a1 = FrozenEmbeddingAtlas(seed=42)
        a2 = FrozenEmbeddingAtlas(seed=42)
        np.testing.assert_array_equal(a1.P, a2.P)

    def test_nearest_token(self):
        atlas = FrozenEmbeddingAtlas(d_action=8, d_embed=16, seed=42)
        tokens = np.eye(16, dtype=np.float32)  # 16 one-hot tokens
        query = tokens[5]  # exact match for token 5
        idx = atlas.nearest_token(query, tokens)
        assert idx == 5


class TestBudgetManager:
    def test_check_budget_within(self):
        bm = BudgetManager(total_budget=10.0, per_island_budget=5.0)
        assert bm.check_budget(0, 1.0) is True

    def test_check_budget_exceeds_total(self):
        bm = BudgetManager(total_budget=1.0, per_island_budget=10.0)
        bm.record_cost(0, 0.5)
        assert bm.check_budget(0, 0.6) is False

    def test_check_budget_exceeds_island(self):
        bm = BudgetManager(total_budget=100.0, per_island_budget=2.0)
        bm.record_cost(0, 1.5)
        assert bm.check_budget(0, 1.0) is False

    def test_record_cost(self):
        bm = BudgetManager(total_budget=10.0)
        bm.record_cost(0, 1.5)
        bm.record_cost(1, 2.0)
        assert bm.spent_total == pytest.approx(3.5)
        assert bm.spent_per_island[0] == pytest.approx(1.5)
        assert bm.spent_per_island[1] == pytest.approx(2.0)

    def test_should_fallback(self):
        bm = BudgetManager(total_budget=100.0, per_island_budget=2.0, fallback_to_mock=True)
        assert bm.should_fallback(0) is False
        bm.record_cost(0, 2.0)
        assert bm.should_fallback(0) is True

    def test_should_fallback_disabled(self):
        bm = BudgetManager(total_budget=100.0, per_island_budget=2.0, fallback_to_mock=False)
        bm.record_cost(0, 2.0)
        assert bm.should_fallback(0) is False

    def test_remaining(self):
        bm = BudgetManager(total_budget=10.0)
        assert bm.remaining == 10.0
        bm.record_cost(0, 3.0)
        assert bm.remaining == pytest.approx(7.0)

    def test_cost_pressure(self):
        bm = BudgetManager(total_budget=10.0)
        assert bm.get_cost_pressure("any") == 0.0
        bm.record_cost(0, 5.0)
        assert bm.get_cost_pressure("any") == pytest.approx(0.5)


class TestMultiLLMBudgetIntegration:
    def test_budget_blocks_api_call(self):
        """MultiLLMEnvironment returns zeros when budget is exhausted."""
        from genesis_v2.env.multi_llm import MultiLLMEnvironment

        env = MultiLLMEnvironment(backend_name="deepseek", n_cells=64)
        bm = BudgetManager(total_budget=0.0, per_island_budget=0.0)
        env.set_budget(bm, island_id=0)

        action = np.random.default_rng(0).standard_normal(D_ACTION).astype(np.float32)
        result = env.interact(action)
        assert result.shape == (64,)
        assert np.allclose(result, 0.0)
        assert env.last_cost == 0.0
