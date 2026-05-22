"""Tests for forward computation v2 (vector-node BLAS + scalar reference)."""

import numpy as np
import pytest

from genesis_v2.genome.forward import forward_blas, forward_scalar
from genesis_v2.genome.graph import D_NODE, D_OUT, GraphConfig, NodeType, new_genome_graph


@pytest.fixture
def sample_graph():
    rng = np.random.default_rng(42)
    cfg = GraphConfig(input_nodes=4, initial_hidden_nodes=3, initial_edge_density=0.4)
    return new_genome_graph(cfg, rng), cfg, rng


class TestForwardBasic:
    def test_output_shape(self, sample_graph):
        g, cfg, rng = sample_graph
        d_in = cfg.input_dim
        x = rng.standard_normal(d_in).astype(np.float32)
        out = forward_blas(g, x)
        assert out.shape == (D_OUT,)
        assert out.dtype == np.float32

    def test_no_nan_inf(self, sample_graph):
        g, cfg, rng = sample_graph
        d_in = cfg.input_dim
        for t in range(20):
            x = rng.standard_normal(d_in).astype(np.float32)
            out = forward_blas(g, x)
            assert np.all(np.isfinite(out)), f"Non-finite at tick {t}"

    def test_deterministic(self, sample_graph):
        g, cfg, rng = sample_graph
        d_in = cfg.input_dim
        x = rng.standard_normal(d_in).astype(np.float32)

        out1 = forward_blas(g, x)
        g.reset_state()
        g.touch_forward_cache()
        out2 = forward_blas(g, x)
        np.testing.assert_array_equal(out1, out2)

    def test_output_partition_is_set(self, sample_graph):
        g, cfg, rng = sample_graph
        d_in = cfg.input_dim
        x = rng.standard_normal(d_in).astype(np.float32)
        out = forward_blas(g, x)
        assert out.shape[0] == D_OUT


class TestForwardScalarVsBlas:
    def test_scalar_matches_blas(self, sample_graph):
        g, cfg, rng = sample_graph
        d_in = cfg.input_dim
        x = rng.standard_normal(d_in).astype(np.float32)

        out_scalar = forward_scalar(g, x)
        g.reset_state()
        g.touch_forward_cache()
        out_blas = forward_blas(g, x)

        np.testing.assert_allclose(out_scalar, out_blas, atol=1e-5,
                                   err_msg="Scalar and BLAS forward mismatch")

    def test_multi_tick_alignment(self, sample_graph):
        g, cfg, rng = sample_graph
        d_in = cfg.input_dim

        for t in range(10):
            x = rng.standard_normal(d_in).astype(np.float32)
            out_scalar = forward_scalar(g, x)
            out_blas = forward_blas(g, x)
            np.testing.assert_allclose(out_scalar, out_blas, atol=1e-5,
                                       err_msg=f"Mismatch at tick {t}")
            # Both paths update _last_hidden identically
            g.reset_state()


class TestRecurrentEdge:
    def test_recurrent_reads_last_hidden(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig(input_nodes=2, initial_hidden_nodes=1, initial_edge_density=0.0)
        g = new_genome_graph(cfg, rng)
        d_in = cfg.input_dim

        # Add edges: input→hidden (forward) + input→hidden (recurrent) + hidden→output
        from genesis_v2.genome.graph import EdgeKind
        input_id = g.input_nodes[0]
        hidden_id = [n.id for n in g.nodes.values() if n.type == NodeType.HIDDEN][0]
        output_id = g.output_nodes[0]

        w_fwd = rng.standard_normal((D_NODE, D_NODE)).astype(np.float32) * 0.3
        g._add_edge(src=input_id, dst=hidden_id, kind=EdgeKind.FORWARD, weight=w_fwd)

        w_rec = rng.standard_normal((D_NODE, D_NODE)).astype(np.float32) * 0.3
        g._add_edge(src=hidden_id, dst=hidden_id, kind=EdgeKind.RECURRENT, weight=w_rec)

        w_out = rng.standard_normal((D_NODE, D_NODE)).astype(np.float32) * 0.3
        g._add_edge(src=hidden_id, dst=output_id, kind=EdgeKind.FORWARD, weight=w_out)

        # Run tick 1
        x1 = rng.standard_normal(d_in).astype(np.float32)
        out1 = forward_scalar(g, x1)

        # _last_hidden should be populated
        assert len(g._last_hidden) > 0

        # Run tick 2 — recurrent edge reads _last_hidden from tick 1
        x2 = rng.standard_normal(d_in).astype(np.float32)
        out2 = forward_scalar(g, x2)

        # Outputs should differ (different inputs + recurrent state)
        assert not np.allclose(out1, out2)


class TestForwardCache:
    def test_cache_invalidation(self, sample_graph):
        g, cfg, rng = sample_graph
        d_in = cfg.input_dim
        x = rng.standard_normal(d_in).astype(np.float32)

        out1 = forward_blas(g, x)
        # Mutate weight
        e = list(g.edges.values())[0]
        e.weight += rng.standard_normal(e.weight.shape).astype(np.float32) * 0.5
        g.touch_forward_cache()

        out2 = forward_blas(g, x)
        # Should produce different output after weight change
        assert not np.allclose(out1, out2)
