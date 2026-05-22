"""Tests for GenomeGraph v2 (vector-node version)."""

import numpy as np
import pytest

from genesis_v2.genome.graph import (
    D_NODE, D_OUT, EdgeKind, GraphConfig, GenomeGraph, NodeType,
    new_genome_graph,
)


class TestGenomeGraph:
    def test_construction(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        g = new_genome_graph(cfg, rng)

        assert len(g.input_nodes) == cfg.input_nodes
        assert len(g.output_nodes) == cfg.output_nodes
        assert g.node_count() >= cfg.input_nodes + cfg.output_nodes + cfg.initial_hidden_nodes
        assert g.edge_count() > 0

    def test_node_types(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig(input_nodes=4, output_nodes_action=1, output_nodes_message=1,
                          output_nodes_state=1, output_nodes_selfmod=1, initial_hidden_nodes=2)
        g = new_genome_graph(cfg, rng)

        types = {n.type for n in g.nodes.values()}
        assert NodeType.INPUT in types
        assert NodeType.OUTPUT in types
        assert NodeType.HIDDEN in types

    def test_output_partition_helpers(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        g = new_genome_graph(cfg, rng)

        assert len(g.get_action_nodes()) == 4
        assert len(g.get_message_nodes()) == 2
        assert len(g.get_state_nodes()) == 2
        assert len(g.get_selfmod_nodes()) == 1

    def test_copy_preserves_ids(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig(initial_hidden_nodes=2)
        g = new_genome_graph(cfg, rng)
        g2 = g.copy()

        assert g2.node_count() == g.node_count()
        assert g2.edge_count() == g.edge_count()
        assert set(g2.nodes.keys()) == set(g.nodes.keys())
        assert set(g2.edges.keys()) == set(g.edges.keys())
        # Recurrent state should be cleared
        assert len(g2._last_hidden) == 0

    def test_copy_independent_weights(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        g = new_genome_graph(cfg, rng)
        g2 = g.copy()

        # Mutate original — copy should be unaffected
        for e in g.edges.values():
            e.weight[:] = 0.0
            break
        for e in g2.edges.values():
            assert not np.all(e.weight == 0.0)
            break

    def test_entropy(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig(initial_hidden_nodes=4)
        g = new_genome_graph(cfg, rng)
        h = g.entropy()
        assert h > 0.0  # mixed types should have positive entropy

    def test_entropy_empty(self):
        g = GenomeGraph()
        assert g.entropy() == 0.0

    def test_reset_state(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig(initial_hidden_nodes=2)
        g = new_genome_graph(cfg, rng)
        g._last_hidden = {0: np.zeros(D_NODE)}
        g.reset_state()
        assert len(g._last_hidden) == 0

    def test_determinism(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        cfg = GraphConfig()
        g1 = new_genome_graph(cfg, rng1)
        g2 = new_genome_graph(cfg, rng2)

        assert g1.node_count() == g2.node_count()
        assert g1.edge_count() == g2.edge_count()
        for eid in sorted(g1.edges):
            assert eid in g2.edges
            np.testing.assert_array_equal(g1.edges[eid].weight, g2.edges[eid].weight)

    def test_edge_weight_shapes(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        g = new_genome_graph(cfg, rng)

        for e in g.edges.values():
            src_dim = g.nodes[e.src].dim
            dst_dim = g.nodes[e.dst].dim
            assert e.weight.shape == (dst_dim, src_dim)

    def test_persistence_roundtrip(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig(initial_hidden_nodes=2)
        g = new_genome_graph(cfg, rng)

        payload = g.to_payload()
        g2 = GenomeGraph.from_payload(payload)

        assert g2.node_count() == g.node_count()
        assert g2.edge_count() == g.edge_count()
        for eid in sorted(g.edges):
            np.testing.assert_array_equal(g.edges[eid].weight, g2.edges[eid].weight)

    def test_touch_forward_cache(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        g = new_genome_graph(cfg, rng)
        rev_before = g._forward_cache_revision
        g._add_node(NodeType.HIDDEN)
        assert g._forward_cache_revision == rev_before + 1
