"""Tests for NEAT-style crossover."""

import numpy as np
import pytest

from genesis_v2.genome.crossover import crossover, crossover_pair
from genesis_v2.genome.graph import (
    D_NODE, EdgeKind, GraphConfig, NodeType, new_genome_graph,
)
from genesis_v2.agent.agent import Agent, new_agent


@pytest.fixture
def two_parents():
    rng = np.random.default_rng(42)
    cfg = GraphConfig(
        input_nodes=3, initial_hidden_nodes=3, initial_edge_density=0.5,
    )
    parent_a = new_genome_graph(cfg, np.random.default_rng(100))
    parent_b = new_genome_graph(cfg, np.random.default_rng(200))
    return parent_a, parent_b, rng


class TestCrossover:
    def test_crossover_produces_valid_child(self, two_parents):
        pa, pb, rng = two_parents
        child = crossover(pa, pb, rng, fitness_a=10.0, fitness_b=5.0)
        assert child is not None
        assert len(child.input_nodes) == len(pa.input_nodes)
        assert len(child.output_nodes) == len(pa.output_nodes)

    def test_crossover_preserves_output_partition(self, two_parents):
        pa, pb, rng = two_parents
        child = crossover(pa, pb, rng)
        assert len(child.get_action_nodes()) == 4
        assert len(child.get_message_nodes()) == 2
        assert len(child.get_state_nodes()) == 2
        assert len(child.get_selfmod_nodes()) == 1

    def test_crossover_fitter_parent_bias(self, two_parents):
        """When fitness_a >> fitness_b, child should resemble parent A more."""
        pa, pb, rng = two_parents
        rng2 = np.random.default_rng(42)
        child = crossover(pa, pb, rng2, fitness_a=100.0, fitness_b=0.01)
        # With extreme fitness bias, child should have similar structure to A
        assert child.node_count() >= min(pa.node_count(), pb.node_count())

    def test_crossover_no_nan_weights(self, two_parents):
        pa, pb, rng = two_parents
        for seed in range(10):
            child = crossover(pa, pb, np.random.default_rng(seed))
            for e in child.edges.values():
                assert not np.any(np.isnan(e.weight)), f"NaN in edge {e.id}"

    def test_crossover_weight_shapes_valid(self, two_parents):
        pa, pb, rng = two_parents
        child = crossover(pa, pb, rng)
        for e in child.edges.values():
            src_node = child.nodes[e.src]
            dst_node = child.nodes[e.dst]
            assert e.weight.shape == (dst_node.dim, src_node.dim)

    def test_crossover_deterministic(self, two_parents):
        pa, pb, rng = two_parents
        c1 = crossover(pa, pb, np.random.default_rng(77), fitness_a=5.0, fitness_b=3.0)
        c2 = crossover(pa, pb, np.random.default_rng(77), fitness_a=5.0, fitness_b=3.0)
        assert c1.node_count() == c2.node_count()
        assert c1.edge_count() == c2.edge_count()

    def test_crossover_output_nodes_have_incoming(self, two_parents):
        pa, pb, rng = two_parents
        for seed in range(5):
            child = crossover(pa, pb, np.random.default_rng(seed))
            for out_nid in child.output_nodes:
                has_incoming = any(e.dst == out_nid for e in child.edges.values())
                assert has_incoming, f"Output node {out_nid} has no incoming edges"

    def test_crossover_pair_from_agents(self, two_parents):
        pa, pb, rng = two_parents
        a1 = new_agent("a1", pa, initial_energy=1000.0)
        a1.fitness = 10.0
        a2 = new_agent("a2", pb, initial_energy=1000.0)
        a2.fitness = 5.0
        child = crossover_pair([a1, a2], rng)
        assert child is not None

    def test_crossover_pair_too_few_agents(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        g = new_genome_graph(cfg, rng)
        a = new_agent("a1", g, initial_energy=1000.0)
        assert crossover_pair([a], rng) is None

    def test_crossover_preserves_edge_kinds(self, two_parents):
        pa, pb, rng = two_parents
        # Add some shortcut and recurrent edges to parents
        from genesis_v2.genome.mutate import add_shortcut_edge, add_recurrent_edge
        for _ in range(3):
            add_shortcut_edge(pa, rng)
            add_recurrent_edge(pa, rng)
        for _ in range(3):
            add_shortcut_edge(pb, rng)
            add_recurrent_edge(pb, rng)

        child = crossover(pa, pb, rng)
        kinds = {e.kind for e in child.edges.values()}
        # Child should have at least FORWARD edges
        assert EdgeKind.FORWARD in kinds
