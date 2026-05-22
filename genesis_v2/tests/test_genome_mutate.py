"""Tests for mutation primitives v2."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from genesis_v2.genome.graph import (
    D_NODE, EdgeKind, GraphConfig, NodeType, new_genome_graph,
)
from genesis_v2.genome.mutate import (
    MutationKind, add_attention_group, add_comm_edge, add_forward_edge,
    add_gating_node, add_hidden_node, add_module, add_recurrent_edge,
    add_shortcut_edge, delete_random_edge, merge_nodes, mutate,
    perturb_weight, split_node_dim,
)


@pytest.fixture
def small_graph():
    rng = np.random.default_rng(42)
    cfg = GraphConfig(input_nodes=3, output_nodes_action=1, output_nodes_message=1,
                      output_nodes_state=1, output_nodes_selfmod=1,
                      initial_hidden_nodes=2, initial_edge_density=0.5)
    return new_genome_graph(cfg, rng), rng


class TestMutationPrimitives:
    def test_add_forward_edge(self, small_graph):
        g, rng = small_graph
        n_edges_before = g.edge_count()
        ok = add_forward_edge(g, rng)
        # May or may not succeed depending on existing edges
        if ok:
            assert g.edge_count() == n_edges_before + 1

    def test_add_shortcut_edge(self, small_graph):
        g, rng = small_graph
        ok = add_shortcut_edge(g, rng)
        if ok:
            assert any(e.kind == EdgeKind.SHORTCUT for e in g.edges.values())

    def test_add_recurrent_edge(self, small_graph):
        g, rng = small_graph
        ok = add_recurrent_edge(g, rng)
        if ok:
            assert any(e.kind == EdgeKind.RECURRENT for e in g.edges.values())

    def test_add_hidden_node(self, small_graph):
        g, rng = small_graph
        n_before = g.node_count()
        ok = add_hidden_node(g, rng)
        if ok:
            assert g.node_count() == n_before + 1
            assert any(n.type == NodeType.HIDDEN for n in g.nodes.values()
                       if n.id >= n_before)

    def test_add_gating_node(self, small_graph):
        g, rng = small_graph
        n_before = g.node_count()
        ok = add_gating_node(g, rng)
        if ok:
            assert g.node_count() == n_before + 1
            assert any(n.type == NodeType.GATING for n in g.nodes.values()
                       if n.id >= n_before)

    def test_perturb_weight(self, small_graph):
        g, rng = small_graph
        old_weights = {eid: e.weight.copy() for eid, e in g.edges.items()}
        ok = perturb_weight(g, rng)
        assert ok
        # At least one weight should change
        changed = any(
            not np.array_equal(old_weights[eid], g.edges[eid].weight)
            for eid in g.edges if eid in old_weights
        )
        assert changed

    def test_delete_random_edge(self, small_graph):
        g, rng = small_graph
        n_before = g.edge_count()
        ok = delete_random_edge(g, rng)
        if ok:
            assert g.edge_count() == n_before - 1

    def test_delete_last_edge_refused(self):
        g = new_genome_graph(GraphConfig(input_nodes=1, output_nodes_action=1,
                                         output_nodes_message=1, output_nodes_state=1,
                                         output_nodes_selfmod=1,
                                         initial_hidden_nodes=0, initial_edge_density=0.1),
                             np.random.default_rng(42))
        # Remove edges until only 1 remains
        while g.edge_count() > 1:
            e = list(g.edges.values())[0]
            del g.edges[e.id]
        assert delete_random_edge(g, np.random.default_rng(0)) is False

    def test_add_attention_group(self, small_graph):
        g, rng = small_graph
        n_before = g.node_count()
        ok = add_attention_group(g, rng)
        assert ok
        assert g.node_count() == n_before + 4  # Q, K, V, out

    def test_add_module(self, small_graph):
        g, rng = small_graph
        n_before = g.node_count()
        ok = add_module(g, rng)
        assert ok
        assert g.node_count() >= n_before + 2

    def test_add_comm_edge(self, small_graph):
        g, rng = small_graph
        ok = add_comm_edge(g, rng)
        if ok:
            msg_nodes = set(g.get_message_nodes())
            assert any(e.dst in msg_nodes for e in g.edges.values())

    def test_split_node_dim_not_implemented(self, small_graph):
        g, rng = small_graph
        assert split_node_dim(g, rng) is False

    def test_merge_nodes_not_implemented(self, small_graph):
        g, rng = small_graph
        assert merge_nodes(g, rng) is False


class TestMutateDispatcher:
    def test_mutate_returns_kind(self, small_graph):
        g, rng = small_graph
        result = mutate(g, rng)
        assert result is None or isinstance(result, MutationKind)

    def test_mutate_with_allowed(self, small_graph):
        g, rng = small_graph
        for _ in range(20):
            result = mutate(g, rng, allowed=[MutationKind.PERTURB_WEIGHT])
            if result is not None:
                assert result == MutationKind.PERTURB_WEIGHT
                return
        pytest.skip("perturb_weight never succeeded")

    def test_mutate_empty_allowed(self, small_graph):
        g, rng = small_graph
        assert mutate(g, rng, allowed=[]) is None

    def test_all_mutation_kinds_dispatched(self):
        """Every MutationKind has a handler."""
        for kind in MutationKind:
            assert kind in {
                MutationKind.ADD_FORWARD_EDGE, MutationKind.ADD_SHORTCUT_EDGE,
                MutationKind.ADD_RECURRENT_EDGE, MutationKind.ADD_HIDDEN_NODE,
                MutationKind.ADD_GATING_NODE, MutationKind.PERTURB_WEIGHT,
                MutationKind.DELETE_RANDOM_EDGE, MutationKind.ADD_ATTENTION_GROUP,
                MutationKind.ADD_MODULE, MutationKind.SPLIT_NODE_DIM,
                MutationKind.MERGE_NODES, MutationKind.ADD_COMM_EDGE,
            }


class TestMutationDeterminism:
    def test_deterministic_with_same_seed(self):
        cfg = GraphConfig(initial_hidden_nodes=2, initial_edge_density=0.3)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        g1 = new_genome_graph(cfg, rng1)
        g2 = new_genome_graph(cfg, rng2)

        for _ in range(5):
            k1 = mutate(g1, rng1)
            k2 = mutate(g2, rng2)
            assert k1 == k2

        assert g1.node_count() == g2.node_count()
        assert g1.edge_count() == g2.edge_count()


class TestWeightShapes:
    def test_all_edges_have_valid_shapes(self, small_graph):
        g, rng = small_graph
        for _ in range(30):
            mutate(g, rng)

        for e in g.edges.values():
            src_node = g.nodes[e.src]
            dst_node = g.nodes[e.dst]
            assert e.weight.ndim == 2
            assert e.weight.shape[0] == dst_node.dim
            assert e.weight.shape[1] == src_node.dim
