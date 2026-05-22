"""Tests for conversation quality probes."""

import numpy as np
import pytest

from genesis_v2.agent.agent import new_agent
from genesis_v2.genome.graph import GraphConfig, new_genome_graph
from genesis_v2.metrics.probes.conversation import (
    ConversationProbeResult,
    cross_llm_consistency,
    multi_turn_coherence,
    response_diversity,
    run_conversation_probes,
    semantic_similarity,
)
from genesis_v2.translation.translator import Translator


@pytest.fixture
def translator():
    return Translator(backend_name="mock", n_input_nodes=8, d_node=64)


@pytest.fixture
def agent():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    return new_agent("probe-1", g, initial_energy=5000.0)


class TestResponseDiversity:
    def test_identical_responses_low_diversity(self):
        responses = ["identical"] * 10
        div = response_diversity(responses)
        assert div == 0.0  # single word repeated = zero entropy

    def test_different_responses_higher_diversity(self):
        responses = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
        div = response_diversity(responses)
        assert div > 0.0

    def test_empty_responses(self):
        assert response_diversity([]) == 0.0

    def test_single_response(self):
        div = response_diversity(["hello world"])
        assert div >= 0.0


class TestSemanticSimilarity:
    def test_identical_vectors(self):
        vec = np.array([1.0, 2.0, 3.0])
        assert semantic_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert semantic_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert semantic_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        assert semantic_similarity(a, b) == 0.0


class TestMultiTurnCoherence:
    def test_identical_responses(self):
        responses = ["same"] * 5
        assert multi_turn_coherence(responses) == 1.0

    def test_no_overlap(self):
        responses = ["aaa", "bbb", "ccc"]
        assert multi_turn_coherence(responses) == 0.0

    def test_partial_overlap(self):
        responses = ["hello world", "hello there", "goodbye world"]
        coh = multi_turn_coherence(responses)
        assert 0.0 < coh < 1.0

    def test_single_response(self):
        assert multi_turn_coherence(["only one"]) == 1.0

    def test_empty_list(self):
        assert multi_turn_coherence([]) == 1.0


class TestCrossLLMConsistency:
    def test_consistency_score(self, agent, translator):
        prompts = ["hello", "what are you"]
        score = cross_llm_consistency(agent, translator, prompts, n_translations=3)
        assert -1.0 <= score <= 1.0

    def test_empty_prompts(self, agent, translator):
        score = cross_llm_consistency(agent, translator, [], n_translations=3)
        assert score == 0.0


class TestRunConversationProbes:
    def test_returns_result(self, agent, translator):
        result = run_conversation_probes(agent, translator)
        assert isinstance(result, ConversationProbeResult)
        assert result.n_turns > 0

    def test_custom_prompts(self, agent, translator):
        prompts = ["test prompt 1", "test prompt 2"]
        result = run_conversation_probes(agent, translator, test_prompts=prompts)
        assert result.n_turns == 2

    def test_probe_values_in_range(self, agent, translator):
        result = run_conversation_probes(agent, translator)
        assert result.response_diversity >= 0.0
        assert -1.0 <= result.semantic_similarity <= 1.0
        assert 0.0 <= result.multi_turn_coherence <= 1.0
        assert -1.0 <= result.cross_llm_consistency <= 1.0
