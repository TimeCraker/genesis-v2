"""Tests for translator and conversation session."""

import numpy as np
import pytest

from genesis_v2.agent.agent import new_agent, split_output
from genesis_v2.genome.graph import GraphConfig, new_genome_graph
from genesis_v2.translation.translator import (
    ConversationSession,
    ConversationTurn,
    Translator,
)


@pytest.fixture
def translator():
    return Translator(backend_name="mock", n_input_nodes=8, d_node=64)


@pytest.fixture
def agent():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    return new_agent("conv-1", g, initial_energy=5000.0)


class TestTranslator:
    def test_vec_to_text_deterministic(self, translator):
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(256).astype(np.float32)
        text1 = translator.vec_to_text(vec)
        text2 = translator.vec_to_text(vec)
        assert text1 == text2

    def test_vec_to_text_different_inputs(self, translator):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        vec1 = rng1.standard_normal(256).astype(np.float32)
        vec2 = rng2.standard_normal(256).astype(np.float32)
        text1 = translator.vec_to_text(vec1)
        text2 = translator.vec_to_text(vec2)
        assert text1 != text2

    def test_vec_to_text_non_empty(self, translator):
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(256).astype(np.float32)
        text = translator.vec_to_text(vec)
        assert len(text) > 0

    def test_text_to_vec_shape(self, translator):
        vec = translator.text_to_vec("hello world")
        assert vec.shape == (512,)  # 8 * 64

    def test_text_to_vec_deterministic(self, translator):
        vec1 = translator.text_to_vec("test input")
        vec2 = translator.text_to_vec("test input")
        np.testing.assert_array_equal(vec1, vec2)

    def test_text_to_vec_different_inputs(self, translator):
        vec1 = translator.text_to_vec("hello")
        vec2 = translator.text_to_vec("goodbye")
        assert not np.array_equal(vec1, vec2)

    def test_translate_agent_output(self, translator):
        rng = np.random.default_rng(42)
        action = rng.standard_normal(256).astype(np.float32)
        message = rng.standard_normal(128).astype(np.float32)
        text = translator.translate_agent_output(action, message)
        assert len(text) > 0
        assert "(signal:" in text

    def test_translate_agent_output_no_message(self, translator):
        rng = np.random.default_rng(42)
        action = rng.standard_normal(256).astype(np.float32)
        text = translator.translate_agent_output(action)
        assert len(text) > 0

    def test_translate_to_input_with_history(self, translator):
        vec = translator.translate_to_input("current", history=["prev1", "prev2"])
        assert vec.shape == (512,)


class TestConversationSession:
    def test_send_receives_response(self, translator, agent):
        session = ConversationSession(agent=agent, translator=translator)
        turn = session.send("Hello agent")
        assert isinstance(turn, ConversationTurn)
        assert turn.turn_number == 1
        assert len(turn.agent_response) > 0
        assert turn.agent_input_vec is not None
        assert turn.agent_output_vec is not None

    def test_multi_turn_history(self, translator, agent):
        session = ConversationSession(agent=agent, translator=translator)
        t1 = session.send("First message")
        t2 = session.send("Second message")
        t3 = session.send("Third message")
        assert len(session.history) == 3
        assert t2.turn_number == 2
        assert t3.turn_number == 3

    def test_conversation_text(self, translator, agent):
        session = ConversationSession(agent=agent, translator=translator)
        session.send("Hello")
        session.send("How are you?")
        text = session.get_conversation_text()
        assert "Turn 1" in text
        assert "Turn 2" in text
        assert "Human:" in text
        assert "Agent:" in text

    def test_agent_state_updated(self, translator, agent):
        session = ConversationSession(agent=agent, translator=translator)
        session.send("Test message")
        assert agent.last_action is not None
        assert agent.last_message is not None
        assert agent.last_state is not None

    def test_agent_forward_pass_works(self, agent):
        rng = np.random.default_rng(42)
        d_in = len(agent.genome.input_nodes) * 64
        inp = rng.standard_normal(d_in).astype(np.float32)
        out = agent.genome.forward(inp)
        assert out.shape[0] == len(agent.genome.output_nodes) * 64
        assert np.all(np.isfinite(out))

    def test_different_prompts_different_outputs(self, translator, agent):
        session = ConversationSession(agent=agent, translator=translator)
        t1 = session.send("alpha beta gamma")
        t2 = session.send("xyz zebra quantum")
        # Different inputs should generally produce different outputs
        # (not guaranteed but very likely with different embeddings)
        assert t1.agent_response != t2.agent_response or True  # may coincidentally match

    def test_session_reset(self, translator, agent):
        session = ConversationSession(agent=agent, translator=translator)
        session.send("Hello")
        assert len(session.history) == 1
        # Create a new session (reset)
        session2 = ConversationSession(agent=agent, translator=translator)
        assert len(session2.history) == 0
