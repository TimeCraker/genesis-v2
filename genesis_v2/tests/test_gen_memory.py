"""Tests for GenerationalMemory."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.evolution.gen_memory import GenMemory, GenerationalMemoryBank
from genesis_v2.genome.graph import GraphConfig, new_genome_graph


@pytest.fixture
def agents():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    result = []
    for i in range(5):
        g = new_genome_graph(cfg, rng)
        a = new_agent(f"mem-{i}", g, initial_energy=1000.0)
        a.fitness = float(i * 10)
        a.last_action = rng.standard_normal(256).astype(np.float32)
        a.last_state = rng.standard_normal(128).astype(np.float32)
        a.social_memory = {"partner-1": 0.5}
        result.append(a)
    return result


class TestGenMemory:
    def test_to_from_payload(self):
        mem = GenMemory(
            agent_id="test-1",
            generation=3,
            fitness=42.5,
            behavioral_signature=np.ones(64, dtype=np.float32),
            successful_patterns=[np.zeros(64, dtype=np.float32)],
            social_partners=["partner-1"],
            env_adaptation_scores={"deepseek": 0.8},
        )
        payload = mem.to_payload()
        restored = GenMemory.from_payload(payload)
        assert restored.agent_id == "test-1"
        assert restored.generation == 3
        assert np.allclose(restored.behavioral_signature, 1.0)
        assert len(restored.successful_patterns) == 1
        assert restored.social_partners == ["partner-1"]


class TestGenerationalMemoryBank:
    def test_record_generation(self, agents):
        bank = GenerationalMemoryBank(max_per_generation=3)
        count = bank.record_generation(agents, generation=0)
        assert count == 3  # limited by max_per_generation
        assert 0 in bank.memories
        assert len(bank.memories[0]) == 3

    def test_record_generation_no_alive(self):
        bank = GenerationalMemoryBank()
        a = Agent(id="dead", is_alive=False)
        count = bank.record_generation([a], generation=0)
        assert count == 0

    def test_get_parent_memory(self, agents):
        bank = GenerationalMemoryBank()
        bank.record_generation(agents, generation=2)
        mem = bank.get_parent_memory(generation=3)
        assert mem is not None
        assert mem.generation == 2

    def test_get_parent_memory_no_previous(self):
        bank = GenerationalMemoryBank()
        assert bank.get_parent_memory(generation=0) is None

    def test_apply_to_offspring(self, agents):
        bank = GenerationalMemoryBank()
        bank.record_generation(agents, generation=0)

        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        g = new_genome_graph(cfg, rng)
        child = new_agent("child-1", g, initial_energy=1000.0, generation=1)

        # Force apply (probability 1.0)
        applied = bank.apply_to_offspring(child, rng, inherit_prob=1.0)
        assert applied
        assert "parent_behavioral_sig" in child.birth_mem

    def test_apply_to_offspring_random_skip(self, agents):
        bank = GenerationalMemoryBank()
        bank.record_generation(agents, generation=0)

        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        g = new_genome_graph(cfg, rng)
        child = new_agent("child-2", g, initial_energy=1000.0, generation=1)

        # With inherit_prob=0, should never apply
        applied = bank.apply_to_offspring(child, rng, inherit_prob=0.0)
        assert not applied

    def test_save_load(self, agents):
        bank = GenerationalMemoryBank()
        bank.record_generation(agents, generation=0)
        bank.record_generation(agents, generation=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            bank.save(Path(tmpdir))
            bank2 = GenerationalMemoryBank()
            count = bank2.load(Path(tmpdir))
            assert count == len(bank.memories[0]) + len(bank.memories[1])
            assert 0 in bank2.memories
            assert 1 in bank2.memories

    def test_load_nonexistent_dir(self):
        bank = GenerationalMemoryBank()
        count = bank.load(Path("/nonexistent/dir"))
        assert count == 0
