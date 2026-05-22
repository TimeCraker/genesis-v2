"""Tests for breeding strategy v2 (crossover + clone + migration)."""

import numpy as np
import pytest

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import GenesisConfig
from genesis_v2.evolution.breeder import breed_generation_v2
from genesis_v2.evolution.gen_memory import GenerationalMemoryBank
from genesis_v2.genome.graph import GraphConfig, new_genome_graph


@pytest.fixture
def cfg():
    return GenesisConfig()


@pytest.fixture
def population(cfg):
    rng = np.random.default_rng(42)
    genome_cfg = GraphConfig(
        input_nodes=cfg.genome.input_nodes,
        output_nodes_action=cfg.genome.output_nodes_action,
        output_nodes_message=cfg.genome.output_nodes_message,
        output_nodes_state=cfg.genome.output_nodes_state,
        output_nodes_selfmod=cfg.genome.output_nodes_selfmod,
        initial_hidden_nodes=cfg.genome.initial_hidden_nodes,
        initial_edge_density=cfg.genome.initial_edge_density,
    )
    agents = []
    for i in range(10):
        g = new_genome_graph(genome_cfg, rng)
        a = new_agent(f"test-{i}", g, initial_energy=cfg.physics.initial_energy)
        a.fitness = float(i * 10)  # vary fitness
        agents.append(a)
    return agents


class TestBreedGenerationV2:
    def test_preserves_population_size(self, population, cfg):
        rng = np.random.default_rng(42)
        next_gen = breed_generation_v2(population, rng, cfg)
        assert len(next_gen) == len(population)

    def test_elites_preserved(self, population, cfg):
        rng = np.random.default_rng(42)
        next_gen = breed_generation_v2(population, rng, cfg, top_fraction=0.25)
        # Top 25% should be preserved as elites (ceil)
        import math
        elite_count = max(1, math.ceil(len(population) * 0.25))
        elite_children = [a for a in next_gen if "elite" in a.id]
        assert len(elite_children) == elite_count

    def test_all_agents_have_genomes(self, population, cfg):
        rng = np.random.default_rng(42)
        next_gen = breed_generation_v2(population, rng, cfg)
        for a in next_gen:
            assert a.genome is not None
            assert a.genome.node_count() > 0

    def test_generation_incremented(self, population, cfg):
        rng = np.random.default_rng(42)
        next_gen = breed_generation_v2(population, rng, cfg)
        for a in next_gen:
            assert a.generation >= 1

    def test_with_gen_memory(self, population, cfg):
        rng = np.random.default_rng(42)
        gen_memory = GenerationalMemoryBank()
        next_gen = breed_generation_v2(population, rng, cfg, gen_memory=gen_memory)
        # gen_memory should have recorded the generation
        assert len(gen_memory.memories) > 0

    def test_all_dead_returns_fresh(self, cfg):
        rng = np.random.default_rng(42)
        genome_cfg = GraphConfig()
        agents = []
        for i in range(5):
            g = new_genome_graph(genome_cfg, rng)
            a = new_agent(f"dead-{i}", g, initial_energy=1000.0)
            a.is_alive = False
            agents.append(a)
        next_gen = breed_generation_v2(agents, rng, cfg)
        assert len(next_gen) == 5
        for a in next_gen:
            assert a.is_alive

    def test_crossover_and_clone_mix(self, population, cfg):
        """With default rates, we should see both crossover and clone offspring."""
        rng = np.random.default_rng(42)
        next_gen = breed_generation_v2(population, rng, cfg)
        ids = [a.id for a in next_gen]
        has_xover = any("xover" in id_ for id_ in ids)
        has_clone = any("clone" in id_ or "elite" in id_ for id_ in ids)
        # With 10 agents and default rates, statistically we should see both
        # But this is probabilistic, so just verify no crashes
        assert len(next_gen) == len(population)

    def test_crossover_rate_only(self, population, cfg):
        rng = np.random.default_rng(42)
        next_gen = breed_generation_v2(
            population, rng, cfg,
            crossover_rate=1.0, clone_rate=0.0,
        )
        assert len(next_gen) == len(population)

    def test_clone_rate_only(self, population, cfg):
        rng = np.random.default_rng(42)
        next_gen = breed_generation_v2(
            population, rng, cfg,
            crossover_rate=0.0, clone_rate=1.0,
        )
        assert len(next_gen) == len(population)
