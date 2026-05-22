"""Tests for cross-LLM migration."""

import numpy as np
import pytest

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.genome.graph import GraphConfig, new_genome_graph
from genesis_v2.population.migration import (
    MigrationTracker,
    migrate_agents,
    migration_adaptation_bonus,
    select_migrants,
)


@pytest.fixture
def tracker():
    return MigrationTracker()


@pytest.fixture
def agents():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    result = []
    for i in range(5):
        g = new_genome_graph(cfg, rng)
        a = new_agent(f"mig-{i}", g, initial_energy=1000.0)
        a.fitness = float(i * 10)
        a.prediction_error = 5.0 - i * 0.5
        result.append(a)
    return result


class TestMigrationTracker:
    def test_register_migration(self, tracker, agents):
        record = tracker.register_migration(agents[0], from_island=0, to_island=1)
        assert record.agent_id == agents[0].id
        assert tracker.is_migrant(agents[0].id)
        assert not tracker.is_migrant("nonexistent")

    def test_get_record(self, tracker, agents):
        tracker.register_migration(agents[0], from_island=0, to_island=1)
        record = tracker.get_record(agents[0].id)
        assert record is not None
        assert record.from_island == 0
        assert record.to_island == 1

    def test_tick_migrants(self, tracker, agents):
        tracker.register_migration(agents[0], from_island=0, to_island=1)
        for _ in range(5):
            tracker.tick_migrants()
        record = tracker.get_record(agents[0].id)
        assert record.ticks_since_migration == 5


class TestMigrationAdaptationBonus:
    def test_bonus_for_recent_migrant(self, tracker, agents):
        agent = agents[0]
        # agent.prediction_error is 5.0 from fixture
        tracker.register_migration(agent, from_island=0, to_island=1)
        # Now reduce prediction_error to simulate adaptation
        agent.prediction_error = 3.0
        # KL at migration was 5.0, now it's 3.0 → improvement = 2.0
        phy = PhysicsConfig()
        bonus = migration_adaptation_bonus(agent, tracker, phy)
        assert bonus > 0.0

    def test_bonus_decays_over_time(self, tracker, agents):
        agent = agents[0]
        # agent.prediction_error is 5.0 from fixture
        tracker.register_migration(agent, from_island=0, to_island=1)
        agent.prediction_error = 3.0
        phy = PhysicsConfig()

        bonus_t0 = migration_adaptation_bonus(agent, tracker, phy)
        for _ in range(10):
            tracker.tick_migrants()
        bonus_t10 = migration_adaptation_bonus(agent, tracker, phy)

        assert bonus_t10 < bonus_t0

    def test_bonus_zero_after_20_ticks(self, tracker, agents):
        agent = agents[0]
        tracker.register_migration(agent, from_island=0, to_island=1)
        phy = PhysicsConfig()

        for _ in range(25):
            tracker.tick_migrants()
        bonus = migration_adaptation_bonus(agent, tracker, phy)
        assert bonus == 0.0

    def test_bonus_zero_for_non_migrant(self, tracker, agents):
        phy = PhysicsConfig()
        bonus = migration_adaptation_bonus(agents[0], tracker, phy)
        assert bonus == 0.0

    def test_bonus_zero_when_kl_worsened(self, tracker, agents):
        agent = agents[0]
        # agent.prediction_error is 5.0 from fixture
        tracker.register_migration(agent, from_island=0, to_island=1)
        agent.prediction_error = 10.0  # worse than migration KL (5.0)
        phy = PhysicsConfig()
        bonus = migration_adaptation_bonus(agent, tracker, phy)
        assert bonus == 0.0


class TestSelectMigrants:
    def test_select_top_n(self, agents):
        migrants = select_migrants(agents, n=2)
        assert len(migrants) == 2
        assert migrants[0].fitness >= migrants[1].fitness

    def test_select_more_than_available(self, agents):
        migrants = select_migrants(agents, n=100)
        assert len(migrants) == 5

    def test_select_from_empty(self):
        migrants = select_migrants([], n=3)
        assert len(migrants) == 0


class _FakeIsland:
    def __init__(self, id_, agents, size):
        self.id = id_
        self.name = f"Island-{id_}"
        self.agents = agents
        self.island_cfg = type("IC", (), {"size": size})()


class TestMigrateAgents:
    def test_ring_migration(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        phy = PhysicsConfig()

        islands = []
        for i in range(3):
            agents = []
            for j in range(5):
                g = new_genome_graph(cfg, rng)
                a = new_agent(f"isle{i}-agent{j}", g, initial_energy=1000.0)
                a.fitness = float(j * 10)
                agents.append(a)
            islands.append(_FakeIsland(i, agents, 5))

        tracker = migrate_agents(islands, rng, phy, n_per_island=2)
        assert len(tracker.records) == 6  # 3 islands × 2 migrants

    def test_too_few_islands(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        phy = PhysicsConfig()
        agents = [new_agent("a", new_genome_graph(cfg, rng), 1000.0)]
        islands = [_FakeIsland(0, agents, 1)]

        tracker = migrate_agents(islands, rng, phy)
        assert len(tracker.records) == 0
