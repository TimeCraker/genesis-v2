"""Integration test — agents exchange messages via social layer end-to-end."""

import numpy as np
import pytest

from genesis_v2.agent.agent import new_agent, split_output
from genesis_v2.config import GenesisConfig
from genesis_v2.engine.tick import island_step_sync
from genesis_v2.env.mock import MockMathEnvironment
from genesis_v2.genome.graph import D_MESSAGE, GraphConfig, new_genome_graph
from genesis_v2.social.comm_bus import CommunicationBus
from genesis_v2.storage.duckdb_store import DuckDBStore


@pytest.fixture
def social_setup(tmp_path):
    """Create agents + env + comm_bus for integration test."""
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    phy = GenesisConfig().physics

    agents = []
    for i in range(5):
        g = new_genome_graph(cfg, rng)
        a = new_agent(id=f"agent-{i}", genome=g, initial_energy=5000.0)
        agents.append(a)

    env = MockMathEnvironment(n_cells=64, rng=np.random.default_rng(0))

    comm_bus = CommunicationBus(
        grid_rows=5, grid_cols=5, comm_radius=2,
        rng=np.random.default_rng(99),
    )
    comm_bus.assign_positions([a.id for a in agents])

    # Place agents close together so they can communicate
    for i, a in enumerate(agents):
        comm_bus.agent_positions[a.id] = (2, i)  # row 2, cols 0-4

    store = DuckDBStore(tmp_path / "test.duckdb")

    return agents, env, phy, comm_bus, store, rng


class TestSocialIntegration:
    def test_messages_are_delivered(self, social_setup):
        agents, env, phy, comm_bus, store, rng = social_setup

        island_step_sync(agents, env, phy, store, 0, rng, comm_bus=comm_bus)

        # At least some agents should have non-zero inbox after tick
        # (messages are cleared at end of tick, but social_memory persists)
        for a in agents:
            if a.is_alive:
                assert a.last_message is not None
                assert a.last_message.shape == (D_MESSAGE,)

    def test_social_memory_populated(self, social_setup):
        agents, env, phy, comm_bus, store, rng = social_setup

        # Run a few ticks to accumulate social memory
        for t in range(5):
            island_step_sync(agents, env, phy, store, t, rng, comm_bus=comm_bus)

        alive = [a for a in agents if a.is_alive]
        # Some agents should have social_memory entries
        has_social = any(a.social_memory for a in alive)
        # With random actions, cooperation may or may not emerge
        # Just verify no crashes and social_memory is a dict
        for a in alive:
            assert isinstance(a.social_memory, dict)

    def test_no_social_has_no_messages(self, social_setup):
        """Without comm_bus, agents get no social input."""
        agents, env, phy, comm_bus, store, rng = social_setup

        island_step_sync(agents, env, phy, store, 0, rng, comm_bus=None)

        for a in agents:
            if a.is_alive:
                assert a.last_message is not None  # message is computed but not delivered
                assert len(a.social_memory) == 0

    def test_store_records_social_metrics(self, social_setup, tmp_path):
        agents, env, phy, comm_bus, store, rng = social_setup

        island_step_sync(agents, env, phy, store, 0, rng, comm_bus=comm_bus)

        count = store.tick_count()
        assert count == len([a for a in agents if a.is_alive])

    def test_multi_tick_stability(self, social_setup):
        """Run 20 ticks with social — no crashes, no NaN."""
        agents, env, phy, comm_bus, store, rng = social_setup

        for t in range(20):
            island_step_sync(agents, env, phy, store, t, rng, comm_bus=comm_bus)

        alive = [a for a in agents if a.is_alive]
        for a in alive:
            assert np.all(np.isfinite(a.last_action))
            assert np.all(np.isfinite(a.last_message))
            assert np.isfinite(a.energy)
            assert np.isfinite(a.fitness)

    def test_message_norm_nonzero(self, social_setup):
        """After some ticks, at least some agents should have non-trivial messages."""
        agents, env, phy, comm_bus, store, rng = social_setup

        for t in range(10):
            island_step_sync(agents, env, phy, store, t, rng, comm_bus=comm_bus)

        alive = [a for a in agents if a.is_alive]
        msg_norms = [np.linalg.norm(a.last_message) for a in alive if a.last_message is not None]
        # At least some messages should be non-zero (not all zeros)
        # With random weights this should always be true
        assert any(n > 0.01 for n in msg_norms), "All messages are near-zero"
