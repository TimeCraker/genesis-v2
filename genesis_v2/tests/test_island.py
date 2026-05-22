"""Tests for Island manager — multi-island creation and configuration."""

import numpy as np
import pytest

from genesis_v2.config import GenesisConfig, IslandConfig, PopulationConfig
from genesis_v2.population.island import Island, create_island, create_islands


@pytest.fixture
def cfg():
    c = GenesisConfig()
    c.population = PopulationConfig(islands=[
        IslandConfig(name="Explorer", size=5, backend="mock", mutation_rate=0.3),
        IslandConfig(name="Exploiter", size=3, backend="mock", mutation_rate=0.05),
    ])
    return c


class TestCreateIsland:
    def test_single_island(self):
        cfg = GenesisConfig()
        isl_cfg = IslandConfig(name="Test", size=5, backend="mock")
        rng = np.random.default_rng(42)
        isl = create_island(0, isl_cfg, cfg, rng)

        assert isl.id == 0
        assert isl.name == "Test"
        assert len(isl.agents) == 5
        assert isl.is_mock is True
        assert isl.comm_bus is not None

    def test_agents_have_island_id(self):
        cfg = GenesisConfig()
        isl_cfg = IslandConfig(name="Test", size=3, backend="mock")
        rng = np.random.default_rng(42)
        isl = create_island(7, isl_cfg, cfg, rng)

        for a in isl.agents:
            assert a.island_id == 7
            assert "isle7" in a.id

    def test_agents_assigned_to_grid(self):
        cfg = GenesisConfig()
        isl_cfg = IslandConfig(name="Test", size=5, backend="mock")
        rng = np.random.default_rng(42)
        isl = create_island(0, isl_cfg, cfg, rng)

        assert len(isl.comm_bus.agent_positions) == 5
        for a in isl.agents:
            assert a.id in isl.comm_bus.agent_positions

    def test_social_disabled(self):
        cfg = GenesisConfig()
        isl_cfg = IslandConfig(name="Test", size=3, backend="mock")
        rng = np.random.default_rng(42)
        isl = create_island(0, isl_cfg, cfg, rng, social=False)

        assert isl.comm_bus.comm_radius == 0


class TestCreateIslands:
    def test_multi_island(self, cfg):
        rng = np.random.default_rng(42)
        islands = create_islands(cfg, rng)

        assert len(islands) == 2
        assert islands[0].name == "Explorer"
        assert islands[1].name == "Exploiter"
        assert len(islands[0].agents) == 5
        assert len(islands[1].agents) == 3

    def test_island_ids_sequential(self, cfg):
        rng = np.random.default_rng(42)
        islands = create_islands(cfg, rng)
        for i, isl in enumerate(islands):
            assert isl.id == i

    def test_default_island_no_config(self):
        cfg = GenesisConfig()  # no islands configured
        rng = np.random.default_rng(42)
        islands = create_islands(cfg, rng)
        assert len(islands) == 1
        assert islands[0].name == "Default"
        assert len(islands[0].agents) == 10

    def test_shared_budget(self, cfg):
        rng = np.random.default_rng(42)
        islands = create_islands(cfg, rng)
        # Both islands should share the same budget instance
        assert islands[0].budget is islands[1].budget
