"""Tests for DuckDB store."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from genesis_v2.agent.agent import new_agent
from genesis_v2.genome.graph import GraphConfig, new_genome_graph
from genesis_v2.storage.duckdb_store import DuckDBStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.duckdb"
    s = DuckDBStore(db_path)
    yield s
    s.close()


@pytest.fixture
def sample_agent():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    return new_agent(id="store-0", genome=g, initial_energy=1000.0)


class TestDuckDBStore:
    def test_record_tick(self, store, sample_agent):
        store.record_tick(tick_id=1, agent=sample_agent)
        assert store.tick_count() == 1

    def test_record_multiple_ticks(self, store, sample_agent):
        for t in range(10):
            store.record_tick(tick_id=t, agent=sample_agent)
        assert store.tick_count() == 10

    def test_record_generation(self, store):
        store.record_generation(
            generation=1, island_id=0, alive_count=10,
            mean_fitness=0.5, mean_energy=100.0, best_fitness=1.0,
        )
        # No assertion on count since we don't expose generation count;
        # just verify no exception

    def test_flush(self, store):
        store.flush()  # Should not raise

    def test_empty_store(self, store):
        assert store.tick_count() == 0
