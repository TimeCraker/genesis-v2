"""Tests for survivor bank."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis_v2.agent.agent import new_agent
from genesis_v2.evolution.survivor_bank import (
    load_agent, load_top_survivors, save_agent, list_survivors,
)
from genesis_v2.genome.graph import GraphConfig, new_genome_graph


def _make_agent(agent_id: str, fitness: float = 0.0, gen: int = 0):
    rng = np.random.default_rng(42 + gen)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    a = new_agent(id=agent_id, genome=g, initial_energy=1000.0, generation=gen)
    a.fitness = fitness
    return a


class TestSurvivorBank:
    def test_save_and_load(self, tmp_path):
        agent = _make_agent("elite-1", fitness=42.5)
        fpath = save_agent(agent, directory=tmp_path)
        assert fpath.exists()
        assert fpath.suffix == ".json"

        loaded = load_agent(fpath)
        assert loaded.id == agent.id
        assert loaded.fitness == agent.fitness
        assert loaded.genome.node_count() == agent.genome.node_count()

    def test_list_survivors(self, tmp_path):
        for i in range(3):
            agent = _make_agent(f"elite-{i}", fitness=float(i * 10))
            save_agent(agent, directory=tmp_path)

        files = list_survivors(tmp_path)
        assert len(files) == 3
        # Should be sorted by fitness descending
        f0 = float(files[0].stem.split("_fit")[-1])
        f1 = float(files[1].stem.split("_fit")[-1])
        assert f0 >= f1

    def test_load_top_n(self, tmp_path):
        for i in range(5):
            agent = _make_agent(f"top-{i}", fitness=float(i * 5))
            save_agent(agent, directory=tmp_path)

        top3 = load_top_survivors(n=3, directory=tmp_path)
        assert len(top3) == 3
        assert top3[0].fitness >= top3[1].fitness

    def test_roundtrip_preserves_genome(self, tmp_path):
        agent = _make_agent("roundtrip-0", fitness=99.9)
        # Mutate to make genome non-trivial
        rng = np.random.default_rng(42)
        for _ in range(5):
            agent.genome.mutate(rng)

        fpath = save_agent(agent, directory=tmp_path)
        loaded = load_agent(fpath)

        assert loaded.genome.node_count() == agent.genome.node_count()
        assert loaded.genome.edge_count() == agent.genome.edge_count()
