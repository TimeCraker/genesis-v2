"""Tests for all 7 cognitive probes + probe runner."""

import numpy as np
import pytest

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.genome.graph import GenomeGraph, GraphConfig, new_genome_graph
from genesis_v2.metrics.probes.communication import CommunicationResult, probe_communication
from genesis_v2.metrics.probes.exploration_effect import ExplorationResult, probe_exploration
from genesis_v2.metrics.probes.modularity import ModularityResult, probe_modularity
from genesis_v2.metrics.probes.multi_llm import MultiLLMResult, probe_multi_llm
from genesis_v2.metrics.probes.multiscale import MultiScaleResult, probe_multiscale
from genesis_v2.metrics.probes.ood import OODResult, probe_ood
from genesis_v2.metrics.probes.runner import ProbeReport, run_all_probes, save_probe_report
from genesis_v2.metrics.probes.self_mod import SelfModResult, probe_selfmod


@pytest.fixture
def agent():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    a = new_agent("probe-test", g, initial_energy=5000.0)
    # Simulate one tick
    d_in = len(g.input_nodes) * 64
    inp = rng.standard_normal(d_in).astype(np.float32)
    out = g.forward(inp)
    a.last_action = out[:256]
    a.last_message = out[256:384]
    a.last_state = out[384:512]
    a.last_selfmod = out[512:528] if len(out) > 512 else np.zeros(16, dtype=np.float32)
    return a


@pytest.fixture
def agents(agent):
    """Create a small population."""
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    result = [agent]
    for i in range(4):
        g = new_genome_graph(cfg, rng)
        a = new_agent(f"probe-{i}", g, initial_energy=5000.0)
        d_in = len(g.input_nodes) * 64
        inp = rng.standard_normal(d_in).astype(np.float32)
        out = g.forward(inp)
        a.last_action = out[:256]
        a.last_message = out[256:384]
        a.last_state = out[384:512]
        a.fitness = float(i * 10)
        a.exploration_bonus = float(i) * 0.5
        result.append(a)
    return result


class TestOODProbe:
    def test_returns_result(self, agent):
        rng = np.random.default_rng(42)
        result = probe_ood(agent, rng, n_ticks=10, n_cells=64)
        assert isinstance(result, OODResult)
        assert result.kl_ratio >= 0.0

    def test_no_genome(self):
        a = Agent(id="no-genome")
        result = probe_ood(a, np.random.default_rng(42))
        assert result.kl_ratio == 0.0


class TestModularityProbe:
    def test_returns_result(self, agent):
        result = probe_modularity(agent.genome)
        assert isinstance(result, ModularityResult)
        assert -0.5 <= result.q_score <= 1.0
        assert result.n_nodes > 0

    def test_empty_genome(self):
        g = GenomeGraph()
        result = probe_modularity(g)
        assert result.q_score == 0.0


class TestMultiscaleProbe:
    def test_returns_result(self, agent):
        rng = np.random.default_rng(42)
        result = probe_multiscale(agent, rng, horizons=[1, 4], n_ticks=20, n_cells=64)
        assert isinstance(result, MultiScaleResult)
        assert result.consistency_ratio >= 1.0

    def test_kls_populated(self, agent):
        rng = np.random.default_rng(42)
        result = probe_multiscale(agent, rng, horizons=[1], n_ticks=10, n_cells=64)
        assert 1 in result.kl_by_horizon


class TestMultiLLMProbe:
    def test_returns_result(self, agent):
        rng = np.random.default_rng(42)
        result = probe_multi_llm(agent, rng, n_ticks=10, n_cells=64)
        assert isinstance(result, MultiLLMResult)
        assert result.kl_ratio >= 0.0
        assert result.adaptation_speed >= 0

    def test_env_kls_populated(self, agent):
        rng = np.random.default_rng(42)
        result = probe_multi_llm(agent, rng, n_ticks=10, n_cells=64)
        assert len(result.env_kls) == 4  # Rule110, Rule30, Rule90, Mixed


class TestCommunicationProbe:
    def test_returns_result(self, agents):
        result = probe_communication(agents)
        assert isinstance(result, CommunicationResult)
        assert result.mutual_info >= 0.0
        assert result.n_agents == len(agents)

    def test_empty_population(self):
        result = probe_communication([])
        assert result.mutual_info == 0.0
        assert result.n_agents == 0

    def test_msg_norm_reported(self, agents):
        result = probe_communication(agents)
        assert result.msg_norm_mean >= 0.0


class TestSelfModProbe:
    def test_returns_result(self, agents):
        result = probe_selfmod(agents)
        assert isinstance(result, SelfModResult)
        assert 0.0 <= result.survival_rate <= 1.0

    def test_no_selfmod_attempts(self, agents):
        for a in agents:
            a.selfmod_count = 0
        result = probe_selfmod(agents)
        assert result.total_attempts == 0

    def test_with_selfmod_attempts(self, agents):
        agents[0].selfmod_count = 5
        agents[0].selfmod_survived = 2
        result = probe_selfmod(agents)
        assert result.total_attempts == 5
        assert result.total_survived == 2
        assert result.survival_rate == pytest.approx(0.4)


class TestExplorationProbe:
    def test_returns_result(self, agents):
        result = probe_exploration(agents)
        assert isinstance(result, ExplorationResult)
        assert 0.0 <= result.exploration_ratio <= 1.0

    def test_no_explorers(self, agents):
        for a in agents:
            a.exploration_bonus = 0.0
        result = probe_exploration(agents)
        assert result.exploration_ratio == 0.0

    def test_all_explorers(self, agents):
        for a in agents:
            a.exploration_bonus = 1.0
        result = probe_exploration(agents)
        assert result.exploration_ratio == 1.0


class TestProbeRunner:
    def test_run_all_probes(self, agents):
        rng = np.random.default_rng(42)
        report = run_all_probes(agents, rng, generation=0, preset_name="test", quick=True)
        assert isinstance(report, ProbeReport)
        assert report.generation == 0
        assert report.alive_count > 0

    def test_report_to_dict(self, agents):
        rng = np.random.default_rng(42)
        report = run_all_probes(agents, rng, generation=0, quick=True)
        d = report.to_dict()
        assert "generation" in d
        assert "ood_kl_ratio" in d
        assert "modularity_q" in d

    def test_report_summary(self, agents):
        rng = np.random.default_rng(42)
        report = run_all_probes(agents, rng, generation=0, quick=True)
        summary = report.summary()
        assert "Probe Report" in summary
        assert "OOD" in summary
        assert "Modularity" in summary

    def test_save_probe_report(self, agents, tmp_path):
        rng = np.random.default_rng(42)
        report = run_all_probes(agents, rng, generation=0, preset_name="test", quick=True)
        path = save_probe_report(report, directory=tmp_path)
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["generation"] == 0

    def test_empty_population(self):
        rng = np.random.default_rng(42)
        report = run_all_probes([], rng, generation=0, quick=True)
        assert report.alive_count == 0
        assert report.mean_fitness == 0.0

    def test_tier_assessment(self, agents):
        rng = np.random.default_rng(42)
        report = run_all_probes(agents, rng, generation=0, quick=True)
        summary = report.summary()
        assert "Tier:" in summary
