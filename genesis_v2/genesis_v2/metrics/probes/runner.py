"""Probe runner — orchestrates all 7+1 probes and produces a unified report.

Runs on a population of agents after each generation, collecting all probe
scores into a single ProbeReport that can be stored and compared across
ablation presets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from genesis_v2.metrics.probes.communication import CommunicationResult, probe_communication
from genesis_v2.metrics.probes.exploration_effect import ExplorationResult, probe_exploration
from genesis_v2.metrics.probes.modularity import ModularityResult, probe_modularity
from genesis_v2.metrics.probes.multi_llm import MultiLLMResult, probe_multi_llm
from genesis_v2.metrics.probes.multiscale import MultiScaleResult, probe_multiscale
from genesis_v2.metrics.probes.ood import OODResult, probe_ood
from genesis_v2.metrics.probes.self_mod import SelfModResult, probe_selfmod


@dataclass
class ProbeReport:
    """Unified report from all probes at one generation."""

    generation: int = 0
    preset_name: str = ""

    # Probe 1: OOD Generalization
    ood_kl_ratio: float = 0.0
    ood_train_kl: float = 0.0
    ood_ood_kl: float = 0.0

    # Probe 2: Topology Modularity
    modularity_q: float = 0.0
    modularity_communities: int = 0

    # Probe 3: Multi-scale Prediction
    multiscale_consistency: float = 0.0
    multiscale_kls: dict = field(default_factory=dict)

    # Probe 4: Multi-LLM Adaptability
    multi_llm_kl_ratio: float = 0.0
    multi_llm_adaptation_speed: int = 0

    # Probe 5: Communication Emergence
    comm_mutual_info: float = 0.0
    comm_msg_action_corr: float = 0.0
    comm_msg_norm_mean: float = 0.0

    # Probe 6: Self-modification Efficiency
    selfmod_survival_rate: float = 0.0
    selfmod_fitness_delta: float = 0.0
    selfmod_total_attempts: int = 0

    # Probe 7: Exploration Effectiveness
    explore_ratio: float = 0.0
    explore_fitness_delta: float = 0.0
    explore_mean_bonus: float = 0.0

    # Population stats
    alive_count: int = 0
    mean_fitness: float = 0.0
    best_fitness: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for JSON/DuckDB storage."""
        return {
            "generation": self.generation,
            "preset_name": self.preset_name,
            "ood_kl_ratio": round(self.ood_kl_ratio, 4),
            "modularity_q": round(self.modularity_q, 4),
            "multiscale_consistency": round(self.multiscale_consistency, 4),
            "multi_llm_kl_ratio": round(self.multi_llm_kl_ratio, 4),
            "multi_llm_adaptation_speed": self.multi_llm_adaptation_speed,
            "comm_mutual_info": round(self.comm_mutual_info, 4),
            "selfmod_survival_rate": round(self.selfmod_survival_rate, 4),
            "selfmod_fitness_delta": round(self.selfmod_fitness_delta, 2),
            "explore_ratio": round(self.explore_ratio, 4),
            "explore_fitness_delta": round(self.explore_fitness_delta, 2),
            "alive_count": self.alive_count,
            "mean_fitness": round(self.mean_fitness, 2),
            "best_fitness": round(self.best_fitness, 2),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Probe Report (gen={self.generation}, preset={self.preset_name}) ===",
            f"Population: {self.alive_count} alive, best={self.best_fitness:.2f}, mean={self.mean_fitness:.2f}",
            "",
            f"1. OOD Generalization:    kl_ratio={self.ood_kl_ratio:.2f}  (<2 = generalize, >10 = memorize)",
            f"2. Topology Modularity:   Q={self.modularity_q:.3f}  (>0.4 = structured, <0.2 = random)",
            f"3. Multi-scale Consist.:  ratio={self.multiscale_consistency:.2f}  (<3 = world model, >20 = fitter)",
            f"4. Multi-LLM Adapt:       kl_ratio={self.multi_llm_kl_ratio:.2f}, adapt={self.multi_llm_adaptation_speed} ticks",
            f"5. Communication:         MI={self.comm_mutual_info:.3f} bits, corr={self.comm_msg_action_corr:.3f}",
            f"6. Self-mod Efficiency:   survival={self.selfmod_survival_rate:.1%}, fit_delta={self.selfmod_fitness_delta:.2f}",
            f"7. Exploration:           ratio={self.explore_ratio:.1%}, fit_delta={self.explore_fitness_delta:.2f}",
        ]

        # Tier assessment
        passes = sum([
            self.ood_kl_ratio < 2.0,
            self.modularity_q > 0.4,
            self.multiscale_consistency < 3.0,
            self.multi_llm_kl_ratio < 2.0,
            self.comm_mutual_info > 0.5,
            self.selfmod_survival_rate > 0.3,
            self.explore_ratio > 0.1 and self.explore_fitness_delta > 0,
        ])
        lines.append(f"\nProbes passed: {passes}/7")
        if passes >= 4:
            lines.append("Tier: A-tier or higher (structural emergence)")
        elif passes >= 2:
            lines.append("Tier: B-tier (evolution signals)")
        else:
            lines.append("Tier: C-tier (system runs, limited emergence)")

        return "\n".join(lines)


def run_all_probes(
    agents: list,
    rng: np.random.Generator,
    generation: int = 0,
    preset_name: str = "",
    n_cells: int = 64,
    quick: bool = False,
) -> ProbeReport:
    """Run all probes on the current population.

    Args:
        agents: All agents (alive + dead).
        rng: Random generator.
        generation: Current generation number.
        preset_name: Name of ablation preset.
        n_cells: Environment cell count.
        quick: If True, use fewer ticks for faster execution.
    """
    alive = [a for a in agents if a.is_alive]
    n_ticks = 20 if quick else 50

    report = ProbeReport(
        generation=generation,
        preset_name=preset_name,
        alive_count=len(alive),
        mean_fitness=float(np.mean([a.fitness for a in alive])) if alive else 0.0,
        best_fitness=float(max(a.fitness for a in alive)) if alive else 0.0,
    )

    if not alive:
        return report

    # Pick best agent for single-agent probes
    best = max(alive, key=lambda a: a.fitness)

    # Probe 1: OOD
    try:
        ood = probe_ood(best, rng, n_ticks=n_ticks, n_cells=n_cells)
        report.ood_kl_ratio = ood.kl_ratio
        report.ood_train_kl = ood.train_kl
        report.ood_ood_kl = ood.ood_kl
    except Exception:
        pass

    # Probe 2: Modularity
    try:
        mod = probe_modularity(best.genome)
        report.modularity_q = mod.q_score
        report.modularity_communities = mod.n_communities
    except Exception:
        pass

    # Probe 3: Multi-scale
    try:
        ms = probe_multiscale(best, rng, n_ticks=n_ticks, n_cells=n_cells)
        report.multiscale_consistency = ms.consistency_ratio
        report.multiscale_kls = ms.kl_by_horizon
    except Exception:
        pass

    # Probe 4: Multi-LLM
    try:
        mllm = probe_multi_llm(best, rng, n_ticks=n_ticks, n_cells=n_cells)
        report.multi_llm_kl_ratio = mllm.kl_ratio
        report.multi_llm_adaptation_speed = mllm.adaptation_speed
    except Exception:
        pass

    # Probe 5: Communication
    try:
        comm = probe_communication(alive)
        report.comm_mutual_info = comm.mutual_info
        report.comm_msg_action_corr = comm.msg_action_corr
        report.comm_msg_norm_mean = comm.msg_norm_mean
    except Exception:
        pass

    # Probe 6: Self-modification
    try:
        sm = probe_selfmod(agents)
        report.selfmod_survival_rate = sm.survival_rate
        report.selfmod_fitness_delta = sm.fitness_delta
        report.selfmod_total_attempts = sm.total_attempts
    except Exception:
        pass

    # Probe 7: Exploration
    try:
        exp = probe_exploration(alive)
        report.explore_ratio = exp.exploration_ratio
        report.explore_fitness_delta = exp.fitness_delta
        report.explore_mean_bonus = exp.mean_exploration_bonus
    except Exception:
        pass

    return report


def save_probe_report(report: ProbeReport, directory: Path | None = None) -> Path:
    """Save probe report to JSON."""
    directory = directory or Path("data/probes")
    directory.mkdir(parents=True, exist_ok=True)
    fname = f"probes_{report.preset_name}_gen{report.generation:04d}.json"
    path = directory / fname
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path
