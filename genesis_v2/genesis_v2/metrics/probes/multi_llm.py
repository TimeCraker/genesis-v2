"""Multi-LLM Adaptability Probe — test generalization across different environments.

Measures KL ratio between different CA rule environments and how quickly
the agent adapts after a rule switch. Low ratio (< 2.0) and fast adaptation
(< 50 ticks) = general understanding. High ratio (> 10.0) = memorization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis_v2.engine.metabolism import kl_action_truth
from genesis_v2.env.mock import MockMathEnvironment, RULE30, RULE90, RULE110


@dataclass
class MultiLLMResult:
    kl_ratio: float  # max_env_kl / min_env_kl across rule sets
    adaptation_speed: int  # ticks to recover 50% of KL gap after rule switch
    env_kls: dict[str, float]  # rule_name → mean KL


def _run_env(agent, rules, rng, n_ticks, n_cells, d_in) -> float:
    """Run agent in a specific rule environment and return mean KL."""
    env = MockMathEnvironment(
        n_cells=n_cells,
        rng=np.random.default_rng(rng.integers(2**31)),
        rules=rules,
    )
    kls = []
    for _ in range(n_ticks):
        obs = env.observe()
        inp = np.tile(obs, len(agent.genome.input_nodes)).astype(np.float32)
        if len(inp) > d_in:
            inp = inp[:d_in]
        elif len(inp) < d_in:
            inp = np.pad(inp, (0, d_in - len(inp)))

        out = agent.genome.forward(inp)
        action = out[:256]
        feedback = env.interact(action[:64].copy())
        truth = env.true_distribution(feedback.astype(np.float32))
        min_len = min(len(action), len(truth))
        kls.append(kl_action_truth(action[:min_len], truth[:min_len]))

    agent.genome.reset_state()
    return float(np.mean(kls)) if kls else 0.0


def _measure_adaptation(agent, rng, n_cells, d_in, switch_ticks=50) -> int:
    """Measure ticks to adapt after a rule switch."""
    # Start with Rule110
    env = MockMathEnvironment(
        n_cells=n_cells,
        rng=np.random.default_rng(rng.integers(2**31)),
        rules=[RULE110],
    )

    # Run 20 ticks to stabilize
    for _ in range(20):
        obs = env.observe()
        inp = np.tile(obs, len(agent.genome.input_nodes)).astype(np.float32)
        if len(inp) > d_in:
            inp = inp[:d_in]
        elif len(inp) < d_in:
            inp = np.pad(inp, (0, d_in - len(inp)))
        out = agent.genome.forward(inp)
        env.interact(out[:64].copy())

    # Measure baseline KL
    baseline_kls = []
    for _ in range(10):
        obs = env.observe()
        inp = np.tile(obs, len(agent.genome.input_nodes)).astype(np.float32)
        if len(inp) > d_in:
            inp = inp[:d_in]
        elif len(inp) < d_in:
            inp = np.pad(inp, (0, d_in - len(inp)))
        out = agent.genome.forward(inp)
        action = out[:256]
        feedback = env.interact(action[:64].copy())
        truth = env.true_distribution(feedback.astype(np.float32))
        min_len = min(len(action), len(truth))
        baseline_kls.append(kl_action_truth(action[:min_len], truth[:min_len]))

    baseline_kl = float(np.mean(baseline_kls)) if baseline_kls else 0.0

    # Switch to Rule30 (very different)
    env._rules = [RULE30]
    env._current_rule_idx = 0

    # Measure post-switch KL and find adaptation point
    spike_kl = 0.0
    for t in range(switch_ticks):
        obs = env.observe()
        inp = np.tile(obs, len(agent.genome.input_nodes)).astype(np.float32)
        if len(inp) > d_in:
            inp = inp[:d_in]
        elif len(inp) < d_in:
            inp = np.pad(inp, (0, d_in - len(inp)))
        out = agent.genome.forward(inp)
        action = out[:256]
        feedback = env.interact(action[:64].copy())
        truth = env.true_distribution(feedback.astype(np.float32))
        min_len = min(len(action), len(truth))
        kl = kl_action_truth(action[:min_len], truth[:min_len])

        if t == 0:
            spike_kl = kl
        # Check if KL dropped back to 50% of the spike
        target = baseline_kl + 0.5 * (spike_kl - baseline_kl)
        if kl <= target and t > 0:
            agent.genome.reset_state()
            return t

    agent.genome.reset_state()
    return switch_ticks  # didn't adapt within window


def probe_multi_llm(
    agent,
    rng: np.random.Generator,
    n_ticks: int = 50,
    n_cells: int = 64,
) -> MultiLLMResult:
    """Measure cross-environment adaptability."""
    if agent.genome is None:
        return MultiLLMResult(kl_ratio=0.0, adaptation_speed=0, env_kls={})

    d_in = len(agent.genome.input_nodes) * n_cells

    env_configs = {
        "Rule110": [RULE110],
        "Rule30": [RULE30],
        "Rule90": [RULE90],
        "Mixed": [RULE110, RULE30, RULE90],
    }

    env_kls = {}
    for name, rules in env_configs.items():
        env_kls[name] = _run_env(agent, rules, rng, n_ticks, n_cells, d_in)

    vals = [v for v in env_kls.values() if v > 0]
    kl_ratio = max(vals) / max(min(vals), 1e-9) if len(vals) >= 2 else 1.0

    adapt_speed = _measure_adaptation(agent, rng, n_cells, d_in)

    return MultiLLMResult(
        kl_ratio=float(kl_ratio),
        adaptation_speed=adapt_speed,
        env_kls=env_kls,
    )
