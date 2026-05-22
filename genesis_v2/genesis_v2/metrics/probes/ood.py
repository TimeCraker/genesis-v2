"""OOD Generalization Probe — test robustness to unseen environment rules.

Swap the CA rule set, run the agent in the new environment, and measure
how much KL diverges from the training environment. Low ratio = generalization.
High ratio = memorization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis_v2.engine.metabolism import kl_action_truth
from genesis_v2.env.mock import MockMathEnvironment, RULE30, RULE90, RULE110


@dataclass
class OODResult:
    train_kl: float
    ood_kl: float
    kl_ratio: float  # ood_kl / train_kl (>10 = memorization, <2 = generalization)


def probe_ood(
    agent,
    rng: np.random.Generator,
    n_ticks: int = 50,
    n_cells: int = 64,
) -> OODResult:
    """Run agent in training env and OOD env, compare KL divergence."""
    if agent.genome is None:
        return OODResult(train_kl=0.0, ood_kl=0.0, kl_ratio=0.0)

    d_in = len(agent.genome.input_nodes) * n_cells

    # Training env: standard 3-rule rotation
    train_env = MockMathEnvironment(n_cells=n_cells, rng=np.random.default_rng(rng.integers(2**31)))
    train_kls = _run_and_measure_kl(agent, train_env, rng, n_ticks, d_in)
    agent.genome.reset_state()

    # OOD env: different rule set (Rule30 only — chaotic, very different)
    ood_env = MockMathEnvironment(
        n_cells=n_cells,
        rng=np.random.default_rng(rng.integers(2**31)),
        rules=[RULE30],
    )
    ood_kls = _run_and_measure_kl(agent, ood_env, rng, n_ticks, d_in)
    agent.genome.reset_state()

    train_kl = float(np.mean(train_kls)) if train_kls else 0.0
    ood_kl = float(np.mean(ood_kls)) if ood_kls else 0.0
    ratio = ood_kl / max(train_kl, 1e-9)

    return OODResult(train_kl=train_kl, ood_kl=ood_kl, kl_ratio=ratio)


def _run_and_measure_kl(
    agent,
    env,
    rng: np.random.Generator,
    n_ticks: int,
    d_in: int,
) -> list[float]:
    """Run agent in env and collect per-tick KL values."""
    kls = []
    for _ in range(n_ticks):
        obs = env.observe()
        inp = np.tile(obs, len(agent.genome.input_nodes)).astype(np.float32)
        if len(inp) > d_in:
            inp = inp[:d_in]
        elif len(inp) < d_in:
            inp = np.pad(inp, (0, d_in - len(inp)))

        out = agent.genome.forward(inp)
        action = out[:256]  # action partition
        feedback = env.interact(action[:64].copy())
        truth = env.true_distribution(feedback.astype(np.float32))

        min_len = min(len(action), len(truth))
        kl = kl_action_truth(action[:min_len], truth[:min_len])
        kls.append(kl)

    return kls
