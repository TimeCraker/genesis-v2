"""Multi-scale Prediction Consistency Probe — test world model vs one-step fitting.

Measures KL at multiple prediction horizons {1, 4, 16}. A true world model
should maintain consistent KL across scales (ratio < 3.0). A one-step fitter
will degrade rapidly (ratio > 20.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis_v2.engine.metabolism import kl_action_truth


@dataclass
class MultiScaleResult:
    kl_by_horizon: dict[int, float]  # horizon → mean KL
    consistency_ratio: float  # max(kl) / min(kl) — lower = more consistent


def probe_multiscale(
    agent,
    rng: np.random.Generator,
    horizons: list[int] | None = None,
    n_ticks: int = 100,
    n_cells: int = 64,
) -> MultiScaleResult:
    """Measure prediction accuracy at multiple time horizons."""
    if horizons is None:
        horizons = [1, 4, 16]

    if agent.genome is None:
        return MultiScaleResult(kl_by_horizon={}, consistency_ratio=0.0)

    from genesis_v2.env.mock import MockMathEnvironment

    d_in = len(agent.genome.input_nodes) * n_cells
    kl_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}

    env = MockMathEnvironment(n_cells=n_cells, rng=np.random.default_rng(rng.integers(2**31)))

    # Collect predictions at each horizon
    action_history: list[np.ndarray] = []

    for t in range(n_ticks + max(horizons)):
        obs = env.observe()
        inp = np.tile(obs, len(agent.genome.input_nodes)).astype(np.float32)
        if len(inp) > d_in:
            inp = inp[:d_in]
        elif len(inp) < d_in:
            inp = np.pad(inp, (0, d_in - len(inp)))

        out = agent.genome.forward(inp)
        action = out[:256]
        action_history.append(action.copy())

        feedback = env.interact(action[:64].copy())
        truth = env.true_distribution(feedback.astype(np.float32))

        # Measure KL at each horizon (comparing action at time t-h with truth at time t)
        for h in horizons:
            if t >= h:
                past_action = action_history[t - h]
                min_len = min(len(past_action), len(truth))
                kl = kl_action_truth(past_action[:min_len], truth[:min_len])
                kl_by_horizon[h].append(kl)

    agent.genome.reset_state()

    mean_kls = {h: float(np.mean(v)) if v else 0.0 for h, v in kl_by_horizon.items()}
    vals = [v for v in mean_kls.values() if v > 0]
    if len(vals) >= 2:
        ratio = max(vals) / max(min(vals), 1e-9)
    else:
        ratio = 1.0

    return MultiScaleResult(kl_by_horizon=mean_kls, consistency_ratio=ratio)
