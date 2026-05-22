"""Communication Emergence Probe — measure if agents transmit meaningful information.

Computes mutual information I(message; action) across agents. High MI (> 0.5 bits)
indicates messages encode behavioral information (communication emerged).
MI ≈ 0 indicates random noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CommunicationResult:
    mutual_info: float  # I(msg; action) in bits
    msg_action_corr: float  # mean absolute correlation
    n_agents: int
    msg_norm_mean: float
    msg_norm_std: float


def _discretize(vec: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Discretize continuous vector into bins."""
    # Normalize to [0, 1]
    vmin, vmax = vec.min(), vec.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vec, dtype=int)
    normalized = (vec - vmin) / (vmax - vmin)
    return np.clip((normalized * n_bins).astype(int), 0, n_bins - 1)


def _mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 8) -> float:
    """Estimate MI between two discrete arrays using histogram method."""
    x_disc = _discretize(x, n_bins)
    y_disc = _discretize(y, n_bins)

    n = len(x_disc)
    if n == 0:
        return 0.0

    # Joint distribution
    joint = np.zeros((n_bins, n_bins))
    for i in range(n):
        joint[x_disc[i], y_disc[i]] += 1
    joint /= n

    # Marginals
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 1e-12 and px[i] > 1e-12 and py[j] > 1e-12:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))

    return max(0.0, float(mi))


def probe_communication(agents: list) -> CommunicationResult:
    """Measure communication emergence across a population of agents."""
    alive = [a for a in agents if a.is_alive and a.last_action is not None and a.last_message is not None]

    if not alive:
        return CommunicationResult(mutual_info=0.0, msg_action_corr=0.0, n_agents=0, msg_norm_mean=0.0, msg_norm_std=0.0)

    # Collect message and action vectors
    messages = np.stack([a.last_message for a in alive])
    actions = np.stack([a.last_action for a in alive])

    # MI between message and action (across agents)
    # Reduce each to 1D by taking norm per segment
    msg_1d = np.linalg.norm(messages.reshape(len(alive), -1, 16), axis=2).flatten()
    act_1d = np.linalg.norm(actions.reshape(len(alive), -1, 16), axis=2).flatten()

    min_len = min(len(msg_1d), len(act_1d))
    mi = _mutual_information(msg_1d[:min_len], act_1d[:min_len])

    # Correlation between message and action norms
    msg_norms = np.linalg.norm(messages, axis=1)
    act_norms = np.linalg.norm(actions, axis=1)
    if len(msg_norms) >= 2:
        corr = float(np.corrcoef(msg_norms, act_norms)[0, 1])
        corr = abs(corr) if not np.isnan(corr) else 0.0
    else:
        corr = 0.0

    return CommunicationResult(
        mutual_info=mi,
        msg_action_corr=corr,
        n_agents=len(alive),
        msg_norm_mean=float(np.mean(msg_norms)),
        msg_norm_std=float(np.std(msg_norms)),
    )
