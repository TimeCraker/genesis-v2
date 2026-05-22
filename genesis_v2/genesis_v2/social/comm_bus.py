"""CommunicationBus — grid-topology vector-space messaging between agents.

Agents are placed on a 2D grid. Messages (128-dim vectors) are delivered
to all neighbors within L2 distance <= comm_radius.
"""

from __future__ import annotations

import math

import numpy as np

from genesis_v2.agent.agent import Agent
from genesis_v2.genome.graph import D_MESSAGE


class CommunicationBus:
    def __init__(
        self,
        grid_rows: int = 10,
        grid_cols: int = 10,
        comm_radius: int = 2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.comm_radius = comm_radius
        self._rng = rng or np.random.default_rng(0)
        self.agent_positions: dict[str, tuple[int, int]] = {}
        self._inboxes: dict[str, list[np.ndarray]] = {}

    def assign_positions(self, agent_ids: list[str]) -> None:
        """Randomly place agents on grid positions."""
        positions = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                positions.append((r, c))
        self._rng.shuffle(positions)
        self.agent_positions.clear()
        for i, aid in enumerate(agent_ids):
            self.agent_positions[aid] = positions[i % len(positions)]
        self._inboxes = {aid: [] for aid in agent_ids}

    def _neighbors(self, agent_id: str) -> list[str]:
        """Find all agents within comm_radius (L2 distance)."""
        pos = self.agent_positions.get(agent_id)
        if pos is None:
            return []
        r0, c0 = pos
        neighbors = []
        for aid, (r1, c1) in self.agent_positions.items():
            if aid == agent_id:
                continue
            dist = math.sqrt((r0 - r1) ** 2 + (c0 - c1) ** 2)
            if dist <= self.comm_radius:
                neighbors.append(aid)
        return neighbors

    def deliver(self, sender_id: str, message_vec: np.ndarray) -> int:
        """Deliver message to all neighbors. Returns count of recipients."""
        neighbors = self._neighbors(sender_id)
        for nid in neighbors:
            if nid in self._inboxes:
                self._inboxes[nid].append(message_vec.copy())
        return len(neighbors)

    def get_inbox(self, agent_id: str) -> np.ndarray:
        """Mean-pool all received messages. Returns D_MESSAGE-dim vector."""
        messages = self._inboxes.get(agent_id, [])
        if not messages:
            return np.zeros(D_MESSAGE, dtype=np.float32)
        return np.mean(np.stack(messages, axis=0), axis=0).astype(np.float32)

    def get_inbox_raw(self, agent_id: str) -> list[np.ndarray]:
        """Get raw (non-pooled) inbox messages."""
        return self._inboxes.get(agent_id, [])

    def clear_all(self) -> None:
        """Reset all inboxes (call at end of each tick)."""
        for aid in self._inboxes:
            self._inboxes[aid] = []

    def message_count(self, agent_id: str) -> int:
        """Number of messages received this tick."""
        return len(self._inboxes.get(agent_id, []))
