"""BatchedEnvironment — wraps MultiLLMEnvironment with micro-batching.

Collects agent actions into batches of `batch_size`, sends them as
a single API call (concatenated prompts), then splits responses.
"""

from __future__ import annotations

import numpy as np

from genesis_v2.env.multi_llm import MultiLLMEnvironment


class BatchedEnvironment:
    """Micro-batch wrapper around MultiLLMEnvironment."""

    def __init__(
        self,
        llm_env: MultiLLMEnvironment,
        batch_size: int = 16,
    ) -> None:
        self.llm_env = llm_env
        self.batch_size = batch_size
        self._pending_actions: list[np.ndarray] = []
        self._last_feedback: np.ndarray | None = None

    def observe(self) -> np.ndarray:
        return self.llm_env.observe()

    def interact(self, action: np.ndarray) -> np.ndarray:
        """Add action to batch; when full, send all at once."""
        self._pending_actions.append(action.copy())

        if len(self._pending_actions) >= self.batch_size:
            self._flush_batch()

        if self._last_feedback is not None:
            return self._last_feedback.copy()
        return np.zeros(self.llm_env.n_cells, dtype=np.float32)

    def _flush_batch(self) -> None:
        """Send accumulated batch to LLM."""
        if not self._pending_actions:
            return
        # Use mean action as representative
        mean_action = np.mean(
            np.stack(self._pending_actions, axis=0), axis=0
        ).astype(np.float32)
        self._last_feedback = self.llm_env.interact(mean_action)
        self._pending_actions.clear()

    def flush(self) -> None:
        """Force-send any remaining pending actions."""
        if self._pending_actions:
            self._flush_batch()

    def true_distribution(self, history: np.ndarray) -> np.ndarray:
        return self.llm_env.true_distribution(history)

    @property
    def last_cost(self) -> float:
        return self.llm_env.last_cost

    def close(self) -> None:
        self.flush()
        self.llm_env.close()
