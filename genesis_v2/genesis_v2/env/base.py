"""Environment Protocol — the universe's interface to agents."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Environment(Protocol):
    """All environments (Mock, MultiLLM, Batched) implement this interface."""

    def observe(self) -> np.ndarray:
        """Current environment state → Agent input vector."""
        ...

    def interact(self, action: np.ndarray) -> np.ndarray:
        """Agent action → environment feedback."""
        ...

    def true_distribution(self, history: np.ndarray) -> np.ndarray:
        """Given history, return the universe's true next-step distribution."""
        ...

    def close(self) -> None: ...
