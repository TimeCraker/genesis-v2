"""MockMathEnvironment v2 — multi-rule CA (Rule110 / Rule30 / Rule90)."""

from __future__ import annotations

import numpy as np


def _apply_rule(tape: np.ndarray, rule: int) -> np.ndarray:
    """Apply a Wolfram elementary CA rule to a binary tape."""
    L = len(tape)
    nxt = np.empty_like(tape)
    for i in range(L):
        l = int(tape[(i - 1) % L])
        c = int(tape[i])
        r = int(tape[(i + 1) % L])
        idx = (l << 2) | (c << 1) | r
        nxt[i] = (rule >> idx) & 1
    return nxt


# Standard rules
RULE110 = 0b01101110  # 110 — Turing-complete
RULE30 = 0b00011110   # 30 — chaotic
RULE90 = 0b01011010   # 90 — linear (XOR)


class MockMathEnvironment:
    """Multi-rule CA environment; observation = first n_cells as float32."""

    def __init__(
        self,
        n_cells: int = 64,
        ring_size: int = 128,
        rng: np.random.Generator | None = None,
        rules: list[int] | None = None,
    ) -> None:
        self.n_cells = n_cells
        self.ring_size = ring_size
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._rules = rules or [RULE110, RULE30, RULE90]
        self._current_rule_idx = 0
        self._tape = self._rng.integers(0, 2, size=ring_size, dtype=np.int8)
        self._ticks = 0
        self._history: list[np.ndarray] = []

    @property
    def _current_rule(self) -> int:
        return self._rules[self._current_rule_idx % len(self._rules)]

    def _encode(self, tape: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_cells, dtype=np.float32)
        n = min(self.n_cells, len(tape))
        out[:n] = tape[:n].astype(np.float32) * 2.0 - 1.0  # map {0,1} → {-1,1}
        return out

    def observe(self) -> np.ndarray:
        return self._encode(self._tape)

    def interact(self, action: np.ndarray) -> np.ndarray:
        """Advance CA; optional weak coupling from action."""
        # Mild coupling: flip one cell if |mean(action)| > 0.5
        if float(np.mean(np.abs(action))) > 0.5:
            idx = int(np.abs(action[0] * 1000.0)) % self.ring_size
            self._tape[idx] ^= 1

        self._tape = _apply_rule(self._tape, self._current_rule)
        self._ticks += 1

        # Rotate rule every 50 ticks for multi-rule pressure
        if self._ticks % 50 == 0:
            self._current_rule_idx += 1

        obs = self._encode(self._tape)
        self._history.append(obs.copy())
        return obs

    def true_distribution(self, history: np.ndarray) -> np.ndarray:
        """One-step-ahead distribution before interact mutates state."""
        nxt = _apply_rule(self._tape, self._current_rule)
        return self._encode(nxt)

    def close(self) -> None:
        pass
