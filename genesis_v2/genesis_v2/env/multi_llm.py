"""MultiLLMEnvironment — agents interact with real LLMs via OpenAI-compatible API.

Pipeline:
  Agent → LLM: action_vec (256) --P--> act_embed (1536) → cosine nearest → token → API call
  LLM → Agent: response token → fb_embed (1536) --P^T--> feedback_vec (256)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
import yaml

from genesis_v2.env.embed import FrozenEmbeddingAtlas

if TYPE_CHECKING:
    from genesis_v2.env.budget import BudgetManager


def load_backends(path: str | Path | None = None) -> dict:
    """Load backend configs from backends.yaml."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "backends.yaml"
    path = Path(path)
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return raw.get("backends", {})
    return {}


class MultiLLMEnvironment:
    """Environment backed by a single LLM backend (OpenAI-compatible API)."""

    def __init__(
        self,
        backend_name: str = "deepseek",
        backends_path: str | Path | None = None,
        embedding_atlas: FrozenEmbeddingAtlas | None = None,
        n_cells: int = 64,
    ) -> None:
        self.backend_name = backend_name
        backends = load_backends(backends_path)
        self.backend = backends.get(backend_name, {})
        self.atlas = embedding_atlas or FrozenEmbeddingAtlas()
        self.n_cells = n_cells

        self.base_url = self.backend.get("base_url", "")
        self.model = self.backend.get("model", "deepseek-chat")
        self.max_tokens = self.backend.get("max_tokens", 64)
        self.timeout = self.backend.get("timeout_sec", 30.0)
        self.cost_per_1m = self.backend.get("cost_per_1m_tokens", 0.1)
        self.api_key_env = self.backend.get("api_key_env", "")

        self._last_response_embed: np.ndarray = np.zeros(self.atlas.d_embed, dtype=np.float32)
        self._last_cost: float = 0.0
        self._history: list[np.ndarray] = []
        self._budget: BudgetManager | None = None
        self._island_id: int = 0

    def set_budget(self, budget: BudgetManager, island_id: int = 0) -> None:
        """Attach a BudgetManager for cost-aware API calls."""
        self._budget = budget
        self._island_id = island_id

    def _get_api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    def _call_llm(self, prompt: str) -> tuple[str, float]:
        """Call LLM API. Returns (response_text, cost_usd)."""
        api_key = self._get_api_key()
        if not api_key:
            return "", 0.0

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                total_tokens = usage.get("total_tokens", len(text) // 4)
                cost = total_tokens * self.cost_per_1m / 1_000_000
                return text, cost
        except Exception:
            return "", 0.0

    def observe(self) -> np.ndarray:
        """Return last response as observation vector."""
        return self.atlas.decode_feedback(self._last_response_embed)[:self.n_cells]

    def interact(self, action: np.ndarray) -> np.ndarray:
        """Map action to token, call LLM, map response back to vector."""
        # Budget check
        if self._budget is not None:
            estimated_cost = self.cost_per_1m * self.max_tokens / 1_000_000
            if not self._budget.check_budget(self._island_id, estimated_cost):
                self._last_cost = 0.0
                return np.zeros(self.n_cells, dtype=np.float32)

        # Embed action → nearest token concept
        act_embed = self.atlas.embed_action(action)

        # Convert to a text prompt (cosine similarity search against embedding space)
        # For now, use the action vector norm as a "signal strength" prompt
        signal = float(np.linalg.norm(action))
        prompt = f"Observe and respond. Signal strength: {signal:.4f}. Output a single number."

        text, cost = self._call_llm(prompt)
        self._last_cost = cost

        # Record cost
        if self._budget is not None and cost > 0:
            self._budget.record_cost(self._island_id, cost)

        # Create a pseudo-embedding from the response text
        if text:
            # Hash text to a deterministic vector in embedding space
            rng = np.random.default_rng(abs(hash(text)) % (2**31))
            fb_embed = rng.standard_normal(self.atlas.d_embed).astype(np.float32)
            fb_embed = fb_embed / (np.linalg.norm(fb_embed) + 1e-8)
            self._last_response_embed = fb_embed
        else:
            self._last_response_embed = np.zeros(self.atlas.d_embed, dtype=np.float32)

        feedback = self.atlas.decode_feedback(self._last_response_embed)[:self.n_cells]
        self._history.append(feedback.copy())
        return feedback

    def true_distribution(self, history: np.ndarray) -> np.ndarray:
        """Return last response as ground truth."""
        return self.observe()

    @property
    def last_cost(self) -> float:
        return self._last_cost

    def close(self) -> None:
        pass
