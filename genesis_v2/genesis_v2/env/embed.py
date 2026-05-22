"""Frozen Embedding Atlas — projection matrix P for agent↔LLM vector mapping.

P ∈ R^{D_action × D_embed} is created once with a fixed seed and never trained.
  Agent → LLM: action_vec (256) --P--> act_embed (1536) → cosine nearest → token
  LLM → Agent: token → fb_embed (1536) --P^T--> feedback_vec (256)
"""

from __future__ import annotations

import numpy as np

from genesis_v2.genome.graph import D_ACTION


class FrozenEmbeddingAtlas:
    def __init__(
        self,
        d_action: int = D_ACTION,
        d_embed: int = 1536,
        seed: int = 42,
    ) -> None:
        self.d_action = d_action
        self.d_embed = d_embed
        rng = np.random.default_rng(seed)
        # P: (d_action, d_embed) — maps action space to embedding space
        self.P = rng.standard_normal((d_action, d_embed)).astype(np.float32)
        # Normalize rows for stable cosine similarity
        norms = np.linalg.norm(self.P, axis=1, keepdims=True)
        self.P = self.P / np.maximum(norms, 1e-8)

    def embed_action(self, action_vec: np.ndarray) -> np.ndarray:
        """Map action vector (256) to embedding space (1536)."""
        return (action_vec[:self.d_action] @ self.P).astype(np.float32)

    def decode_feedback(self, embed_vec: np.ndarray) -> np.ndarray:
        """Map embedding vector (1536) back to feedback space (256)."""
        return (embed_vec @ self.P.T).astype(np.float32)

    def nearest_token(
        self,
        embed_vec: np.ndarray,
        token_embeddings: np.ndarray,
    ) -> int:
        """Find nearest token by cosine similarity.

        Args:
            embed_vec: (d_embed,) query vector
            token_embeddings: (vocab_size, d_embed) matrix of all token embeddings

        Returns:
            Token index with highest cosine similarity.
        """
        # Normalize query
        q_norm = np.linalg.norm(embed_vec)
        if q_norm < 1e-8:
            return 0
        q = embed_vec / q_norm
        # Normalize all token embeddings
        t_norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        t_normed = token_embeddings / np.maximum(t_norms, 1e-8)
        # Cosine similarity
        sims = t_normed @ q
        return int(np.argmax(sims))
