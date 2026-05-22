"""Multi-LLM translator — bidirectional mapping between agent vectors and natural language.

Pipeline:
  Agent → Text:  action_vec (256) → pseudo-prompt → LLM → response text
  Text → Agent:  text → embedding (hash or API) → P^T → feedback_vec (input_dim)

Uses FrozenEmbeddingAtlas for the projection matrix P.
Supports mock mode (no API key needed) for testing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from genesis_v2.env.embed import FrozenEmbeddingAtlas
from genesis_v2.genome.graph import D_ACTION, D_NODE


def _text_to_seed(text: str) -> int:
    """Deterministic seed from text hash."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


class Translator:
    """Bidirectional translator between agent vector space and natural language.

    In mock mode (default), translation is deterministic but lossy — it uses
    hash-based embeddings rather than real LLM calls. This is suitable for
    testing the architecture without API costs.

    With a real LLM backend, the translator calls the API to generate text
    from agent signals and to embed human text.
    """

    def __init__(
        self,
        atlas: FrozenEmbeddingAtlas | None = None,
        backend_name: str = "mock",
        n_input_nodes: int = 8,
        d_node: int = D_NODE,
    ) -> None:
        self.atlas = atlas or FrozenEmbeddingAtlas()
        self.backend_name = backend_name
        self.n_input_nodes = n_input_nodes
        self.d_node = d_node
        self.d_in = n_input_nodes * d_node

        # LLM API config (only used when backend_name != "mock")
        self._api_base_url: str = ""
        self._api_model: str = ""
        self._api_key: str = ""
        self._api_timeout: float = 30.0
        self._api_max_tokens: int = 128

    def configure_api(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 30.0,
        max_tokens: int = 128,
    ) -> None:
        """Configure real LLM API for non-mock translation."""
        self._api_base_url = base_url
        self._api_model = model
        self._api_key = api_key
        self._api_timeout = timeout
        self._api_max_tokens = max_tokens

    # ---------- Agent → Text ----------

    def vec_to_text(self, action_vec: np.ndarray) -> str:
        """Translate agent action vector to natural language text.

        Mock mode: deterministic pseudo-text derived from action vector hash.
        API mode: embed action → construct prompt → call LLM.
        """
        if self.backend_name == "mock" or not self._api_key:
            return self._vec_to_text_mock(action_vec)
        return self._vec_to_text_api(action_vec)

    def _vec_to_text_mock(self, action_vec: np.ndarray) -> str:
        """Mock translation: action vector → deterministic pseudo-text."""
        # Use the action vector to seed a deterministic "translation"
        vec_hash = hashlib.sha256(action_vec.tobytes()).hexdigest()[:8]
        norm = float(np.linalg.norm(action_vec))
        mean = float(np.mean(action_vec))
        std = float(np.std(action_vec))

        # Generate a pseudo-text that encodes the vector's statistical properties
        words = []
        # Map vector segments to word-like tokens
        segments = np.array_split(action_vec[:64], 8)
        for i, seg in enumerate(segments):
            seg_mean = float(np.mean(seg))
            if seg_mean > 0.5:
                words.append(["alpha", "bright", "forward", "high", "rise", "peak", "strong", "up"][i % 8])
            elif seg_mean < -0.5:
                words.append(["beta", "dark", "backward", "low", "fall", "valley", "weak", "down"][i % 8])
            else:
                words.append(["gamma", "neutral", "still", "mid", "flat", "plain", "calm", "center"][i % 8])

        return " ".join(words) + f" [{vec_hash}]"

    def _vec_to_text_api(self, action_vec: np.ndarray) -> str:
        """Real LLM translation: embed action → prompt → API call."""
        import httpx

        # Map action to a conceptual prompt via embedding
        act_embed = self.atlas.embed_action(action_vec)
        signal_strength = float(np.linalg.norm(action_vec))
        signal_direction = "positive" if float(np.mean(action_vec)) > 0 else "negative"

        prompt = (
            f"A digital organism has produced an output signal. "
            f"Signal strength: {signal_strength:.4f}, direction: {signal_direction}. "
            f"Describe what the organism might be trying to communicate in one sentence."
        )

        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self._api_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self._api_max_tokens,
                "temperature": 0.0,
            }
            with httpx.Client(timeout=self._api_timeout) as client:
                resp = client.post(
                    f"{self._api_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return self._vec_to_text_mock(action_vec)

    # ---------- Text → Agent ----------

    def text_to_vec(self, text: str) -> np.ndarray:
        """Translate natural language text to agent input vector.

        Mock mode: deterministic hash-based embedding → P^T projection.
        API mode: call embedding API → P^T projection.
        """
        if self.backend_name == "mock" or not self._api_key:
            return self._text_to_vec_mock(text)
        return self._text_to_vec_api(text)

    def _text_to_vec_mock(self, text: str) -> np.ndarray:
        """Mock translation: text → hash-based embedding → projection."""
        seed = _text_to_seed(text)
        rng = np.random.default_rng(seed)
        # Create a pseudo-embedding in embedding space
        embed = rng.standard_normal(self.atlas.d_embed).astype(np.float32)
        embed = embed / (np.linalg.norm(embed) + 1e-8)
        # Project to feedback space via P^T
        feedback = self.atlas.decode_feedback(embed)
        # Pad/trim to d_in
        result = np.zeros(self.d_in, dtype=np.float32)
        copy_len = min(len(feedback), self.d_in)
        result[:copy_len] = feedback[:copy_len]
        return result

    def _text_to_vec_api(self, text: str) -> np.ndarray:
        """Real LLM translation: text → embedding API → projection."""
        # For now, fall back to mock (embedding APIs require separate setup)
        return self._text_to_vec_mock(text)

    # ---------- Conversation helpers ----------

    def translate_agent_output(
        self,
        action_vec: np.ndarray,
        message_vec: np.ndarray | None = None,
    ) -> str:
        """Translate full agent output (action + optional message) to text."""
        text = self.vec_to_text(action_vec)
        if message_vec is not None:
            msg_norm = float(np.linalg.norm(message_vec))
            if msg_norm > 1.0:
                msg_text = self.vec_to_text(message_vec)
                text = f"{text} (signal: {msg_text})"
        return text

    def translate_to_input(self, text: str, history: list[str] | None = None) -> np.ndarray:
        """Translate human text (+ optional conversation history) to agent input."""
        # Combine current text with recent history
        if history:
            combined = " ".join(history[-3:] + [text])
        else:
            combined = text
        return self.text_to_vec(combined)


@dataclass
class ConversationTurn:
    """One turn in a human-agent conversation."""

    turn_number: int
    human_input: str
    agent_input_vec: np.ndarray | None = None
    agent_output_vec: np.ndarray | None = None
    agent_response: str = ""
    agent_energy: float = 0.0
    agent_fitness: float = 0.0


@dataclass
class ConversationSession:
    """Multi-turn conversation between a human and an agent.

    The session manages:
    1. Translating human text to agent input vectors
    2. Running agent forward pass
    3. Translating agent output back to text
    4. Maintaining conversation history
    """

    agent: object  # Agent
    translator: Translator
    history: list[ConversationTurn] = field(default_factory=list)
    max_turns: int = 20

    def send(self, human_text: str) -> ConversationTurn:
        """Send a message to the agent and get a response.

        Returns a ConversationTurn with both the raw vectors and translated text.
        """
        turn_num = len(self.history) + 1

        # Build input for agent
        prev_texts = [t.human_input for t in self.history[-3:]]
        agent_input = self.translator.translate_to_input(human_text, history=prev_texts)

        # Agent forward pass
        if self.agent.genome is None:
            raise ValueError("Agent has no genome")

        output = self.agent.genome.forward(agent_input)

        from genesis_v2.agent.agent import split_output
        action, message, state, selfmod = split_output(output)

        # Update agent state
        self.agent.last_action = action
        self.agent.last_message = message
        self.agent.state_buffer = self.agent.last_state
        self.agent.last_state = state
        self.agent.last_selfmod = selfmod

        # Translate response
        response_text = self.translator.translate_agent_output(action, message)

        turn = ConversationTurn(
            turn_number=turn_num,
            human_input=human_text,
            agent_input_vec=agent_input.copy(),
            agent_output_vec=output.copy(),
            agent_response=response_text,
            agent_energy=self.agent.energy,
            agent_fitness=self.agent.fitness,
        )
        self.history.append(turn)
        return turn

    def get_conversation_text(self) -> str:
        """Get full conversation as formatted text."""
        lines = []
        for turn in self.history:
            lines.append(f"[Turn {turn.turn_number}]")
            lines.append(f"  Human: {turn.human_input}")
            lines.append(f"  Agent: {turn.agent_response}")
            lines.append(f"  (energy={turn.agent_energy:.1f}, fitness={turn.agent_fitness:.2f})")
            lines.append("")
        return "\n".join(lines)
