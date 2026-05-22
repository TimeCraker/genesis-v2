"""Conversation quality probes — evaluate agent's ability to converse.

Probes:
1. Response diversity — entropy of response text (non-degenerate output)
2. Semantic similarity — cosine similarity between prompt and response vectors
3. Multi-turn coherence — topic consistency across turns
4. Cross-LLM consistency — same prompt, different translation → similar responses
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis_v2.genome.graph import D_OUT


@dataclass
class ConversationProbeResult:
    """Results from conversation quality probes."""

    response_diversity: float = 0.0       # higher = more diverse responses
    semantic_similarity: float = 0.0      # prompt-response cosine similarity
    multi_turn_coherence: float = 0.0     # topic consistency (0-1)
    cross_llm_consistency: float = 0.0    # same-prompt consistency (0-1)
    n_turns: int = 0
    n_unique_responses: int = 0


def response_diversity(responses: list[str]) -> float:
    """Measure diversity of agent responses (Shannon entropy of word distribution).

    Returns 0.0 if all responses are identical, higher for more varied output.
    """
    if not responses:
        return 0.0

    # Build word frequency distribution
    word_counts: dict[str, int] = {}
    total_words = 0
    for resp in responses:
        words = resp.lower().split()
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
            total_words += 1

    if total_words == 0:
        return 0.0

    # Shannon entropy
    entropy = 0.0
    for count in word_counts.values():
        p = count / total_words
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)


def semantic_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Returns value in [-1, 1]. Higher = more similar.
    """
    a = vec_a.astype(np.float64).flatten()
    b = vec_b.astype(np.float64).flatten()
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def multi_turn_coherence(
    responses: list[str],
    window: int = 3,
) -> float:
    """Measure topic consistency across conversation turns.

    Compares consecutive windows of responses using word overlap (Jaccard).
    Returns average Jaccard similarity (0-1). Higher = more coherent.
    """
    if len(responses) < 2:
        return 1.0

    similarities = []
    for i in range(len(responses) - 1):
        words_a = set(responses[i].lower().split())
        words_b = set(responses[i + 1].lower().split())
        if not words_a and not words_b:
            similarities.append(1.0)
        elif not words_a or not words_b:
            similarities.append(0.0)
        else:
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            similarities.append(intersection / union if union > 0 else 0.0)

    return float(np.mean(similarities))


def cross_llm_consistency(
    agent,
    translator,
    prompts: list[str],
    n_translations: int = 3,
) -> float:
    """Measure consistency of agent responses under different input translations.

    For each prompt, generates n_translations slightly different input vectors
    (by perturbing the text), then measures how similar the agent's output
    vectors are. Higher = more consistent = more robust.

    Returns average cosine similarity of outputs for same prompt.
    """
    from genesis_v2.translation.translator import Translator

    consistencies = []

    for prompt in prompts:
        # Generate variant translations of the same prompt
        outputs = []
        for i in range(n_translations):
            variant = f"{prompt} [{i}]"  # slight perturbation
            inp = translator.text_to_vec(variant)
            out = agent.genome.forward(inp)
            outputs.append(out)

        # Compute pairwise cosine similarities
        if len(outputs) < 2:
            continue

        pairs = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = semantic_similarity(outputs[i], outputs[j])
                pairs.append(sim)

        consistencies.append(float(np.mean(pairs)))

    return float(np.mean(consistencies)) if consistencies else 0.0


def run_conversation_probes(
    agent,
    translator,
    test_prompts: list[str] | None = None,
) -> ConversationProbeResult:
    """Run all conversation quality probes on an agent.

    Args:
        agent: The agent to probe.
        translator: Translator instance.
        test_prompts: Prompts to use (defaults to built-in set).

    Returns:
        ConversationProbeResult with all probe scores.
    """
    if test_prompts is None:
        test_prompts = [
            "Hello, what are you?",
            "Describe your environment.",
            "What do you predict will happen next?",
            "Tell me about your neighbors.",
            "How do you feel about your energy level?",
            "What patterns do you see?",
            "Can you explain your behavior?",
            "What would you change about yourself?",
        ]

    from genesis_v2.translation.translator import ConversationSession

    # Run conversation
    session = ConversationSession(agent=agent, translator=translator)
    responses = []
    response_vecs = []
    prompt_vecs = []

    for prompt in test_prompts[:8]:
        turn = session.send(prompt)
        responses.append(turn.agent_response)
        if turn.agent_output_vec is not None:
            response_vecs.append(turn.agent_output_vec)
        prompt_vecs.append(turn.agent_input_vec)

    # Compute probes
    div = response_diversity(responses)

    # Semantic similarity: average prompt-response cosine
    sims = []
    for i in range(min(len(prompt_vecs), len(response_vecs))):
        sims.append(semantic_similarity(prompt_vecs[i], response_vecs[i]))
    sem_sim = float(np.mean(sims)) if sims else 0.0

    coherence = multi_turn_coherence(responses)
    consistency = cross_llm_consistency(agent, translator, test_prompts[:3])

    n_unique = len(set(responses))

    return ConversationProbeResult(
        response_diversity=div,
        semantic_similarity=sem_sim,
        multi_turn_coherence=coherence,
        cross_llm_consistency=consistency,
        n_turns=len(responses),
        n_unique_responses=n_unique,
    )
