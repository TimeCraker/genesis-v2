"""Cognitive probes — 7 + 1 probes to distinguish understanding from memorization.

Probes:
1. OOD Generalization — robustness to unseen environment rules
2. Topology Modularity — functional specialization in genome graph
3. Multi-scale Prediction — consistency across prediction horizons
4. Multi-LLM Adaptability — cross-environment KL ratio + adaptation speed
5. Communication Emergence — mutual information I(msg; action)
6. Self-modification Efficiency — survival rate + fitness improvement
7. Exploration Effectiveness — exploration bonus vs fitness correlation
8. Conversation Quality — response diversity + semantic similarity + coherence
"""

from genesis_v2.metrics.probes.conversation import ConversationProbeResult, run_conversation_probes
from genesis_v2.metrics.probes.exploration_effect import ExplorationResult, probe_exploration
from genesis_v2.metrics.probes.modularity import ModularityResult, probe_modularity
from genesis_v2.metrics.probes.multi_llm import MultiLLMResult, probe_multi_llm
from genesis_v2.metrics.probes.multiscale import MultiScaleResult, probe_multiscale
from genesis_v2.metrics.probes.ood import OODResult, probe_ood
from genesis_v2.metrics.probes.runner import ProbeReport, run_all_probes, save_probe_report
from genesis_v2.metrics.probes.self_mod import SelfModResult, probe_selfmod

__all__ = [
    "OODResult", "probe_ood",
    "ModularityResult", "probe_modularity",
    "MultiScaleResult", "probe_multiscale",
    "MultiLLMResult", "probe_multi_llm",
    "SelfModResult", "probe_selfmod",
    "ExplorationResult", "probe_exploration",
    "ConversationProbeResult", "run_conversation_probes",
    "ProbeReport", "run_all_probes", "save_probe_report",
]
