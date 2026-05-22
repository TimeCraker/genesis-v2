"""Elite seed bank — persist top agents to disk for future breeding.

Agents are saved as JSON to data/survivors/ directory.
Each file is named: {agent_id}_gen{generation}_fit{fitness:.2f}.json
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis_v2.agent.agent import Agent

SURVIVORS_DIR = Path(__file__).parent.parent.parent / "data" / "survivors"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_agent(agent: Agent, directory: Path | None = None) -> Path:
    """Save an agent's genome + metadata to disk. Returns the file path."""
    directory = directory or SURVIVORS_DIR
    _ensure_dir(directory)

    payload = agent.to_payload()
    fname = f"{agent.id}_gen{agent.generation}_fit{agent.fitness:.2f}.json"
    fpath = directory / fname

    # Convert numpy arrays in payload to lists for JSON serialization
    fpath.write_text(json.dumps(payload, indent=2, default=_json_default))
    return fpath


def load_agent(fpath: Path | str) -> Agent:
    """Load an agent from a survivor file."""
    payload = json.loads(Path(fpath).read_text())
    return Agent.from_payload(payload)


def list_survivors(directory: Path | None = None) -> list[Path]:
    """List all survivor files, sorted by fitness (highest first)."""
    directory = directory or SURVIVORS_DIR
    if not directory.exists():
        return []
    files = list(directory.glob("*.json"))
    # Sort by fitness in filename
    def _fitness_key(p: Path) -> float:
        try:
            return float(p.stem.split("_fit")[-1])
        except (IndexError, ValueError):
            return 0.0
    return sorted(files, key=_fitness_key, reverse=True)


def load_top_survivors(n: int = 5, directory: Path | None = None) -> list[Agent]:
    """Load top N survivors from disk."""
    files = list_survivors(directory)[:n]
    return [load_agent(f) for f in files]


def auto_save_elites(agents: list[Agent], top_fraction: float = 0.1, directory: Path | None = None) -> int:
    """Save top fraction of agents (by fitness) to survivor bank. Returns count saved."""
    if not agents:
        return 0
    alive = [a for a in agents if a.is_alive]
    if not alive:
        return 0
    n = max(1, int(len(alive) * top_fraction))
    elites = sorted(alive, key=lambda a: a.fitness, reverse=True)[:n]
    for a in elites:
        save_agent(a, directory)
    return len(elites)


def auto_load_seeds(n_agents: int, directory: Path | None = None) -> list[Agent]:
    """Load survivors as seeds. Returns list (may be shorter than n_agents)."""
    survivors = load_top_survivors(n=n_agents, directory=directory)
    # Reset state for fresh start
    for a in survivors:
        a.energy = 0.0  # will be re-initialized by caller
        a.fitness = 0.0
        a.tick_alive = 0
        a.is_alive = True
        a.genome.reset_state()
        for attr in ("_prev_kl", "_prev_comp", "_death_reason"):
            if hasattr(a, attr):
                delattr(a, attr)
    return survivors


def _json_default(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
