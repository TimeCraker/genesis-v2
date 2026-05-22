"""Pydantic configuration models for Genesis v2."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class PhysicsConfig(BaseModel):
    alpha: float = 0.01       # token cost
    beta: float = 0.005       # node cost
    gamma: float = 0.001      # latency cost
    delta: float = 0.002      # edge cost
    epsilon: float = 0.1      # API cost multiplier
    zeta: float = 0.01        # message cost

    w_pred: float = 1.0
    w_comp: float = 0.5
    w_bvar: float = 0.3
    w_adapt: float = 0.5
    w_social: float = 0.3
    w_explore: float = 0.2

    death_penalty: float = 500.0
    initial_energy: float = 5000.0
    topology_entropy_threshold: float = 5.0

    selfmod_energy_threshold: float = 10000.0
    selfmod_energy_cost: float = 1000.0
    selfmod_death_rate: float = 0.7


class EvolutionConfig(BaseModel):
    tick_rate: int = 2
    generation_ticks: int = 200
    migration_interval_generations: int = 50


class IslandConfig(BaseModel):
    name: str
    size: int = 100
    mutation_rate: float = 0.15
    backend: str = "mock"
    allowed_mutations: list[str] | None = None


class PopulationConfig(BaseModel):
    islands: list[IslandConfig] = []


class GenomeYaml(BaseModel):
    node_dim: int = 64
    input_nodes: int = 8
    output_nodes_action: int = 4
    output_nodes_message: int = 2
    output_nodes_state: int = 2
    output_nodes_selfmod: int = 1
    initial_hidden_nodes: int = 4
    initial_edge_density: float = 0.2


class EnvironmentConfig(BaseModel):
    type: str = "mock"
    embed_dim: int = 1536
    projection_seed: int = 42
    batch_size: int = 16
    batch_wait_ms: int = 100
    context_window: int = 64
    temperature: float = 0.0
    top_p: float = 0.0
    comm_radius: int = 2
    grid_rows: int = 10
    grid_cols: int = 10
    total_budget_usd: float = 50.0
    per_island_budget_usd: float = 15.0
    fallback_to_mock: bool = True


class EvaluationConfig(BaseModel):
    probe_interval_generations: int = 10
    ood_prompt_file: str = "./data/ood_prompts.txt"
    modularity_algo: str = "louvain"
    multiscale_horizons: list[int] = [1, 4, 16]


class GenesisConfig(BaseModel):
    physics: PhysicsConfig = PhysicsConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    population: PopulationConfig = PopulationConfig()
    genome: GenomeYaml = GenomeYaml()
    environment: EnvironmentConfig = EnvironmentConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def load_config(path: str | Path | None = None) -> GenesisConfig:
    """Load config from YAML file, falling back to defaults."""
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "genesis_v2.yaml"
    path = Path(path)
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        return GenesisConfig.model_validate(raw)
    return GenesisConfig()
