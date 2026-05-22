"""Island — bundles agents, environment, comm_bus, and budget for one island.

Each island represents an isolated sub-population potentially bound to a
different LLM backend. Agents within an island communicate via CommunicationBus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import GenesisConfig, IslandConfig
from genesis_v2.env.budget import BudgetManager
from genesis_v2.env.mock import MockMathEnvironment
from genesis_v2.env.multi_llm import MultiLLMEnvironment
from genesis_v2.genome.graph import GraphConfig, new_genome_graph
from genesis_v2.social.comm_bus import CommunicationBus


@dataclass
class Island:
    """One isolated sub-population with its own environment and social bus."""

    id: int
    name: str
    agents: list[Agent]
    env: object  # Environment protocol (Mock or MultiLLM)
    comm_bus: CommunicationBus
    budget: BudgetManager | None
    island_cfg: IslandConfig
    is_mock: bool = True


def create_island(
    island_id: int,
    island_cfg: IslandConfig,
    cfg: GenesisConfig,
    rng: np.random.Generator,
    budget: BudgetManager | None = None,
    social: bool = True,
) -> Island:
    """Create a single island with agents, environment, and comm_bus."""
    # Environment
    backend = island_cfg.backend.lower()
    is_mock = backend == "mock"

    if is_mock:
        env = MockMathEnvironment(
            n_cells=cfg.genome.node_dim,
            rng=np.random.default_rng(rng.integers(2**31)),
        )
    else:
        env = MultiLLMEnvironment(
            backend_name=backend,
            n_cells=cfg.genome.node_dim,
        )

    # Communication bus
    if social:
        comm_bus = CommunicationBus(
            grid_rows=cfg.environment.grid_rows,
            grid_cols=cfg.environment.grid_cols,
            comm_radius=cfg.environment.comm_radius,
            rng=np.random.default_rng(rng.integers(2**31)),
        )
    else:
        comm_bus = CommunicationBus(
            grid_rows=cfg.environment.grid_rows,
            grid_cols=cfg.environment.grid_cols,
            comm_radius=0,  # no neighbors = no communication
            rng=np.random.default_rng(rng.integers(2**31)),
        )

    # Create agents
    genome_cfg = GraphConfig(
        node_dim=cfg.genome.node_dim,
        input_nodes=cfg.genome.input_nodes,
        output_nodes_action=cfg.genome.output_nodes_action,
        output_nodes_message=cfg.genome.output_nodes_message,
        output_nodes_state=cfg.genome.output_nodes_state,
        output_nodes_selfmod=cfg.genome.output_nodes_selfmod,
        initial_hidden_nodes=cfg.genome.initial_hidden_nodes,
        initial_edge_density=cfg.genome.initial_edge_density,
    )

    agents = []
    for i in range(island_cfg.size):
        g = new_genome_graph(genome_cfg, rng)
        a = new_agent(
            id=f"isle{island_id}-g0-{i}",
            genome=g,
            initial_energy=cfg.physics.initial_energy,
            generation=0,
            island_id=island_id,
        )
        agents.append(a)

    # Assign agents to grid positions
    comm_bus.assign_positions([a.id for a in agents])

    return Island(
        id=island_id,
        name=island_cfg.name,
        agents=agents,
        env=env,
        comm_bus=comm_bus,
        budget=budget,
        island_cfg=island_cfg,
        is_mock=is_mock,
    )


def create_islands(
    cfg: GenesisConfig,
    rng: np.random.Generator,
    social: bool = True,
    budget: BudgetManager | None = None,
) -> list[Island]:
    """Create all islands from config.

    If no islands are configured, creates a single default island on Mock.
    """
    island_configs = cfg.population.islands
    if not island_configs:
        island_configs = [IslandConfig(name="Default", size=10, backend="mock")]

    # Create shared budget if not provided
    if budget is None:
        budget = BudgetManager(
            total_budget=cfg.environment.total_budget_usd,
            per_island_budget=cfg.environment.per_island_budget_usd,
            fallback_to_mock=cfg.environment.fallback_to_mock,
        )

    islands = []
    for i, isl_cfg in enumerate(island_configs):
        isl = create_island(
            island_id=i,
            island_cfg=isl_cfg,
            cfg=cfg,
            rng=rng,
            budget=budget,
            social=social,
        )
        islands.append(isl)

    return islands
