"""Breeding strategy — Crossover 50% + Clone 30% + Migration 20%.

Replaces the simple clone+mutate breeding from Phase 0/1 with a mixed
strategy that combines NEAT crossover, elite cloning, and cross-island
migration.
"""

from __future__ import annotations

import math

import numpy as np

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import GenesisConfig
from genesis_v2.genome.crossover import crossover
from genesis_v2.genome.mutate import mutate
from genesis_v2.evolution.gen_memory import GenerationalMemoryBank


def breed_generation_v2(
    agents: list[Agent],
    rng: np.random.Generator,
    cfg: GenesisConfig,
    gen_memory: GenerationalMemoryBank | None = None,
    top_fraction: float = 0.25,
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.50,
    clone_rate: float = 0.30,
    # migration_rate is implicitly 1.0 - crossover_rate - clone_rate
) -> list[Agent]:
    """Breed next generation using mixed strategy.

    Strategy:
    - Top elites are preserved directly (top_fraction)
    - Remaining slots filled by:
      - crossover_rate: NEAT crossover of two elite parents + mutation
      - clone_rate: clone single elite + mutations
      - (1 - crossover_rate - clone_rate): clone single elite + more mutations

    Also applies generational memory to offspring if gen_memory provided.
    """
    alive = [a for a in agents if a.is_alive]
    if not alive:
        return _create_fresh_from_config(cfg, rng, len(agents))

    # Record generation memory before breeding
    if gen_memory is not None:
        gen = max(a.generation for a in alive)
        gen_memory.record_generation(alive, gen)

    k = max(1, int(math.ceil(len(alive) * top_fraction)))
    elites = sorted(alive, key=lambda a: a.fitness, reverse=True)[:k]

    target_size = len(agents)
    next_gen: list[Agent] = []

    # 1. Preserve top elites directly
    for i, elite in enumerate(elites):
        child = new_agent(
            id=f"g{elite.generation + 1}-elite-{i}",
            genome=elite.genome.copy(),
            initial_energy=cfg.physics.initial_energy,
            generation=elite.generation + 1,
            island_id=elite.island_id,
        )
        child.genome.reset_state()
        next_gen.append(child)

    # 2. Fill remaining slots with mixed strategy
    while len(next_gen) < target_size:
        roll = rng.random()

        if roll < crossover_rate and len(elites) >= 2:
            # --- Crossover ---
            parent_a, parent_b = _pick_two(elites, rng)
            child_genome = crossover(
                parent_a.genome, parent_b.genome,
                rng=rng,
                fitness_a=parent_a.fitness,
                fitness_b=parent_b.fitness,
            )
            # Apply 1 mutation after crossover
            mutate(child_genome, rng)

            child = new_agent(
                id=f"g{max(parent_a.generation, parent_b.generation) + 1}-xover-{len(next_gen)}",
                genome=child_genome,
                initial_energy=cfg.physics.initial_energy,
                generation=max(parent_a.generation, parent_b.generation) + 1,
                island_id=parent_a.island_id,
            )

        elif roll < crossover_rate + clone_rate:
            # --- Clone ---
            parent = elites[int(rng.integers(len(elites)))]
            child_genome = parent.genome.copy()
            n_mutations = max(1, int(round(rng.exponential(max(mutation_rate, 0.01)))))
            for _ in range(n_mutations):
                mutate(child_genome, rng)

            child = new_agent(
                id=f"g{parent.generation + 1}-clone-{len(next_gen)}",
                genome=child_genome,
                initial_energy=cfg.physics.initial_energy,
                generation=parent.generation + 1,
                island_id=parent.island_id,
            )
        else:
            # --- Exploration clone (more mutations) ---
            parent = elites[int(rng.integers(len(elites)))]
            child_genome = parent.genome.copy()
            n_mutations = max(2, int(round(rng.exponential(max(mutation_rate * 2, 0.02)))))
            for _ in range(n_mutations):
                mutate(child_genome, rng)

            child = new_agent(
                id=f"g{parent.generation + 1}-explore-{len(next_gen)}",
                genome=child_genome,
                initial_energy=cfg.physics.initial_energy,
                generation=parent.generation + 1,
                island_id=parent.island_id,
            )

        # Apply generational memory
        if gen_memory is not None:
            gen_memory.apply_to_offspring(child, rng)

        next_gen.append(child)

    return next_gen


def _pick_two(pool: list[Agent], rng: np.random.Generator) -> tuple[Agent, Agent]:
    """Pick two distinct agents from pool."""
    if len(pool) < 2:
        return pool[0], pool[0]
    idx = rng.choice(len(pool), size=2, replace=False)
    return pool[int(idx[0])], pool[int(idx[1])]


def _create_fresh_from_config(
    cfg: GenesisConfig,
    rng: np.random.Generator,
    n: int,
) -> list[Agent]:
    """Create fresh population when all agents are dead."""
    from genesis_v2.genome.graph import GraphConfig, new_genome_graph

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
    for i in range(n):
        g = new_genome_graph(genome_cfg, rng)
        a = new_agent(
            id=f"g0-fresh-{i}",
            genome=g,
            initial_energy=cfg.physics.initial_energy,
            generation=0,
        )
        agents.append(a)
    return agents
