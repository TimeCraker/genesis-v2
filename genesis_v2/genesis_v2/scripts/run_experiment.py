"""Experiment runner with multi-island generation loop, social layer, and auto-survivor save/load."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import GenesisConfig, load_config
from genesis_v2.engine.reaper import sweep_island
from genesis_v2.engine.tick import island_step_sync, multi_island_step
from genesis_v2.env.mock import MockMathEnvironment
from genesis_v2.evolution.breeder import breed_generation_v2
from genesis_v2.evolution.gen_memory import GenerationalMemoryBank
from genesis_v2.evolution.survivor_bank import (
    SURVIVORS_DIR, auto_load_seeds, auto_save_elites, load_agent,
)
from genesis_v2.genome.graph import GraphConfig, new_genome_graph
from genesis_v2.population.island import Island, create_island, create_islands
from genesis_v2.population.migration import MigrationTracker, migrate_agents
from genesis_v2.storage.duckdb_store import DuckDBStore

DATA_DIR = Path(__file__).parent.parent.parent / "data"
STATUS_FILE = DATA_DIR / "experiment_status.json"
ENV_KEYS_FILE = DATA_DIR / "api_keys.env"
REPORT_DIR = Path(__file__).parent.parent.parent.parent / "docx" / "experiment_list"


def _load_api_keys_to_env() -> None:
    """Load persisted API keys into os.environ (safety net for subprocess)."""
    import os
    if not ENV_KEYS_FILE.exists():
        return
    for line in ENV_KEYS_FILE.read_text("utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            if v:
                os.environ[k.strip()] = v.strip()


def _write_status(status: dict) -> None:
    status["pid"] = os.getpid()
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=_json_default))


def _json_default(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _create_fresh_population(
    cfg: GenesisConfig,
    rng: np.random.Generator,
    n: int,
) -> list[Agent]:
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
            id=f"g0-agent-{i}",
            genome=g,
            initial_energy=cfg.physics.initial_energy,
            generation=0,
        )
        agents.append(a)
    return agents


def _load_specific_seeds(seed_paths: list[str]) -> list[Agent]:
    """Load specific survivor files by path, reset their state for fresh run."""
    agents = []
    for p in seed_paths:
        try:
            a = load_agent(p)
            a.energy = 0.0
            a.fitness = 0.0
            a.tick_alive = 0
            a.is_alive = True
            a.genome.reset_state()
            for attr in ("_prev_kl", "_prev_comp", "_death_reason"):
                if hasattr(a, attr):
                    delattr(a, attr)
            agents.append(a)
        except Exception as e:
            print(f"[warn] Failed to load seed {p}: {e}")
    return agents


def _gather_island_stats(islands: list) -> tuple[list[dict], int, float, float, float, float, float]:
    """Gather per-island and aggregate stats from current agent state."""
    all_alive = []
    island_stats = []
    for isl in islands:
        alive = [a for a in isl.agents if a.is_alive]
        all_alive.extend(alive)
        if alive:
            isl_stat = {
                "island": isl.name,
                "alive": len(alive),
                "best_fitness": round(float(max(a.fitness for a in alive)), 2),
                "mean_fitness": round(float(np.mean([a.fitness for a in alive])), 2),
                "mean_energy": round(float(np.mean([a.energy for a in alive])), 1),
            }
        else:
            isl_stat = {"island": isl.name, "alive": 0, "best_fitness": 0, "mean_fitness": 0, "mean_energy": 0}
        island_stats.append(isl_stat)

    alive_count = len(all_alive)
    mean_fitness = float(np.mean([a.fitness for a in all_alive])) if all_alive else 0.0
    best_fitness = float(max(a.fitness for a in all_alive)) if all_alive else 0.0
    mean_energy = float(np.mean([a.energy for a in all_alive])) if all_alive else 0.0
    mean_pred_err = float(np.mean([a.prediction_error for a in all_alive])) if all_alive else 0.0
    mean_comp = float(np.mean([a.compression for a in all_alive])) if all_alive else 0.0
    return island_stats, alive_count, mean_fitness, best_fitness, mean_energy, mean_pred_err, mean_comp


def _build_top_agents(islands: list, n: int = 10) -> list[dict]:
    """Build leaderboard of top-N agents by fitness across all islands."""
    all_alive = [a for isl in islands for a in isl.agents if a.is_alive]
    return _rank_top(all_alive, n)


def _rank_top(agents: list, n: int = 10) -> list[dict]:
    """Rank a flat agent list by fitness and return top-N leaderboard entries."""
    alive = [a for a in agents if a.is_alive]
    if not alive:
        return []
    ranked = sorted(alive, key=lambda a: a.fitness, reverse=True)[:n]
    return [
        {
            "rank": i + 1,
            "id": a.id,
            "fitness": round(float(a.fitness), 2),
            "energy": round(float(a.energy), 1),
            "tick_alive": a.tick_alive,
            "nodes": a.genome.node_count() if a.genome else 0,
            "edges": a.genome.edge_count() if a.genome else 0,
        }
        for i, a in enumerate(ranked)
    ]


def generate_experiment_report(
    history: list[dict],
    elapsed: float,
    total_generations: int,
    ticks_per_gen: int,
    total_agents: int,
    seed: int,
    mode: str,
    social: bool,
    preset_name: str = "",
) -> Path:
    """Generate a concise experiment report and write to docx/experiment_list/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"experiment_{ts}.md"

    lines: list[str] = []
    lines.append(f"# Experiment Report {ts}")
    lines.append("")
    lines.append(f"**Mode**: {mode} | **Seed**: {seed} | **Social**: {'ON' if social else 'OFF'}")
    if preset_name:
        lines.append(f"**Preset**: {preset_name}")
    lines.append(f"**Runtime**: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    lines.append(f"**Scale**: {total_agents} agents x {total_generations} gens x {ticks_per_gen} ticks/gen = {total_generations * ticks_per_gen} total ticks")
    lines.append("")

    if not history:
        lines.append("No generation data recorded.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    # --- Key metrics table ---
    first = history[0]
    last = history[-1]
    mid_idx = len(history) // 2
    mid = history[mid_idx]

    lines.append("## Key Metrics")
    lines.append("")
    lines.append("| Metric | Gen 0 | Gen %d | Gen %d | Trend |" % (mid["generation"], last["generation"]))
    lines.append("|--------|-------|--------|--------|-------")

    def _trend(v0, v1, higher_is_better=True):
        if v0 == 0:
            return "N/A"
        ratio = v1 / v0 if v0 != 0 else 0
        if higher_is_better:
            if ratio > 1.5:
                return f"UP {ratio:.1f}x"
            elif ratio < 0.67:
                return f"DOWN {ratio:.2f}x"
            return "FLAT"
        else:
            if ratio < 0.67:
                return f"DOWN {ratio:.2f}x (good)"
            elif ratio > 1.5:
                return f"UP {ratio:.1f}x (bad)"
            return "FLAT"

    lines.append("| Best Fitness | %.2f | %.2f | %.2f | %s |" % (
        first.get("best_fitness", 0), mid.get("best_fitness", 0),
        last.get("best_fitness", 0), _trend(first.get("best_fitness", 0), last.get("best_fitness", 0))))
    lines.append("| Mean Fitness | %.2f | %.2f | %.2f | %s |" % (
        first.get("mean_fitness", 0), mid.get("mean_fitness", 0),
        last.get("mean_fitness", 0), _trend(first.get("mean_fitness", 0), last.get("mean_fitness", 0))))
    lines.append("| Mean Energy | %.0f | %.0f | %.0f | %s |" % (
        first.get("mean_energy", 0), mid.get("mean_energy", 0),
        last.get("mean_energy", 0), _trend(first.get("mean_energy", 0), last.get("mean_energy", 0), True)))
    lines.append("| Pred Error | %.4f | %.4f | %.4f | %s |" % (
        first.get("mean_pred_err", 0), mid.get("mean_pred_err", 0),
        last.get("mean_pred_err", 0), _trend(first.get("mean_pred_err", 1), last.get("mean_pred_err", 1), False)))
    lines.append("| Alive | %d | %d | %d | %s |" % (
        first.get("alive_count", 0), mid.get("alive_count", 0),
        last.get("alive_count", 0), _trend(first.get("alive_count", 1), last.get("alive_count", 1))))
    lines.append("")

    # --- Per-island breakdown (multi-island mode) ---
    if "islands" in last and last["islands"]:
        lines.append("## Per-Island Results (Final Gen)")
        lines.append("")
        lines.append("| Island | Alive | Best Fitness | Mean Fitness | Mean Energy |")
        lines.append("|--------|-------|-------------|-------------|-------------|")
        for isl in last["islands"]:
            lines.append("| %s | %d | %.2f | %.2f | %.0f |" % (
                isl.get("island", "?"), isl.get("alive", 0),
                isl.get("best_fitness", 0), isl.get("mean_fitness", 0),
                isl.get("mean_energy", 0)))
        lines.append("")

    # --- Best agents ---
    if "top_agents" in last and last["top_agents"]:
        top = last["top_agents"]
        lines.append("## Top Agents (Final Gen)")
        lines.append("")
        lines.append("| Rank | ID | Fitness | Energy | Ticks Alive | Nodes | Edges |")
        lines.append("|------|----|---------|--------|-------------|-------|-------|")
        for a in top[:5]:
            lines.append("| %d | %s | %.2f | %.0f | %d | %d | %d |" % (
                a.get("rank", 0), a.get("id", "?"), a.get("fitness", 0),
                a.get("energy", 0), a.get("tick_alive", 0),
                a.get("nodes", 0), a.get("edges", 0)))
        lines.append("")

    # --- Key conclusions ---
    lines.append("## Conclusions")
    lines.append("")

    best0 = first.get("best_fitness", 0)
    bestN = last.get("best_fitness", 0)
    alive0 = first.get("alive_count", total_agents)
    aliveN = last.get("alive_count", 0)
    pred0 = first.get("mean_pred_err", 0)
    predN = last.get("mean_pred_err", 0)

    if best0 > 0 and bestN / best0 > 3:
        lines.append(f"- **Evolution signal strong**: best fitness {best0:.0f} -> {bestN:.0f} ({bestN/best0:.1f}x growth)")
    elif best0 > 0 and bestN / best0 > 1.5:
        lines.append(f"- **Evolution signal moderate**: best fitness {best0:.0f} -> {bestN:.0f} ({bestN/best0:.1f}x growth)")
    elif best0 > 0:
        lines.append(f"- **Evolution signal weak**: best fitness {best0:.0f} -> {bestN:.0f} ({bestN/best0:.1f}x)")
    else:
        lines.append(f"- **No evolution signal**: best fitness remained at 0")

    survival_rate = aliveN / alive0 if alive0 > 0 else 0
    if survival_rate > 0.8:
        lines.append(f"- **High survival**: {aliveN}/{alive0} ({survival_rate:.0%}) agents alive at end")
    elif survival_rate > 0.4:
        lines.append(f"- **Moderate attrition**: {aliveN}/{alive0} ({survival_rate:.0%}) agents survived")
    else:
        lines.append(f"- **Mass die-off**: {aliveN}/{alive0} ({survival_rate:.0%}) agents survived — check energy/reward tuning")

    if pred0 > 0 and predN > pred0 * 1.2:
        lines.append(f"- **Overfitting signal**: prediction error rose {pred0:.4f} -> {predN:.4f} — agents may be memorizing, not generalizing")
    elif pred0 > 0 and predN < pred0 * 0.8:
        lines.append(f"- **Prediction improving**: error dropped {pred0:.4f} -> {predN:.4f}")

    if "islands" in last and last["islands"]:
        best_isl = max(last["islands"], key=lambda x: x.get("best_fitness", 0))
        worst_isl = min(last["islands"], key=lambda x: x.get("best_fitness", 0) if x.get("alive", 0) > 0 else float("inf"))
        if best_isl.get("best_fitness", 0) > 0:
            lines.append(f"- **Top island**: {best_isl['island']} (best={best_isl['best_fitness']:.0f})")
        if worst_isl.get("best_fitness", 0) == 0 and worst_isl.get("alive", 0) == 0:
            lines.append(f"- **Extinct island**: {worst_isl['island']} — all agents died")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] Saved to {report_path}", flush=True)
    return report_path


def run_experiment(
    config: GenesisConfig | None = None,
    n_agents: int = 10,
    total_generations: int = 5,
    ticks_per_gen: int = 100,
    top_fraction: float = 0.25,
    mutation_rate: float = 0.15,
    seed: int = 42,
    survivors_dir: Path | None = None,
    db_path: Path | None = None,
    social: bool = True,
    multi_island: bool = True,
    preset_name: str = "",
    run_probes: bool = False,
    probe_interval: int = 10,
    seed_paths: list[str] | None = None,
) -> None:
    """Run a full experiment with multi-island generation loop."""
    # Force line-buffered stdout so subprocess logs appear in real-time
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    _load_api_keys_to_env()
    cfg = config or load_config()
    rng = np.random.default_rng(seed)
    survivors_dir = survivors_dir or SURVIVORS_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "experiments").mkdir(parents=True, exist_ok=True)

    db_path = db_path or DATA_DIR / "experiments" / f"run_{int(time.time())}.duckdb"
    store = DuckDBStore(db_path)

    # --- Multi-island mode ---
    if multi_island and cfg.population.islands:
        _run_multi_island(
            cfg, rng, store, total_generations, ticks_per_gen,
            top_fraction, mutation_rate, survivors_dir, seed, social,
            preset_name=preset_name, run_probes=run_probes,
            probe_interval=probe_interval,
            seed_paths=seed_paths,
        )
    else:
        # --- Legacy single-population mode ---
        _run_single_island(
            cfg, rng, store, n_agents, total_generations, ticks_per_gen,
            top_fraction, mutation_rate, survivors_dir, seed, social,
            preset_name=preset_name, run_probes=run_probes,
            probe_interval=probe_interval,
            seed_paths=seed_paths,
        )

    store.close()


def _run_multi_island(
    cfg: GenesisConfig,
    rng: np.random.Generator,
    store: DuckDBStore,
    total_generations: int,
    ticks_per_gen: int,
    top_fraction: float,
    mutation_rate: float,
    survivors_dir: Path,
    seed: int,
    social: bool,
    preset_name: str = "",
    run_probes: bool = False,
    probe_interval: int = 10,
    seed_paths: list[str] | None = None,
) -> None:
    """Run experiment across multiple islands (one per backend)."""
    islands = create_islands(cfg, rng, social=social)

    # Inject selected seeds into first island if provided
    if seed_paths:
        selected = _load_specific_seeds(seed_paths)
        if selected and islands:
            isl = islands[0]
            for i, seed_agent in enumerate(selected):
                if i < len(isl.agents):
                    old_id = isl.agents[i].id
                    seed_agent.id = f"seed-{i}"
                    seed_agent.energy = cfg.physics.initial_energy
                    isl.agents[i] = seed_agent
            print(f"[experiment] Injected {len(selected)} selected seeds into {isl.name}")
    total_agents = sum(len(isl.agents) for isl in islands)
    start_time = time.time()
    global_tick = 0
    history: list[dict] = []

    # Phase 2: gen_memory and migration tracker
    gen_memory = GenerationalMemoryBank()
    migration_tracker = MigrationTracker()
    migration_interval = cfg.evolution.migration_interval_generations

    _write_status({
        "running": True,
        "mode": "multi_island",
        "generation": 0,
        "total_generations": total_generations,
        "tick": 0,
        "alive_count": total_agents,
        "total_agents": total_agents,
        "islands": [isl.name for isl in islands],
        "history": history,
    })

    for gen in range(total_generations):
        # Run ticks across all islands
        for tick_in_gen in range(ticks_per_gen):
            multi_island_step(islands, cfg, store, global_tick, rng)
            migration_tracker.tick_migrants()
            global_tick += 1

            # Tick-level status update every 10 ticks
            if tick_in_gen % 10 == 0:
                elapsed = time.time() - start_time
                island_stats, alive_count, mean_fitness, best_fitness, mean_energy, mean_pred_err, mean_comp = \
                    _gather_island_stats(islands)
                _write_status({
                    "running": True,
                    "mode": "multi_island",
                    "generation": gen,
                    "total_generations": total_generations,
                    "tick": global_tick,
                    "tick_in_gen": tick_in_gen,
                    "ticks_per_gen": ticks_per_gen,
                    "alive_count": alive_count,
                    "total_agents": total_agents,
                    "mean_fitness": round(mean_fitness, 2),
                    "best_fitness": round(best_fitness, 2),
                    "mean_energy": round(mean_energy, 1),
                    "mean_pred_err": round(mean_pred_err, 4),
                    "islands": island_stats,
                    "top_agents": _build_top_agents(islands),
                    "history": history,
                    "elapsed_seconds": round(elapsed, 1),
                })
                print(f"  [tick {tick_in_gen}/{ticks_per_gen}] "
                      f"alive={alive_count}/{total_agents} "
                      f"best_fit={best_fitness:.2f}", flush=True)

        # Gather per-island stats at generation boundary
        island_stats, alive_count, mean_fitness, best_fitness, mean_energy, mean_pred_err, mean_comp = \
            _gather_island_stats(islands)

        for i, isl in enumerate(islands):
            isl_stat = island_stats[i]
            store.record_generation(
                gen, isl.id,
                isl_stat["alive"], isl_stat["mean_fitness"],
                isl_stat["mean_energy"], isl_stat["best_fitness"],
            )

        # Auto-save elites from all islands
        saved = auto_save_elites(
            [a for isl in islands for a in isl.agents if a.is_alive],
            top_fraction=top_fraction, directory=survivors_dir,
        )

        # Run probes at specified intervals
        probe_report = None
        if run_probes and (gen % probe_interval == 0 or gen == total_generations - 1):
            from genesis_v2.metrics.probes.runner import ProbeReport, run_all_probes, save_probe_report
            all_agents = []
            for isl in islands:
                all_agents.extend(isl.agents)
            probe_report = run_all_probes(
                all_agents, rng, generation=gen,
                preset_name=preset_name, quick=True,
            )
            save_probe_report(probe_report)
            print(probe_report.summary(), flush=True)

        gen_stats = {
            "generation": gen,
            "alive_count": alive_count,
            "mean_fitness": round(mean_fitness, 2),
            "best_fitness": round(best_fitness, 2),
            "mean_energy": round(mean_energy, 1),
            "mean_pred_err": round(mean_pred_err, 4),
            "mean_compression": round(mean_comp, 2),
            "islands": island_stats,
        }
        if probe_report:
            gen_stats["probes"] = probe_report.to_dict()
        history.append(gen_stats)

        elapsed = time.time() - start_time
        _write_status({
            "running": True,
            "mode": "multi_island",
            "generation": gen + 1,
            "total_generations": total_generations,
            "tick": global_tick,
            "alive_count": alive_count,
            "total_agents": total_agents,
            "mean_fitness": round(mean_fitness, 2),
            "best_fitness": round(best_fitness, 2),
            "mean_energy": round(mean_energy, 1),
            "mean_pred_err": round(mean_pred_err, 4),
            "islands": island_stats,
            "top_agents": _build_top_agents(islands),
            "history": history,
            "elapsed_seconds": round(elapsed, 1),
        })

        print(f"[gen {gen + 1}/{total_generations}] "
              f"alive={alive_count}/{total_agents} "
              f"best_fit={best_fitness:.2f} "
              f"mean_fit={mean_fitness:.2f}", flush=True)
        for isl_stat in island_stats:
            print(f"  [{isl_stat['island']}] alive={isl_stat['alive']} "
                  f"best={isl_stat['best_fitness']:.2f} "
                  f"mean={isl_stat['mean_fitness']:.2f}", flush=True)

        # Breed next generation (per island) using v2 breeder
        if gen < total_generations - 1:
            # Phase 2: migration every migration_interval generations
            if len(islands) >= 2 and (gen + 1) % migration_interval == 0:
                migration_tracker = migrate_agents(
                    islands, rng, cfg.physics, n_per_island=3,
                )
                print(f"  [migration] {len(migration_tracker.records)} agents migrated")

            for isl in islands:
                isl.agents = breed_generation_v2(
                    isl.agents, rng, cfg,
                    gen_memory=gen_memory,
                    top_fraction=top_fraction,
                    mutation_rate=isl.island_cfg.mutation_rate,
                )
                # Re-assign positions on comm_bus
                isl.comm_bus.assign_positions([a.id for a in isl.agents])

    # Save generational memory
    gen_memory.save()

    elapsed = time.time() - start_time
    final = json.loads(STATUS_FILE.read_text())
    final["running"] = False
    final["elapsed_seconds"] = round(elapsed, 1)
    _write_status(final)
    print(f"\n[done] {total_generations} gens, {global_tick} ticks, {elapsed:.1f}s", flush=True)

    # Auto-generate experiment report
    generate_experiment_report(
        history=history, elapsed=elapsed,
        total_generations=total_generations, ticks_per_gen=ticks_per_gen,
        total_agents=total_agents, seed=seed, mode="multi_island",
        social=social, preset_name=preset_name,
    )


def _run_single_island(
    cfg: GenesisConfig,
    rng: np.random.Generator,
    store: DuckDBStore,
    n_agents: int,
    total_generations: int,
    ticks_per_gen: int,
    top_fraction: float,
    mutation_rate: float,
    survivors_dir: Path,
    seed: int,
    social: bool,
    preset_name: str = "",
    run_probes: bool = False,
    probe_interval: int = 10,
    seed_paths: list[str] | None = None,
) -> None:
    """Legacy single-population experiment (backward compatible)."""
    env = MockMathEnvironment(
        n_cells=cfg.genome.node_dim,
        rng=np.random.default_rng(seed + 1),
    )

    # Social bus for single island
    comm_bus = None
    if social:
        from genesis_v2.social.comm_bus import CommunicationBus
        comm_bus = CommunicationBus(
            grid_rows=cfg.environment.grid_rows,
            grid_cols=cfg.environment.grid_cols,
            comm_radius=cfg.environment.comm_radius,
            rng=np.random.default_rng(seed + 2),
        )

    # Load specific seeds if provided, otherwise auto-load from bank
    if seed_paths:
        seeds = _load_specific_seeds(seed_paths)
        print(f"[experiment] Loaded {len(seeds)} selected seeds")
    else:
        seeds = auto_load_seeds(n_agents, directory=survivors_dir)

    if len(seeds) >= n_agents:
        agents = seeds[:n_agents]
        for i, a in enumerate(agents):
            a.id = f"g{a.generation}-seed-{i}"
            a.energy = cfg.physics.initial_energy
        loaded = len(seeds)
    else:
        agents = _create_fresh_population(cfg, rng, n_agents)
        for i, seed_agent in enumerate(seeds):
            seed_agent.id = f"g{seed_agent.generation}-seed-{i}"
            seed_agent.energy = cfg.physics.initial_energy
            agents[i] = seed_agent
        loaded = len(seeds)

    if comm_bus is not None:
        comm_bus.assign_positions([a.id for a in agents])

    start_time = time.time()
    global_tick = 0
    history: list[dict] = []

    _write_status({
        "running": True,
        "mode": "single_island",
        "generation": 0,
        "total_generations": total_generations,
        "tick": 0,
        "alive_count": len(agents),
        "total_agents": n_agents,
        "mean_fitness": 0.0,
        "best_fitness": 0.0,
        "mean_energy": float(cfg.physics.initial_energy),
        "survivors_loaded": loaded,
        "history": history,
    })

    gen_memory = GenerationalMemoryBank()

    for gen in range(total_generations):
        for tick_in_gen in range(ticks_per_gen):
            island_step_sync(agents, env, cfg.physics, store, global_tick, rng, comm_bus=comm_bus)
            global_tick += 1

            # Tick-level status update every 10 ticks
            if tick_in_gen % 10 == 0:
                alive_now = [a for a in agents if a.is_alive]
                alive_count_now = len(alive_now)
                mean_fit_now = float(np.mean([a.fitness for a in alive_now])) if alive_now else 0.0
                best_fit_now = float(max(a.fitness for a in alive_now)) if alive_now else 0.0
                mean_eng_now = float(np.mean([a.energy for a in alive_now])) if alive_now else 0.0
                elapsed = time.time() - start_time
                _write_status({
                    "running": True,
                    "mode": "single_island",
                    "generation": gen,
                    "total_generations": total_generations,
                    "tick": global_tick,
                    "tick_in_gen": tick_in_gen,
                    "ticks_per_gen": ticks_per_gen,
                    "alive_count": alive_count_now,
                    "total_agents": n_agents,
                    "mean_fitness": round(mean_fit_now, 2),
                    "best_fitness": round(best_fit_now, 2),
                    "mean_energy": round(mean_eng_now, 1),
                    "history": history,
                    "elapsed_seconds": round(elapsed, 1),
                })
                print(f"  [tick {tick_in_gen}/{ticks_per_gen}] "
                      f"alive={alive_count_now}/{n_agents} "
                      f"best_fit={best_fit_now:.2f}", flush=True)

        alive = [a for a in agents if a.is_alive]
        alive_count = len(alive)
        if alive:
            mean_fitness = float(np.mean([a.fitness for a in alive]))
            best_fitness = float(np.max([a.fitness for a in alive]))
            mean_energy = float(np.mean([a.energy for a in alive]))
            mean_pred_err = float(np.mean([a.prediction_error for a in alive]))
            mean_comp = float(np.mean([a.compression for a in alive]))
        else:
            mean_fitness = best_fitness = mean_energy = mean_pred_err = mean_comp = 0.0

        store.record_generation(gen, 0, alive_count, mean_fitness, mean_energy, best_fitness)
        saved = auto_save_elites(agents, top_fraction=top_fraction, directory=survivors_dir)

        # Run probes at specified intervals
        probe_report = None
        if run_probes and (gen % probe_interval == 0 or gen == total_generations - 1):
            from genesis_v2.metrics.probes.runner import ProbeReport, run_all_probes, save_probe_report
            probe_report = run_all_probes(
                agents, rng, generation=gen,
                preset_name=preset_name, quick=True,
            )
            save_probe_report(probe_report)
            print(probe_report.summary(), flush=True)

        gen_stats = {
            "generation": gen,
            "alive_count": alive_count,
            "mean_fitness": round(mean_fitness, 2),
            "best_fitness": round(best_fitness, 2),
            "mean_energy": round(mean_energy, 1),
            "mean_pred_err": round(mean_pred_err, 4),
            "mean_compression": round(mean_comp, 2),
        }
        if probe_report:
            gen_stats["probes"] = probe_report.to_dict()
        history.append(gen_stats)

        elapsed = time.time() - start_time
        _write_status({
            "running": True,
            "mode": "single_island",
            "generation": gen + 1,
            "total_generations": total_generations,
            "tick": global_tick,
            "alive_count": alive_count,
            "total_agents": n_agents,
            "mean_fitness": round(mean_fitness, 2),
            "best_fitness": round(best_fitness, 2),
            "mean_energy": round(mean_energy, 1),
            "mean_pred_err": round(mean_pred_err, 4),
            "mean_compression": round(mean_comp, 2),
            "top_agents": _rank_top(agents),
            "history": history,
            "elapsed_seconds": round(elapsed, 1),
        })

        print(f"[gen {gen + 1}/{total_generations}] "
              f"alive={alive_count}/{n_agents} "
              f"best_fit={best_fitness:.2f} "
              f"mean_fit={mean_fitness:.2f} "
              f"saved={saved} elites", flush=True)

        if gen < total_generations - 1:
            agents = breed_generation_v2(
                agents, rng, cfg,
                gen_memory=gen_memory,
                top_fraction=top_fraction,
                mutation_rate=mutation_rate,
            )
            if comm_bus is not None:
                comm_bus.assign_positions([a.id for a in agents])

    gen_memory.save()

    elapsed = time.time() - start_time
    final = json.loads(STATUS_FILE.read_text())
    final["running"] = False
    final["elapsed_seconds"] = round(elapsed, 1)
    _write_status(final)
    print(f"\n[done] {total_generations} gens, {global_tick} ticks, {elapsed:.1f}s", flush=True)

    # Auto-generate experiment report
    generate_experiment_report(
        history=history, elapsed=elapsed,
        total_generations=total_generations, ticks_per_gen=ticks_per_gen,
        total_agents=n_agents, seed=seed, mode="single_island",
        social=social, preset_name=preset_name,
    )
