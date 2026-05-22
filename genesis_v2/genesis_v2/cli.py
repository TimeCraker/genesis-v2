"""CLI entry point for Genesis v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_smoke(args: argparse.Namespace) -> int:
    """Single agent, 10 forward passes — basic sanity check."""
    import numpy as np
    from genesis_v2.genome.graph import GraphConfig, new_genome_graph
    from genesis_v2.agent.agent import new_agent, split_output

    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    agent = new_agent(id="smoke-0", genome=g, initial_energy=1000.0)

    d_in = cfg.input_dim
    for t in range(10):
        inp = rng.standard_normal(d_in).astype(np.float32)
        out = agent.genome.forward(inp)
        action, message, state, selfmod = split_output(out)
        assert np.all(np.isfinite(out)), f"Non-finite output at tick {t}"

    print(f"[smoke] OK — 10 forward passes, output shape={out.shape}, "
          f"action={action.shape}, message={message.shape}, "
          f"mean={out.mean():.4f}, std={out.std():.4f}")
    return 0


def cmd_mock_loop(args: argparse.Namespace) -> int:
    """Mock closed-loop: N agents × T ticks."""
    import numpy as np
    from genesis_v2.genome.graph import GraphConfig, new_genome_graph
    from genesis_v2.agent.agent import new_agent, split_output
    from genesis_v2.engine.tick import island_step_sync
    from genesis_v2.env.mock import MockMathEnvironment
    from genesis_v2.config import load_config

    n_agents = args.agents
    n_ticks = args.ticks
    dim = args.dim
    social = getattr(args, 'social', True)

    cfg = load_config()
    phy = cfg.physics
    rng = np.random.default_rng(42)
    graph_cfg = GraphConfig(node_dim=dim)

    env = MockMathEnvironment(n_cells=dim, rng=np.random.default_rng(0))

    agents = []
    for i in range(n_agents):
        g = new_genome_graph(graph_cfg, rng)
        a = new_agent(id=f"agent-{i}", genome=g, initial_energy=phy.initial_energy)
        agents.append(a)

    # Communication bus (optional)
    comm_bus = None
    if social:
        from genesis_v2.social.comm_bus import CommunicationBus
        comm_bus = CommunicationBus(
            grid_rows=cfg.environment.grid_rows,
            grid_cols=cfg.environment.grid_cols,
            comm_radius=cfg.environment.comm_radius,
            rng=np.random.default_rng(99),
        )
        comm_bus.assign_positions([a.id for a in agents])
        print(f"[mock-loop] Social mode ON (radius={cfg.environment.comm_radius})")

    for tick in range(n_ticks):
        island_step_sync(agents, env, phy, None, tick, rng, comm_bus=comm_bus)

        if (tick + 1) % max(1, n_ticks // 10) == 0:
            alive = [a for a in agents if a.is_alive]
            energies = [a.energy for a in alive]
            mean_e = np.mean(energies) if energies else 0
            msg_lens = [np.linalg.norm(a.last_message) for a in alive if a.last_message is not None]
            mean_msg = np.mean(msg_lens) if msg_lens else 0
            print(f"[mock-loop] tick={tick + 1}/{n_ticks} "
                  f"alive={len(alive)}/{n_agents} "
                  f"mean_energy={mean_e:.1f} "
                  f"mean_msg_norm={mean_msg:.3f}")

    alive = [a for a in agents if a.is_alive]
    print(f"[mock-loop] DONE — {len(alive)}/{n_agents} survived {n_ticks} ticks")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Run a full experiment with generation loop + auto-survivor management."""
    from genesis_v2.scripts.run_experiment import run_experiment

    social = getattr(args, 'social', True)
    multi_island = getattr(args, 'multi_island', True)

    seed_file = getattr(args, 'seeds_file', None)
    seed_paths = None
    if seed_file:
        import json as _json
        from pathlib import Path
        p = Path(seed_file)
        if p.exists():
            seed_paths = _json.loads(p.read_text(encoding="utf-8"))
            print(f"[experiment] Loading {len(seed_paths)} selected seeds from {seed_file}")

    run_experiment(
        n_agents=args.agents,
        total_generations=args.generations,
        ticks_per_gen=args.ticks,
        top_fraction=args.top_fraction,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        social=social,
        multi_island=multi_island,
        seed_paths=seed_paths,
    )
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "scripts" / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
           "--server.headless", "true"]
    print(f"[dashboard] Launching: {' '.join(cmd)}")
    return subprocess.call(cmd)


def cmd_converse(args: argparse.Namespace) -> int:
    """Interactive conversation with the best evolved agent."""
    import numpy as np
    from genesis_v2.agent.agent import new_agent, split_output
    from genesis_v2.config import load_config
    from genesis_v2.evolution.survivor_bank import load_top_survivors
    from genesis_v2.genome.graph import GraphConfig, new_genome_graph
    from genesis_v2.translation.translator import ConversationSession, Translator

    cfg = load_config()

    # Load best survivor or create fresh agent
    seeds = load_top_survivors(n=1)
    if seeds:
        agent = seeds[0]
        agent.energy = cfg.physics.initial_energy
        agent.fitness = 0.0
        agent.is_alive = True
        agent.genome.reset_state()
        print(f"[converse] Loaded elite agent: {agent.id} "
              f"(gen={agent.generation}, birth_nodes={agent.birth_nodes})")
    else:
        rng = np.random.default_rng(42)
        graph_cfg = GraphConfig(
            node_dim=cfg.genome.node_dim,
            input_nodes=cfg.genome.input_nodes,
            output_nodes_action=cfg.genome.output_nodes_action,
            output_nodes_message=cfg.genome.output_nodes_message,
            output_nodes_state=cfg.genome.output_nodes_state,
            output_nodes_selfmod=cfg.genome.output_nodes_selfmod,
            initial_hidden_nodes=cfg.genome.initial_hidden_nodes,
            initial_edge_density=cfg.genome.initial_edge_density,
        )
        g = new_genome_graph(graph_cfg, rng)
        agent = new_agent(id="fresh-0", genome=g, initial_energy=cfg.physics.initial_energy)
        print(f"[converse] No survivors found, using fresh agent")

    # Create translator
    translator = Translator(
        backend_name="mock",
        n_input_nodes=cfg.genome.input_nodes,
        d_node=cfg.genome.node_dim,
    )

    session = ConversationSession(agent=agent, translator=translator, max_turns=args.max_turns)

    print(f"[converse] Agent {agent.id} ready. Type your message (or 'quit' to exit).")
    print(f"[converse] Agent has {agent.genome.node_count()} nodes, "
          f"{agent.genome.edge_count()} edges")
    print()

    turn = 0
    while turn < args.max_turns:
        try:
            human_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not human_text or human_text.lower() in ("quit", "exit", "q"):
            break

        result = session.send(human_text)
        print(f"Agent: {result.agent_response}")
        print(f"  [energy={result.agent_energy:.1f}, fitness={result.agent_fitness:.2f}]")
        turn += 1

    print(f"\n[converse] Session ended after {turn} turns.")
    if session.history:
        print(session.get_conversation_text())

    # Run probes if requested
    if args.probe:
        from genesis_v2.metrics.probes.conversation import run_conversation_probes
        probe_result = run_conversation_probes(agent, translator)
        print(f"\n[probes] Response diversity: {probe_result.response_diversity:.4f}")
        print(f"[probes] Semantic similarity: {probe_result.semantic_similarity:.4f}")
        print(f"[probes] Multi-turn coherence: {probe_result.multi_turn_coherence:.4f}")
        print(f"[probes] Cross-LLM consistency: {probe_result.cross_llm_consistency:.4f}")
        print(f"[probes] Unique responses: {probe_result.n_unique_responses}/{probe_result.n_turns}")

    return 0


def cmd_mve_run(args: argparse.Namespace) -> int:
    """Run ablation experiment with probes."""
    import yaml
    from genesis_v2.config import GenesisConfig, load_config
    from genesis_v2.scripts.run_experiment import run_experiment

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[mve-run] Config not found: {config_path}")
        return 1

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    presets = raw.get("presets", {})

    if args.compare:
        # Run all presets
        print(f"[mve-run] Running ALL {len(presets)} presets for comparison")
        for preset_name in sorted(presets.keys()):
            print(f"\n{'='*60}")
            print(f"[mve-run] Preset {preset_name}: {presets[preset_name].get('name', '')}")
            print(f"{'='*60}")
            _run_preset(presets[preset_name], args, preset_name)
        print(f"\n[mve-run] All presets complete. Probe reports saved to data/probes/")
        return 0
    else:
        preset_name = args.preset.upper()
        if preset_name not in presets:
            print(f"[mve-run] Preset '{preset_name}' not found. Available: {list(presets.keys())}")
            return 1
        print(f"[mve-run] Running preset {preset_name}: {presets[preset_name].get('name', '')}")
        return _run_preset(presets[preset_name], args, preset_name)


def _run_preset(preset: dict, args: argparse.Namespace, preset_name: str) -> int:
    """Run a single ablation preset."""
    from genesis_v2.config import load_config
    from genesis_v2.scripts.run_experiment import run_experiment

    # Load base config
    cfg = load_config()

    # Apply overrides
    overrides = {k: v for k, v in preset.items() if k not in ("name", "description")}
    _apply_overrides(cfg, overrides)

    run_experiment(
        config=cfg,
        n_agents=args.agents,
        total_generations=args.generations,
        ticks_per_gen=args.ticks,
        seed=args.seed,
        preset_name=preset_name,
        run_probes=True,
        probe_interval=args.probe_interval,
        social=cfg.environment.comm_radius > 0,
        multi_island=bool(cfg.population.islands),
    )
    return 0


def _apply_overrides(cfg: GenesisConfig, overrides: dict) -> None:
    """Apply preset overrides to config."""
    for section, values in overrides.items():
        if not isinstance(values, dict):
            continue
        section_obj = getattr(cfg, section, None)
        if section_obj is None:
            continue
        for key, val in values.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, val)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genesis_v2",
        description="Project Genesis v2 — Multi-LLM AGI Evolution Platform",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 2.0.0")

    sub = parser.add_subparsers(dest="command")

    # smoke
    p_smoke = sub.add_parser("smoke", help="Single agent, 10 forward passes")
    p_smoke.set_defaults(func=cmd_smoke)

    # mock-loop
    p_mock = sub.add_parser("mock-loop", help="N agents × T ticks mock closed-loop")
    p_mock.add_argument("--agents", type=int, default=10, help="Number of agents")
    p_mock.add_argument("--ticks", type=int, default=100, help="Number of ticks")
    p_mock.add_argument("--dim", type=int, default=64, help="Node dimension")
    p_mock.add_argument("--social", dest="social", action="store_true", default=True, help="Enable social layer")
    p_mock.add_argument("--no-social", dest="social", action="store_false", help="Disable social layer")
    p_mock.set_defaults(func=cmd_mock_loop)

    # experiment
    p_exp = sub.add_parser("experiment", help="Run full experiment with generation loop")
    p_exp.add_argument("--agents", type=int, default=10, help="Population size (single-island mode)")
    p_exp.add_argument("--generations", type=int, default=10, help="Total generations")
    p_exp.add_argument("--ticks", type=int, default=100, help="Ticks per generation")
    p_exp.add_argument("--top-fraction", type=float, default=0.25, help="Elite fraction")
    p_exp.add_argument("--mutation-rate", type=float, default=0.15, help="Mutation rate")
    p_exp.add_argument("--seed", type=int, default=42, help="Random seed")
    p_exp.add_argument("--social", dest="social", action="store_true", default=True, help="Enable social layer")
    p_exp.add_argument("--no-social", dest="social", action="store_false", help="Disable social layer")
    p_exp.add_argument("--multi-island", dest="multi_island", action="store_true", default=True, help="Use multi-island mode")
    p_exp.add_argument("--single-island", dest="multi_island", action="store_false", help="Use single-island mode")
    p_exp.add_argument("--seeds-file", dest="seeds_file", type=str, default=None,
                       help="JSON file with list of survivor file paths to load as seeds")
    p_exp.set_defaults(func=cmd_experiment)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    # converse
    p_conv = sub.add_parser("converse", help="Interactive conversation with best evolved agent")
    p_conv.add_argument("--max-turns", type=int, default=20, help="Max conversation turns")
    p_conv.add_argument("--probe", action="store_true", default=False, help="Run conversation quality probes after")
    p_conv.set_defaults(func=cmd_converse)

    # mve-run (ablation experiments)
    p_mve = sub.add_parser("mve-run", help="Run ablation experiment with probes")
    p_mve.add_argument("--config", type=str, default="experiments/ablation.yaml", help="Ablation config file")
    p_mve.add_argument("--preset", type=str, default="A", help="Preset name (A-H)")
    p_mve.add_argument("--generations", type=int, default=5, help="Total generations")
    p_mve.add_argument("--ticks", type=int, default=50, help="Ticks per generation")
    p_mve.add_argument("--agents", type=int, default=10, help="Agents per island (single-island mode)")
    p_mve.add_argument("--probe-interval", type=int, default=1, help="Run probes every N generations")
    p_mve.add_argument("--seed", type=int, default=42, help="Random seed")
    p_mve.add_argument("--compare", action="store_true", default=False, help="Run all presets and compare")
    p_mve.set_defaults(func=cmd_mve_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
