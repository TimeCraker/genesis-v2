# Genesis v2

**Multi-LLM Semantic Wilderness — Autonomous AGI Evolution Platform**

Digital organisms evolve in a multi-universe powered by different LLMs. Through energy constraints, social pressure, self-modification, and cross-generational memory, agents develop their own cognitive structures — no training, no hardcoding, no human priors.

---

## Quick Start

```bash
cd genesis_v2
uv sync --extra dev                  # install dependencies
uv run pytest -q                     # verify (250 tests, ~4s)
uv run python -m genesis_v2 smoke    # single agent sanity check
```

## Run an Experiment

```bash
# Dashboard (recommended) — configure, launch, monitor in one place
uv run python -m genesis_v2 dashboard

# CLI — mock environment (free, fast)
uv run python -m genesis_v2 mock-loop --agents 10 --ticks 100

# CLI — full multi-island experiment
uv run python -m genesis_v2 experiment --agents 10 --generations 10 --ticks 100

# Talk to the best evolved agent
uv run python -m genesis_v2 converse --probe

# Ablation experiments (8 presets A-H)
uv run python -m genesis_v2 mve-run --preset A
```

---

## Dashboard

Streamlit control panel — the main way to configure and run experiments.

```bash
uv run python -m genesis_v2 dashboard    # opens http://localhost:8501
```

| Tab | What it does |
|---|---|
| **Config** | Per-island LLM backend selection, API keys with one-click connectivity test, hardware presets, physics/genome/evolution parameters |
| **Run** | Launch presets (Quick / Standard / Deep / Marathon), seed bank selection (auto or manual), start/stop experiments |
| **Monitor** | Real-time KPI cards, tick-level progress bar, fitness/energy charts, top agents leaderboard — all auto-updating every 5s without page refresh |

**Key features:**

- **Per-island backend** — each of the 4 islands picks its own LLM (DeepSeek / Gemini / MiMo / OpenAI / Mock). Different islands, different models.
- **Real-time monitoring** — `st.fragment`-based auto-refresh. No page reload, no flicker. Data updates in-place every 5 seconds.
- **Reliable stop** — PID persisted to status file. Stop button works even after browser refresh.
- **Stale detection** — dashboard auto-detects crashed experiments and marks them finished.
- **Budget safety** — $50 hard cap per experiment, auto-downgrade to Mock when budget runs out.
- **Bilingual UI** — all labels in Chinese + English.

---

## LLM Configuration

API keys are set via environment variables or the Dashboard UI:

```bash
export GENESIS_DEEPSEEK_KEY=sk-xxx
export GENESIS_GEMINI_KEY=AIzaSy-xxx
export GENESIS_MIMO_KEY=xxx
export GENESIS_OPENAI_KEY=sk-xxx
```

Backend definitions in `configs/backends.yaml` — add a new model by adding a new entry, zero code changes.

---

## Architecture

```
Environment (Multi-LLM: DeepSeek / Gemini / MiMo / OpenAI / Mock-CA)
    |  observe / interact / true_distribution
    v
Agent Population (4 Islands, each with independent LLM backend)
    ├── GenomeGraph (vector nodes, D=64, matrix weights)
    │   ├── 12 mutation primitives (NEAT split, attention groups, modules, ...)
    │   ├── Forward pass (BLAS matrix multiply, Kahn topo-sort)
    │   └── Crossover (NEAT-style sexual reproduction)
    ├── Output Partition (528-dim)
    │   ├── [0:256]   Action   → environment via projection matrix P
    │   ├── [256:384]  Message  → neighbors via CommunicationBus
    │   ├── [384:512]  State    → working memory (feedback loop)
    │   └── [512:528]  SelfMod  → mutation instructions
    ├── Energy System (cost: tokens/nodes/edges/API/messages)
    └── Selection (starvation / entropy overflow / self-mod failure)
```

---

## Key Concepts

| Concept | What it does |
|---|---|
| **Multi-LLM Islands** | 4 islands, each bound to a different LLM. Agents migrate between islands, forcing generalization over memorization. |
| **Energy = Life** | Every tick costs energy (tokens, nodes, edges). Agents that predict well earn energy. Zero energy = death. |
| **Social Layer** | Agents sit on a 2D grid, send 128-dim vectors to neighbors. Communication cost incentivizes efficient signaling. |
| **Self-Modification** | Last 16 dims of output = mutation instructions. 70% death rate. Requires high energy. Drives emergence of self-model. |
| **Exploration Bonus** | Reward for behavior the LLM didn't predict but the agent can. Transition: student -> explorer. |
| **Cross-Generation Memory** | Elite agents' behavioral signatures are passed to offspring (50% probability). |
| **Probes** | 7 cognitive probes distinguish "understanding" from "overfitting": OOD generalization, modularity, multi-scale, multi-LLM adaptation, communication, self-mod efficiency, exploration effect. |

---

## Project Structure

```
genesis_v2/
├── genesis_v2/
│   ├── cli.py                 # CLI entry point
│   ├── config.py              # Pydantic config models
│   ├── agent/agent.py         # Agent entity (output-partitioned)
│   ├── genome/
│   │   ├── graph.py           # GenomeGraph (vector nodes, D=64)
│   │   ├── mutate.py          # 12 mutation primitives
│   │   ├── forward.py         # BLAS forward pass
│   │   └── crossover.py       # NEAT crossover
│   ├── engine/
│   │   ├── tick.py            # Tick loop (single + multi-island)
│   │   ├── metabolism.py      # Energy settlement
│   │   ├── reaper.py          # Death evaluation
│   │   └── selfmod.py         # Self-modification channel
│   ├── env/
│   │   ├── mock.py            # Mock CA environment (free)
│   │   ├── multi_llm.py       # Multi-LLM environment
│   │   ├── budget.py          # Budget manager + fallback
│   │   └── embed.py           # Frozen embedding projection
│   ├── social/
│   │   ├── comm_bus.py        # CommunicationBus (grid topology)
│   │   └── reputation.py      # Reputation system
│   ├── population/
│   │   ├── island.py          # Island class
│   │   └── migration.py       # Cross-LLM migration
│   ├── evolution/
│   │   ├── breeder.py         # Breeding strategy (50% crossover / 30% clone / 20% explore)
│   │   ├── gen_memory.py      # Generational memory bank
│   │   └── survivor_bank.py   # Elite persistence (JSON)
│   ├── metrics/probes/        # 7 cognitive probes + runner
│   ├── translation/translator.py  # Vector <-> natural language
│   ├── storage/duckdb_store.py    # DuckDB telemetry
│   └── scripts/
│       ├── dashboard.py       # Streamlit dashboard
│       └── run_experiment.py  # Experiment runner
├── configs/
│   ├── genesis_v2.yaml        # Physics constants + experiment params
│   └── backends.yaml          # LLM backend definitions
├── data/
│   ├── survivors/             # Elite agent JSON files
│   └── experiments/           # DuckDB experiment databases
└── tests/                     # 250 tests, 26 files
```

---

## Configuration

### Physics Constants (`configs/genesis_v2.yaml`)

```yaml
physics:
  alpha: 0.01        # token cost weight
  beta: 0.005        # node cost weight
  w_pred: 1.0        # prediction reward weight
  w_explore: 0.2     # exploration bonus weight
  initial_energy: 5000.0

evolution:
  generation_ticks: 200
  migration_interval_generations: 50

population:
  islands:
    - name: Explorer
      size: 100
      mutation_rate: 0.30
      backend: deepseek
    - name: Exploiter
      size: 100
      mutation_rate: 0.05
      backend: gemini
```

### LLM Backends (`configs/backends.yaml`)

Each backend entry has: `base_url`, `api_key_env`, `model`, `max_tokens`, `timeout_sec`. Add a new model by adding a new entry — zero code changes needed.

---

## Tests

```bash
uv run pytest -q                                 # 250 passed in ~4s
uv run pytest tests/test_genome_graph.py -v      # single module
uv run pytest tests/test_probes.py -v            # probe tests
```

---

## Current Status

| Phase | Status | Key Result |
|---|---|---|
| Phase 0: Skeleton | Done | 81 tests, 10-agent x 100-tick smoke run |
| Phase 1: Social + Multi-LLM | Done | 130 tests, messages norm ~35, 4 islands 400/400 alive |
| Phase 2: Evolution + Exploration | Done | 193 tests, fitness rising across generations (2254 -> 6308) |
| Phase 3: Translation + Dialogue | Done | 227 tests, human-agent conversation working |
| Phase 4: Deep Experiments + Probes | Done | 250 tests, 7 probes, B-tier (3/7 probes passing) |

**Tier: B** — Evolution signals confirmed. A-tier requires 1 more probe passing.

Full design spec: `docx/AGI_v2_roadmap.md`

---

## Tech Stack

Python 3.12 / NumPy / SciPy / Pydantic / NetworkX / DuckDB / httpx / Streamlit / pytest + hypothesis

## License

Research project. See repository for terms.
