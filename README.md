# Project Genesis v2.0

**Multi-LLM Semantic Wilderness -- Autonomous AGI Evolution Platform**

> In a universe where multiple LLMs serve as independent "laws of physics," digital organisms evolve through energy constraints, social pressure, self-modification, and cross-generational inheritance -- developing their own cognitive structures from scratch. No training, no hardcoding, no human priors.

---

## What Is This?

Project Genesis v2 is an **evolutionary experiment platform for AGI research**. It asks one question:

> If we give a single-celled digital organism social ability, self-modification capability, and multiple different universes to live in -- will civilization emerge on its own?

This is **not** a chatbot, not an agent framework, not a wrapper around LLMs. It is a **digital terrarium** where:

- Agents are **pure mathematical structures** (GenomeGraph with vector nodes, D=64)
- The environment is powered by **real LLM APIs** (DeepSeek, Gemini, MiMo, etc.)
- Evolution is driven by **energy economics** -- predict well or die
- Intelligence is measured by **7 cognitive probes**, not benchmark scores

### Core Philosophy: From "Learning from Humans" to "Surpassing Humans"

```
Phase 1  Learning:    Prediction reward dominates  -> Agent learns human knowledge from LLM
    (prediction reward marginally decreases)
Phase 2  Internalize: Compression reward dominates  -> Agent compresses knowledge into structured representations
    (compression approaches limit)
Phase 3  Transcend:   Exploration reward dominates  -> Agent explores beyond LLM knowledge
    (Agent develops self-model)
Phase 4  Autonomous:  Self-modification emerges     -> Agent optimizes its own topology
```

Each phase transition is **driven by energy gradients naturally**, not hardcoded.

---

## Key Concepts

### Multi-LLM Islands

4 islands, each bound to a different LLM. Agents migrate between islands periodically.

**Why?** If an agent only lives with one LLM, it can survive by memorizing behavioral patterns (like a student memorizing exam answers). But when forced to live across multiple LLMs, it **must** learn:

- The **deep structure** shared across different LLMs (language patterns, logical consistency)
- **Transferable reasoning** (general causal inference, not pattern matching)
- **Environment-adaptive internal models** (distinguishing "physics" from "surface features")

Like a child raised across multiple cultures -- forced to learn "universal social rules" rather than memorizing one culture's behavioral patterns.

### Energy = Life

Every tick costs energy (token usage, node count, edges, API costs, messages sent). Agents that predict well earn energy. **Zero energy = death.** This creates natural selection pressure:

- Efficient agents survive (use fewer tokens for better predictions)
- Bloated topologies die (too many nodes/edges drain energy)
- Poor communicators die (spamming messages wastes energy)

### Output Partition (528-dim)

Each agent's output vector is physically partitioned into functional zones:

```
[0:256]    Action    -> projected to LLM token space via frozen matrix P
[256:384]  Message   -> sent to grid neighbors via CommunicationBus
[384:512]  State     -> feedback loop as working memory for next tick
[512:528]  SelfMod   -> mutation instructions for self-modification
```

These are **physical interfaces of the universe** (like USB pin definitions), not business logic. The agent's internal GenomeGraph remains zero-prior, zero-pretrain.

### Self-Modification

The most ambitious design: the last 16 dimensions of output are interpreted as "surgery instructions" for the agent to modify its own topology.

- 70% chance of degradation or death (simulating brain surgery risk)
- Requires high energy surplus (only "wealthy" agents can afford to try)
- Drives emergence of **self-model** and **causal reasoning** -- core AGI capabilities

### Social Layer

Agents sit on a 2D grid. Each agent has a "mouth" (128-dim message channel), "ears" (inbox), and neighbors. How to use them is up to the agent. Combined with social pressure (resource sharing, cooperation rewards), agents are **forced** to attempt meaningful signaling. Language emerges from the pressure of "communicate or starve."

### Exploration Bonus

```
ExplorationBonus = max(0, surprise_to_LLM - surprise_to_self)
```

When the agent does something the LLM considers impossible, but the agent itself can perfectly predict -- it earns bonus energy. This drives the transition from "student" (learning human knowledge) to "explorer" (discovering beyond human knowledge).

### 7 Cognitive Probes

Distinguishing true understanding from overfitting:

| Probe | Measures | Understanding | Overfitting |
|---|---|---|---|
| OOD Generalization | KL change on domain shift | < 2x amplification | > 10x |
| Topology Modularity Q | Louvain community detection | Q > 0.4 | Q < 0.2 |
| Multi-Scale Consistency | Multi-window KL ratio | < 3.0 | > 20.0 |
| Multi-LLM Adaptation | Cross-model KL + migration speed | < 50 ticks | > 500 ticks |
| Communication Emergence | I(msg; action) mutual information | > 0.5 bits | ~ 0 |
| Self-Mod Efficiency | Survival rate + fitness delta | > 0.3 survival | < 0.05 |
| Exploration Effect | Exploration ratio + fitness change | fitness up | fitness down |

---

## Architecture

```
+-----------------------------------------------------------+
|           Multi-LLM Environment                            |
|      (DeepSeek / Gemini / MiMo / Mock-CA)                 |
+--------------------^--------------------------------------+
                     |
          action / feedback (token semantic flow)
                     |
         [ Interaction Bus + Communication Bus ]
                     |
+--------------------v--------------------------------------+
|           Agent Population (4 Islands)                     |
|   +-- GenomeGraph (vector nodes, D=64, matrix weights)    |
|   |   +-- 12 mutation primitives                          |
|   |   +-- Forward pass (BLAS matrix multiply)             |
|   |   +-- Crossover (NEAT-style sexual reproduction)      |
|   +-- Output Partition (528-dim)                           |
|   |   +-- Action [0:256]  -> environment                  |
|   |   +-- Message [256:384] -> neighbors                  |
|   |   +-- State [384:512]  -> working memory              |
|   |   +-- SelfMod [512:528] -> mutation instructions      |
|   +-- Energy System                                        |
|   +-- Selection (starvation / entropy / self-mod failure)  |
+-----------------------------------------------------------+
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/TimeCraker/genesis-v2.git
cd genesis-v2/genesis_v2

# Install
uv sync --extra dev

# Test (250 tests, ~4s)
uv run pytest -q

# Smoke test
uv run python -m genesis_v2 smoke

# Mock environment run (free, no API keys needed)
uv run python -m genesis_v2 mock-loop --agents 10 --ticks 100

# Dashboard (Streamlit UI)
uv run python -m genesis_v2 dashboard
```

### LLM Configuration

API keys via environment variables:

```bash
export GENESIS_DEEPSEEK_KEY=sk-xxx
export GENESIS_GEMINI_KEY=AIzaSy-xxx
export GENESIS_MIMO_KEY=xxx
export GENESIS_OPENAI_KEY=sk-xxx
```

Backend definitions in `configs/backends.yaml`. Add a new model by adding a new entry -- zero code changes.

---

## Project Structure

```
AGI7/
├── docx/                              # Design documents
│   ├── AGI_v2_roadmap.md              # Full engineering blueprint (14 chapters)
│   ├── PLAN_LIST.md                   # Progress tracker
│   └── 设计理念思路.md                 # Design philosophy (Chinese)
├── genesis_v2/                        # Main project
│   ├── genesis_v2/                    # Source code
│   │   ├── agent/agent.py             # Agent entity (output-partitioned)
│   │   ├── genome/
│   │   │   ├── graph.py               # GenomeGraph (vector nodes, D=64)
│   │   │   ├── mutate.py              # 12 mutation primitives
│   │   │   ├── forward.py             # BLAS forward pass
│   │   │   └── crossover.py           # NEAT crossover
│   │   ├── engine/
│   │   │   ├── tick.py                # Tick loop (single + multi-island)
│   │   │   ├── metabolism.py          # Energy settlement
│   │   │   ├── reaper.py              # Death evaluation
│   │   │   └── selfmod.py             # Self-modification channel
│   │   ├── env/
│   │   │   ├── mock.py                # Mock CA environment (free)
│   │   │   ├── multi_llm.py           # Multi-LLM environment
│   │   │   ├── budget.py              # Budget manager + fallback
│   │   │   └── embed.py               # Frozen embedding projection
│   │   ├── social/
│   │   │   ├── comm_bus.py            # CommunicationBus (grid topology)
│   │   │   └── reputation.py          # Reputation system
│   │   ├── population/
│   │   │   ├── island.py              # Island class
│   │   │   └── migration.py           # Cross-LLM migration
│   │   ├── evolution/
│   │   │   ├── breeder.py             # Breeding strategy
│   │   │   ├── gen_memory.py          # Generational memory bank
│   │   │   └── survivor_bank.py       # Elite persistence
│   │   ├── metrics/probes/            # 7 cognitive probes
│   │   ├── translation/translator.py  # Vector <-> natural language
│   │   ├── storage/duckdb_store.py    # DuckDB telemetry
│   │   └── scripts/dashboard.py       # Streamlit dashboard
│   ├── configs/
│   │   ├── genesis_v2.yaml            # Physics constants + experiment params
│   │   └── backends.yaml              # LLM backend definitions
│   ├── tests/                         # 250 tests across 26 files
│   └── pyproject.toml
└── README.md
```

---

## v1 vs v2

| Dimension | v1 | v2 |
|---|---|---|
| Node computation | Scalar (1 float/node) | Vector (64-dim/node) |
| Edge weight | Scalar float | Matrix W in R^{DxD} |
| Agent interaction | Fully independent | Local communication (grid neighbors) |
| Environment | Single LLM or Mock | Multi-LLM islands (each island bound to different LLM) |
| Evolution | Asexual (clone + mutate) | + Crossover (NEAT-style sexual reproduction) |
| Selection pressure | Prediction + compression + behavioral variance | + Social reward + exploration reward + migration adaptation |
| Self-modification | None | 16-dim output channel -> mutation instructions |
| Cross-gen memory | None (recurrent state reset) | GenerationalMemory (experience summary inheritance) |
| Final goal | Metric curves | Natural language conversation with agents |

---

## Paradigm Red Lines

Things that are **absolutely forbidden** inside the system:

- SFT / RLHF / Supervised learning
- Prompt Engineering / Chain-of-Thought injection
- Agent Workflow / Tool calling / LangChain-style assembly
- Hand-written "cognitive modules" (Memory / Planner / Attention / S1-S2)
- Any `if token == "hello"` style business logic hardcoding

Things that are **the only allowed**:

- Define **physical constants** (energy conversion rates, mutation probabilities, projection matrices)
- Define **physical interfaces** (output partition: action/message/state/self-modification)
- Define **pure math rewards** (KL divergence, MDL, mutual information, exploration bonus formulas)
- Define **physical rule stabilization** (LLM temperature = 0)

---

## Current Status

| Phase | Status | Key Result |
|---|---|---|
| Phase 0: Skeleton | Done | 81 tests, 10-agent x 100-tick smoke run |
| Phase 1: Social + Multi-LLM | Done | 130 tests, messages norm ~35, 4 islands 400/400 alive |
| Phase 2: Evolution + Exploration | Done | 193 tests, fitness rising across generations (2254 -> 6308) |
| Phase 3: Translation + Dialogue | Done | 227 tests, human-agent conversation working |
| Phase 4: Deep Experiments + Probes | Done | 250 tests, 7 probes, B-tier (3/7 probes passing) |

**Tier: B** -- Evolution signals confirmed. A-tier requires 1 more probe passing.

---

## Tech Stack

Python 3.11+ / NumPy / SciPy / Pydantic / NetworkX / DuckDB / httpx / Streamlit / pytest + hypothesis

---

## Worst Case / Best Case

**Worst case:** We get a batch of digital organisms that can quickly adapt across multiple LLMs, communicate with each other, and self-modify -- even if they haven't "understood" anything, these behaviors are already evolutionary phenomena worth studying.

**Best case:** We see language emerge from social pressure, self-awareness sprout from self-modification, and agents begin doing things humans didn't predict.

Either way, it's honest science.

---

## License

[MIT](LICENSE)
