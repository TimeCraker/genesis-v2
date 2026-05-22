<div align="center">

# Project Genesis v2.0

**Multi-LLM Semantic Wilderness -- Autonomous AGI Evolution Platform**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00C853.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-250%20passed-4CAF50)](#testing)
[![Phase](https://img.shields.io/badge/Phase-4%20B--Tier-FF9800)](#current-status)

<br/>

> **English** | [中文](#中文版)

<br/>

*In a universe where multiple LLMs serve as independent "laws of physics,"*
*digital organisms evolve through energy constraints, social pressure,*
*self-modification, and cross-generational inheritance --*
*developing their own cognitive structures from scratch.*
***No training. No hardcoding. No human priors.***

</div>

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Core Philosophy](#core-philosophy)
- [Key Concepts](#key-concepts)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [v1 vs v2](#v1-vs-v2)
- [Paradigm Red Lines](#paradigm-red-lines)
- [Current Status](#current-status)
- [Tech Stack](#tech-stack)

---

## What Is This?

Project Genesis v2 is an **evolutionary experiment platform for AGI research**. It asks one question:

> *If we give a single-celled digital organism social ability, self-modification capability, and multiple different universes to live in -- will civilization emerge on its own?*

This is **not** a chatbot, not an agent framework, not a wrapper around LLMs. It is a **digital terrarium** where:

- Agents are **pure mathematical structures** (GenomeGraph with vector nodes, D=64)
- The environment is powered by **real LLM APIs** (DeepSeek, Gemini, MiMo, etc.)
- Evolution is driven by **energy economics** -- predict well or die
- Intelligence is measured by **7 cognitive probes**, not benchmark scores

---

## Core Philosophy

### From "Learning from Humans" to "Surpassing Humans"

Four-stage emergence path, driven naturally by **energy gradients** -- not hardcoded:

```
Phase 1  Learning      Prediction reward dominates   --> Agent learns human knowledge from LLM
         |                                             prediction reward marginally decreases
         v
Phase 2  Internalize   Compression reward dominates   --> Agent compresses knowledge into structured representations
         |                                             compression approaches limit
         v
Phase 3  Transcend     Exploration reward dominates   --> Agent explores beyond LLM knowledge
         |                                             Agent develops self-model
         v
Phase 4  Autonomous    Self-modification emerges      --> Agent optimizes its own topology
```

### Worst Case / Best Case

| | Outcome |
|---|---|
| **Worst case** | A batch of digital organisms that can quickly adapt across multiple LLMs, communicate with each other, and self-modify. Even if they haven't "understood" anything, these behaviors are already evolutionary phenomena worth studying. |
| **Best case** | Language emerges from social pressure, self-awareness sprouts from self-modification, and agents begin doing things humans didn't predict. |

Either way, it's honest science.

---

## Key Concepts

### Multi-LLM Islands

4 islands, each bound to a different LLM. Agents migrate between islands periodically.

**Why?** If an agent only lives with one LLM, it can survive by memorizing behavioral patterns (like a student memorizing exam answers). But when forced to live across multiple LLMs, it **must** learn:

- The **deep structure** shared across different LLMs (language patterns, logical consistency)
- **Transferable reasoning** (general causal inference, not pattern matching)
- **Environment-adaptive internal models** (distinguishing "physics" from "surface features")

> *Like a child raised across multiple cultures -- forced to learn "universal social rules"*
> *rather than memorizing one culture's behavioral patterns.*

### Energy = Life

Every tick costs energy (token usage, node count, edges, API costs, messages sent). Agents that predict well earn energy. **Zero energy = death.** This creates natural selection pressure:

| Pressure | Effect |
|---|---|
| Efficient agents | Survive -- use fewer tokens for better predictions |
| Bloated topologies | Die -- too many nodes/edges drain energy |
| Poor communicators | Die -- spamming messages wastes energy |

### Output Partition (528-dim)

Each agent's output vector is physically partitioned into functional zones:

```
+-------------------------------------------------------------------+
|  Agent Output Vector (528-dim)                                     |
|                                                                    |
|  [0:256]    Action    --> projected to LLM token space via matrix P|
|  [256:384]  Message   --> sent to grid neighbors via CommBus       |
|  [384:512]  State     --> feedback loop as working memory          |
|  [512:528]  SelfMod   --> mutation instructions for self-mod       |
+-------------------------------------------------------------------+
```

These are **physical interfaces of the universe** (like USB pin definitions), not business logic. The agent's internal GenomeGraph remains zero-prior, zero-pretrain.

### Self-Modification

The most ambitious design: the last 16 dimensions of output are interpreted as "surgery instructions" for the agent to modify its own topology.

- **70%** chance of degradation or death (simulating brain surgery risk)
- Requires **high energy surplus** (only "wealthy" agents can afford to try)
- Drives emergence of **self-model** and **causal reasoning** -- core AGI capabilities

### Social Layer

Agents sit on a 2D grid. Each agent has a "mouth" (128-dim message channel), "ears" (inbox), and neighbors. How to use them is up to the agent. Combined with social pressure (resource sharing, cooperation rewards), agents are **forced** to attempt meaningful signaling. Language emerges from the pressure of *"communicate or starve."*

### Exploration Bonus

```
ExplorationBonus = max(0, surprise_to_LLM - surprise_to_self)
```

When the agent does something the LLM considers impossible, but the agent itself can perfectly predict -- it earns bonus energy. This drives the transition from *"student"* (learning human knowledge) to *"explorer"* (discovering beyond human knowledge).

### 7 Cognitive Probes

Distinguishing **true understanding** from **overfitting**:

| Probe | Measures | Understanding | Overfitting |
|:---|:---|:---:|:---:|
| OOD Generalization | KL change on domain shift | < 2x | > 10x |
| Topology Modularity Q | Louvain community detection | > 0.4 | < 0.2 |
| Multi-Scale Consistency | Multi-window KL ratio | < 3.0 | > 20.0 |
| Multi-LLM Adaptation | Cross-model KL + migration speed | < 50 ticks | > 500 ticks |
| Communication Emergence | I(msg; action) mutual information | > 0.5 bits | ~ 0 |
| Self-Mod Efficiency | Survival rate + fitness delta | > 0.3 | < 0.05 |
| Exploration Effect | Exploration ratio + fitness change | fitness up | fitness down |

---

## Architecture

```
+---------------------------------------------------------------+
|              Multi-LLM Environment                             |
|         (DeepSeek / Gemini / MiMo / Mock-CA)                  |
+-----------------------^---------------------------------------+
                        |
             action / feedback (token semantic flow)
                        |
            [ Interaction Bus + Communication Bus ]
                        |
+-----------------------v---------------------------------------+
|              Agent Population (4 Islands)                      |
|                                                                |
|  +-- GenomeGraph (vector nodes, D=64, matrix weights)         |
|  |   +-- 12 mutation primitives (NEAT split, attention, ...)  |
|  |   +-- Forward pass (BLAS matrix multiply, Kahn topo-sort)  |
|  |   +-- Crossover (NEAT-style sexual reproduction)           |
|  |                                                            |
|  +-- Output Partition (528-dim)                                |
|  |   +-- Action  [0:256]   --> environment via projection P   |
|  |   +-- Message [256:384] --> neighbors via CommunicationBus |
|  |   +-- State   [384:512] --> working memory (feedback loop) |
|  |   +-- SelfMod [512:528] --> mutation instructions          |
|  |                                                            |
|  +-- Energy System (cost/reward per tick)                     |
|  +-- Selection  (starvation / entropy overflow / self-mod)    |
+---------------------------------------------------------------+
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/TimeCraker/genesis-v2.git
cd genesis-v2/genesis_v2

# Install dependencies
uv sync --extra dev

# Run tests (250 tests, ~4s)
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
├── docx/                                # Design documents
│   ├── AGI_v2_roadmap.md                # Full engineering blueprint (14 chapters)
│   ├── PLAN_LIST.md                     # Progress tracker
│   └── 设计理念思路.md                   # Design philosophy (Chinese)
│
├── genesis_v2/                          # Main project
│   ├── genesis_v2/                      # Source code
│   │   ├── agent/agent.py               # Agent entity (output-partitioned)
│   │   ├── genome/
│   │   │   ├── graph.py                 # GenomeGraph (vector nodes, D=64)
│   │   │   ├── mutate.py                # 12 mutation primitives
│   │   │   ├── forward.py               # BLAS forward pass
│   │   │   └── crossover.py             # NEAT crossover
│   │   ├── engine/
│   │   │   ├── tick.py                  # Tick loop (single + multi-island)
│   │   │   ├── metabolism.py            # Energy settlement
│   │   │   ├── reaper.py                # Death evaluation
│   │   │   └── selfmod.py               # Self-modification channel
│   │   ├── env/
│   │   │   ├── mock.py                  # Mock CA environment (free)
│   │   │   ├── multi_llm.py             # Multi-LLM environment
│   │   │   ├── budget.py                # Budget manager + fallback
│   │   │   └── embed.py                 # Frozen embedding projection
│   │   ├── social/
│   │   │   ├── comm_bus.py              # CommunicationBus (grid topology)
│   │   │   └── reputation.py            # Reputation system
│   │   ├── population/
│   │   │   ├── island.py                # Island class
│   │   │   └── migration.py             # Cross-LLM migration
│   │   ├── evolution/
│   │   │   ├── breeder.py               # Breeding strategy
│   │   │   ├── gen_memory.py            # Generational memory bank
│   │   │   └── survivor_bank.py         # Elite persistence
│   │   ├── metrics/probes/              # 7 cognitive probes
│   │   ├── translation/translator.py    # Vector <-> natural language
│   │   ├── storage/duckdb_store.py      # DuckDB telemetry
│   │   └── scripts/dashboard.py         # Streamlit dashboard
│   │
│   ├── configs/
│   │   ├── genesis_v2.yaml              # Physics constants + experiment params
│   │   └── backends.yaml                # LLM backend definitions
│   ├── tests/                           # 250 tests across 26 files
│   └── pyproject.toml
│
└── README.md
```

---

## v1 vs v2

| Dimension | v1 | v2 |
|:---|:---|:---|
| Node computation | Scalar (1 float/node) | **Vector (64-dim/node)** |
| Edge weight | Scalar float | **Matrix W in R^{DxD}** |
| Agent interaction | Fully independent | **Local communication (grid neighbors)** |
| Environment | Single LLM or Mock | **Multi-LLM islands** |
| Evolution | Asexual (clone + mutate) | **+ Crossover (NEAT-style)** |
| Selection pressure | Prediction + compression + variance | **+ Social + exploration + migration** |
| Self-modification | None | **16-dim output -> mutation instructions** |
| Cross-gen memory | None (state reset) | **GenerationalMemory** |
| Final goal | Metric curves | **Natural language conversation** |

---

## Paradigm Red Lines

<details>
<summary><b>Absolutely Forbidden</b></summary>

- SFT / RLHF / Supervised learning
- Prompt Engineering / Chain-of-Thought injection
- Agent Workflow / Tool calling / LangChain-style assembly
- Hand-written "cognitive modules" (Memory / Planner / Attention / S1-S2)
- Any `if token == "hello"` style business logic hardcoding

</details>

<details>
<summary><b>The Only Things Allowed</b></summary>

- Define **physical constants** (energy conversion rates, mutation probabilities, projection matrices)
- Define **physical interfaces** (output partition: action/message/state/self-modification)
- Define **pure math rewards** (KL divergence, MDL, mutual information, exploration bonus formulas)
- Define **physical rule stabilization** (LLM temperature = 0)

</details>

---

## Current Status

| Phase | Status | Key Result |
|:---|:---:|:---|
| Phase 0: Skeleton | **Done** | 81 tests, 10-agent x 100-tick smoke run |
| Phase 1: Social + Multi-LLM | **Done** | 130 tests, messages norm ~35, 4 islands 400/400 alive |
| Phase 2: Evolution + Exploration | **Done** | 193 tests, fitness rising across generations (2254 -> 6308) |
| Phase 3: Translation + Dialogue | **Done** | 227 tests, human-agent conversation working |
| Phase 4: Deep Experiments + Probes | **Done** | 250 tests, 7 probes, B-tier (3/7 probes passing) |

> **Current Tier: B** -- Evolution signals confirmed. A-tier requires 1 more probe passing.

---

## Tech Stack

| Layer | Tools |
|:---|:---|
| Language | Python 3.11+ |
| Computation | NumPy, SciPy |
| Configuration | Pydantic, PyYAML |
| Graph | NetworkX |
| Storage | DuckDB |
| HTTP | httpx |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, hypothesis |

---

## License

[MIT](LICENSE)

---

<br/>
<br/>

<div align="center" id="中文版">

# Project Genesis v2.0

**多 LLM 语义荒野 -- 自主演化 AGI 实验平台**

> **[English](#project-genesis-v20)** | 中文

<br/>

*在由多个 LLM 充当"多元物理法则"的信息宇宙里，*
*数字生命体通过能量守恒、社交压力、自我修改和跨代传承不断演化 --*
*从零开始发展出自己的认知结构。*
***无训练、无硬编码、无先验知识。***

</div>

---

## 目录

- [这是什么？](#这是什么)
- [核心理念](#核心理念)
- [关键概念](#关键概念)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [v1 与 v2 对比](#v1-与-v2-对比)
- [范式红线](#范式红线)
- [当前状态](#当前状态)
- [技术栈](#技术栈)

---

## 这是什么？

Project Genesis v2 是一个**面向 AGI 研究的演化实验平台**。它只问一个问题：

> *如果给一个单细胞数字生物社交能力、自我改造能力、和多个不同的宇宙去生活——文明会不会自己长出来？*

这**不是**聊天机器人，不是 Agent 框架，不是 LLM 的包装器。它是一个**数字生态缸**，其中：

- Agent 是**纯数学结构**（向量节点 GenomeGraph，D=64）
- 环境由**真实 LLM API** 驱动（DeepSeek、Gemini、MiMo 等）
- 演化由**能量经济学**驱动——预测得好就活，预测不好就死
- 智能由**7 项认知探针**衡量，而非基准测试分数

---

## 核心理念

### 从"学习人类"到"超越人类"

四阶段涌现路径，由**能量梯度自然驱动**——而非硬编码：

```
阶段一  学习期    预测奖励主导   --> Agent 从 LLM 学习人类知识
        |                           预测奖励边际收益递减
        v
阶段二  内化期    压缩奖励主导   --> Agent 将知识压缩为结构化表征
        |                           压缩趋于极限
        v
阶段三  超越期    探索奖励主导   --> Agent 探索 LLM 知识之外的领域
        |                           Agent 发展出自我模型
        v
阶段四  自主期    自我修改涌现   --> Agent 能优化自身拓扑结构
```

### 最坏情况 / 最好情况

| | 结果 |
|---|---|
| **最坏情况** | 得到一批能在多个 LLM 之间快速适应、互相通信、自我修改的数字生物——即使它们还没有"理解"任何东西，这些行为本身就已是值得研究的演化现象。 |
| **最好情况** | 看到语言在社交压力中涌现，看到自我意识在自我修改中萌芽，看到 Agent 开始做出人类没有预料到的行为。 |

无论哪种结果，都是诚实的科学。

---

## 关键概念

### 多 LLM 岛屿

4 座岛屿，每座绑定不同的 LLM。Agent 定期在岛屿之间迁移。

**为什么？** 如果 Agent 只和一个 LLM 生活，它可以靠记忆行为模式生存（就像学生背题库）。但当被迫在多个 LLM 之间生活时，它**必须**学会：

- 不同 LLM 共享的**深层结构**（语言规律、逻辑一致性）
- **可迁移的推理能力**（通用因果推理，而非模式匹配）
- **环境自适应的内部模型**（区分"物理法则"和"表面特征"）

> *就像一个在多个文化中长大的孩子——被迫学会"通用社交规则"，*
> *而非死记某个文化的行为模式。*

### 能量即生命

每个 tick 都要消耗能量（token 用量、节点数、边数、API 成本、消息发送量）。预测得好的 Agent 赚取能量。**能量归零 = 死亡。** 这创造了自然选择压力：

| 压力 | 效果 |
|---|---|
| 高效 Agent | 存活——用更少的 token 获得更好的预测 |
| 臃肿拓扑 | 死亡——太多节点/边消耗能量 |
| 低效通信 | 死亡——滥发消息浪费能量 |

### 输出分区（528 维）

每个 Agent 的输出向量被物理性地划分为功能区域：

```
+-------------------------------------------------------------------+
|  Agent 输出向量 (528 维)                                           |
|                                                                    |
|  [0:256]    动作区   --> 通过投影矩阵 P 映射到 LLM token 空间      |
|  [256:384]  消息区   --> 通过通信总线传递给网格邻居                  |
|  [384:512]  状态区   --> 回读为下一 tick 的工作记忆                  |
|  [512:528]  自我修改区 --> 映射到图谱变异指令                        |
+-------------------------------------------------------------------+
```

这些是**宇宙的物理接口**（如同 USB 针脚定义），不是业务逻辑。Agent 内部的 GenomeGraph 仍然零先验、零预训练。

### 自我修改

最大胆的设计：输出的最后 16 维被解释为"手术指令"，让 Agent 修改自身的拓扑结构。

- **70%** 概率恶化或死亡（模拟脑部手术的风险）
- 需要**高能量盈余**（只有"富裕"的 Agent 才负担得起尝试）
- 驱动**自我模型**和**因果推理**的涌现——这两个正是 AGI 的核心能力

### 社交层

Agent 被放置在二维网格上。每个 Agent 有一个"嘴巴"（128 维消息通道）、"耳朵"（收件箱）和邻居。如何使用它们由 Agent 自己决定。配合社交压力（资源共享、合作奖励），Agent 被**迫**尝试传递有意义的信号。语言从"不沟通就会饿死"的压力中涌现。

### 探索奖励

```
探索奖励 = max(0, LLM的惊讶程度 - Agent自身的惊讶程度)
```

当 Agent 做出 LLM 认为"不可能"的行为，但 Agent 自己完全能预测时，获得额外能量。这驱动了从"学生"（学习人类知识）到"探险家"（发现未知领域）的转变。

### 7 项认知探针

区分**真正的理解**和**过拟合**：

| 探针 | 衡量内容 | 真理解 | 纯拟合 |
|:---|:---|:---:|:---:|
| OOD 泛化 | KL 散度在域迁移时的变化 | < 2 倍 | > 10 倍 |
| 拓扑模块度 Q | Louvain 社区检测 | > 0.4 | < 0.2 |
| 多尺度一致性 | 多时间窗 KL 比率 | < 3.0 | > 20.0 |
| 多 LLM 适应性 | 跨模型 KL + 迁移速度 | < 50 tick | > 500 tick |
| 沟通涌现 | I(msg; action) 互信息 | > 0.5 bits | ~ 0 |
| 自我修改效率 | 存活率 + fitness 变化 | > 0.3 | < 0.05 |
| 探索效果 | 探索行为占比 + fitness 变化 | fitness 上升 | fitness 下降 |

---

## 系统架构

```
+---------------------------------------------------------------+
|              多 LLM 环境                                       |
|         (DeepSeek / Gemini / MiMo / Mock-CA)                  |
+-----------------------^---------------------------------------+
                        |
             动作 / 反馈 (token 语义流)
                        |
            [ 交互总线 + 通信总线 ]
                        |
+-----------------------v---------------------------------------+
|              Agent 种群 (4 座岛屿)                              |
|                                                                |
|  +-- GenomeGraph (向量节点, D=64, 矩阵权重)                    |
|  |   +-- 12 种变异原语 (NEAT 分裂、注意力、模块注入...)        |
|  |   +-- 前向计算 (BLAS 矩阵乘法, Kahn 拓扑排序)               |
|  |   +-- 交叉重组 (NEAT 式有性繁殖)                            |
|  |                                                            |
|  +-- 输出分区 (528 维)                                        |
|  |   +-- 动作区  [0:256]   --> 通过投影矩阵 P 作用于环境       |
|  |   +-- 消息区  [256:384] --> 通过通信总线传递给邻居           |
|  |   +-- 状态区  [384:512] --> 工作记忆 (反馈循环)             |
|  |   +-- 自我修改区 [512:528] --> 变异指令                     |
|  |                                                            |
|  +-- 能量系统 (每 tick 的成本/奖励结算)                        |
|  +-- 选择机制 (饥饿 / 熵溢出 / 自我修改致死)                   |
+---------------------------------------------------------------+
```

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/TimeCraker/genesis-v2.git
cd genesis-v2/genesis_v2

# 安装依赖
uv sync --extra dev

# 运行测试 (250 项测试, 约 4 秒)
uv run pytest -q

# 冒烟测试
uv run python -m genesis_v2 smoke

# Mock 环境运行 (免费, 无需 API 密钥)
uv run python -m genesis_v2 mock-loop --agents 10 --ticks 100

# 仪表盘 (Streamlit UI)
uv run python -m genesis_v2 dashboard
```

### LLM 配置

通过环境变量设置 API 密钥：

```bash
export GENESIS_DEEPSEEK_KEY=sk-xxx
export GENESIS_GEMINI_KEY=AIzaSy-xxx
export GENESIS_MIMO_KEY=xxx
export GENESIS_OPENAI_KEY=sk-xxx
```

后端定义在 `configs/backends.yaml`。添加新模型只需新增一条配置——零代码修改。

---

## 项目结构

```
AGI7/
├── docx/                                # 设计文档
│   ├── AGI_v2_roadmap.md                # 完整工程蓝图 (14 章)
│   ├── PLAN_LIST.md                     # 进度清单
│   └── 设计理念思路.md                   # 白话设计理念
│
├── genesis_v2/                          # 主项目
│   ├── genesis_v2/                      # 源代码
│   │   ├── agent/agent.py               # Agent 实体 (输出分区)
│   │   ├── genome/
│   │   │   ├── graph.py                 # GenomeGraph (向量节点, D=64)
│   │   │   ├── mutate.py                # 12 种变异原语
│   │   │   ├── forward.py               # BLAS 前向计算
│   │   │   └── crossover.py             # NEAT 交叉重组
│   │   ├── engine/
│   │   │   ├── tick.py                  # Tick 循环 (单岛/多岛)
│   │   │   ├── metabolism.py            # 能量结算
│   │   │   ├── reaper.py                # 死亡判定
│   │   │   └── selfmod.py               # 自我修改通道
│   │   ├── env/
│   │   │   ├── mock.py                  # Mock CA 环境 (免费)
│   │   │   ├── multi_llm.py             # 多 LLM 环境
│   │   │   ├── budget.py                # 预算管理 + 自动降级
│   │   │   └── embed.py                 # 冻结 Embedding 投影
│   │   ├── social/
│   │   │   ├── comm_bus.py              # 通信总线 (网格拓扑)
│   │   │   └── reputation.py            # 声誉系统
│   │   ├── population/
│   │   │   ├── island.py                # 岛屿模型
│   │   │   └── migration.py             # 跨 LLM 迁移
│   │   ├── evolution/
│   │   │   ├── breeder.py               # 繁殖策略
│   │   │   ├── gen_memory.py            # 跨代记忆库
│   │   │   └── survivor_bank.py         # 精英持久化
│   │   ├── metrics/probes/              # 7 项认知探针
│   │   ├── translation/translator.py    # 向量 <-> 自然语言翻译
│   │   ├── storage/duckdb_store.py      # DuckDB 遥测存储
│   │   └── scripts/dashboard.py         # Streamlit 仪表盘
│   │
│   ├── configs/
│   │   ├── genesis_v2.yaml              # 物理常数 + 实验参数
│   │   └── backends.yaml                # LLM 后端定义
│   ├── tests/                           # 250 项测试, 26 个文件
│   └── pyproject.toml
│
└── README.md
```

---

## v1 与 v2 对比

| 维度 | v1 | v2 |
|:---|:---|:---|
| 节点计算 | 标量 (1 float/节点) | **向量 (64 维/节点)** |
| 边权重 | 标量 float | **矩阵 W in R^{DxD}** |
| Agent 交互 | 完全独立 | **局部通信 (网格邻居)** |
| 环境 | 单一 LLM 或 Mock | **多 LLM 岛屿** |
| 演化方式 | 无性繁殖 (克隆+变异) | **+ 交叉重组 (NEAT 式有性繁殖)** |
| 选择压力 | 预测+压缩+行为方差 | **+ 社交+探索+迁移适应性** |
| 自我修改 | 无 | **16 维输出 -> 变异指令** |
| 跨代记忆 | 无 (状态清零) | **GenerationalMemory** |
| 最终目标 | 指标曲线 | **与 Agent 自然语言对话** |

---

## 范式红线

<details>
<summary><b>系统内部绝对禁止</b></summary>

- SFT / RLHF / 监督学习
- Prompt Engineering / Chain-of-Thought 注入
- Agent Workflow / 工具调用 / LangChain 类组装
- 手写任何"认知模块"（Memory / Planner / Attention / S1-S2）
- 任何 `if token == "hello"` 类的业务逻辑硬编码

</details>

<details>
<summary><b>系统唯一允许</b></summary>

- 定义**物理常数**（能量换算率、突变概率、投影矩阵）
- 定义**物理接口**（输出分区：动作/消息/状态/自我修改）
- 定义**纯数学奖惩**（KL 散度、MDL、互信息、探索奖励公式）
- 定义**物理规则的稳定化**（LLM temperature = 0）

</details>

---

## 当前状态

| 阶段 | 状态 | 关键结果 |
|:---|:---:|:---|
| Phase 0: 骨架 | **完成** | 81 项测试, 10 个体 x 100 tick 冒烟运行 |
| Phase 1: 社交 + 多 LLM | **完成** | 130 项测试, 消息范数 ~35, 4 岛屿 400/400 存活 |
| Phase 2: 演化 + 探索 | **完成** | 193 项测试, fitness 跨代上升 (2254 -> 6308) |
| Phase 3: 翻译 + 对话 | **完成** | 227 项测试, 人-Agent 对话可用 |
| Phase 4: 深度实验 + 探针 | **完成** | 250 项测试, 7 项探针, B 级 (3/7 探针通过) |

> **当前等级：B 级** -- 演化信号已确认。A 级需要再通过 1 项探针。

---

## 技术栈

| 层级 | 工具 |
|:---|:---|
| 语言 | Python 3.11+ |
| 计算 | NumPy, SciPy |
| 配置 | Pydantic, PyYAML |
| 图计算 | NetworkX |
| 存储 | DuckDB |
| HTTP | httpx |
| 仪表盘 | Streamlit, Plotly |
| 测试 | pytest, hypothesis |

---

## 开源协议

[MIT](LICENSE)
