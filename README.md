<div align="center">

<img src="https://raw.githubusercontent.com/TimeCraker/genesis-v2/main/.github/banner.svg" alt="Genesis v2 Banner" width="100%"/>

# 🧬 Project Genesis v2.0

**Multi-LLM Semantic Wilderness · Autonomous AGI Evolution Platform**

[![CI](https://github.com/TimeCraker/genesis-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/TimeCraker/genesis-v2/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Other-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-≥3.12-3776ab?logo=python&logoColor=white)](https://www.python.org)
[![Tests](https://img.shields.io/badge/Tests-24_passing-22c55e)](tests/)
[![DuckDB](https://img.shields.io/badge/Storage-DuckDB-fbbf24)](https://duckdb.org)

> **A self-driving population of LLM agents whose genomes mutate, compete, and inherit — in a closed local loop with no cloud lock-in.**

</div>

---

## 📑 目录

- [🎯 项目定位](#-项目定位)
- [🏗️ 进化循环架构](#-进化循环架构)
- [🚀 快速开始](#-快速开始)
- [🧠 核心概念](#-核心概念)
- [🗂️ 项目结构](#-项目结构)
- [🛠️ 技术栈](#-技术栈)
- [🧪 测试](#-测试)
- [🗺️ 路线图](#-路线图)
- [📜 License](#-license)

---

## 🎯 项目定位

Genesis v2 是一个**纯本地、零云依赖**的 LLM Agent 进化实验平台。  
核心理念：**用多 LLM 互相打分当环境，让 Agent 基因组在"语义荒原"里自主演化。**

它**不是**：

- ❌ ChatGPT 套壳
- ❌ 单一 Agent 框架
- ❌ 需要 GPU / 云服务的训练平台

它是：

- ✅ 一个**闭环的进化模拟器**（6 个阶段：出生 → 变异 → 探索 → 评估 → 收割 → 继承）
- ✅ **多 LLM 互为环境**（DeepSeek 提问 / 智谱 评分 / SiliconFlow 探索）
- ✅ **可插拔的持久化**（DuckDB + CommBus + Survivor Bank + Island Topology）
- ✅ **可观测的进化过程**（Streamlit Dashboard + 24 个测试守住回归）

---

## 🏗️ 进化循环架构

<div align="center">

<img src="https://raw.githubusercontent.com/TimeCraker/genesis-v2/main/.github/architecture.svg" alt="Genesis v2 Architecture" width="100%"/>

</div>

**6 个阶段**（每 0.5 Hz tick 一次）：

| 阶段 | 含义 | 关键模块 |
|------|------|----------|
| 🥚 **BIRTH** | 育种者生成新基因 | `Breeder` |
| 🧬 **MUTATE** | 基因组交叉 + 噪声 | `Genome.crossover / mutate` |
| 🔭 **EXPLORE** | 在多 LLM 语义环境试错 | `MultiLLMEnv` |
| ⚖️ **EVALUATE** | 同行评分 + 自适应探针 | `Reputation / Probes` |
| 💀 **REAP** | 淘汰低声誉基因组 | `Reaper` |
| 🏆 **INHERIT** | 精英进入 Survivor Bank | `SurvivorBank` |

---

## 🚀 快速开始

### 1. 安装（推荐 uv）

```bash
git clone https://github.com/TimeCraker/genesis-v2.git
cd genesis-v2/genesis_v2
uv sync --extra dev
```

### 2. 跑测试（24 个测试应该全绿）

```bash
uv run pytest -v
```

### 3. 启动 Dashboard

```bash
uv run streamlit run scripts/dashboard.py
# 默认 http://localhost:8501
```

### 4. 跑一次小规模进化实验

```bash
uv run python -m genesis_v2.cli run --pop 16 --ticks 100
```

---

## 🧠 核心概念

| 概念 | 一句话解释 |
|------|------------|
| **Genome** | Agent 的"基因"——一组可变异、可交叉的 prompt 模板 + 工具调用偏好 |
| **Reputation** | 由同行 + 探针共同打分的"生存能力"指标 |
| **Island** | 多个子种群，通过 migration 交换基因（避免局部最优） |
| **CommBus** | Agent 之间的发布订阅消息总线（去中心化协作） |
| **Survivor Bank** | 高分基因组持久化池，新一代 birth 时优先采样 |
| **Probe** | 评估 Agent 真实能力的"自适应考卷"（不被刷题刷分） |
| **Self-Mod** | Agent 修改自己 prompt 的能力（受 reputation gate） |

---

## 🗂️ 项目结构

```
genesis-v2/
├── genesis_v2/
│   ├── genesis_v2/                # 主包
│   │   ├── __init__.py
│   │   ├── cli.py                 # CLI 入口
│   │   ├── breeder.py             # 基因育种
│   │   ├── genome.py              # 基因组模型
│   │   ├── multi_llm_env.py       # 多 LLM 语义环境
│   │   ├── reputation.py          # 声誉系统
│   │   ├── probes.py              # 自适应探针
│   │   ├── reaper.py              # 淘汰机制
│   │   ├── survivor_bank.py       # 精英持久化
│   │   ├── comm_bus.py            # 通信总线
│   │   ├── island.py              # 岛拓扑
│   │   ├── gen_memory.py          # 谱系树
│   │   ├── selfmod.py             # 自我修改
│   │   ├── translator.py          # 协议转换
│   │   ├── mock_env.py            # 测试用 mock LLM
│   │   ├── duckdb_store.py        # DuckDB 存储
│   │   ├── config.py              # 配置
│   │   └── scripts/
│   │       └── dashboard.py       # Streamlit 看板
│   ├── tests/                     # 24 个测试
│   │   ├── test_*.py
│   │   └── __init__.py
│   ├── configs/                   # 实验配置
│   ├── data/                      # 运行时数据
│   ├── experiments/               # 实验记录
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── uv.lock
│   ├── README.md                  # 子包 README
│   └── README_chinese.md
├── .github/
│   ├── banner.svg                 # 仓库 banner
│   ├── architecture.svg           # 架构图
│   └── workflows/ci.yml
├── docx/                          # 实验笔记
├── AGENTS.md
├── CLAUDE.md
├── LICENSE
└── README.md                      # ← 你在这里
```

---

## 🛠️ 技术栈

<div align="center">

| 领域 | 选型 | 理由 |
|------|------|------|
| 语言 | **Python 3.12+** | dataclass + asyncio + match 语法 |
| 存储 | **DuckDB** | 列式、单文件、零部署 |
| LLM | **OpenAI 兼容协议** | 一行切到 DeepSeek / 智谱 / SiliconFlow |
| Dashboard | **Streamlit + Plotly** | 写脚本即出图 |
| 演化 | 自研 + `networkx` | 图结构表达 island migration |
| 测试 | **pytest + hypothesis** | 24 个测试 + property-based |
| 包管理 | **uv** | 快 10×，锁文件可靠 |
| CI | **GitHub Actions** | 已配 ruff + pytest |

</div>

---

## 🧪 测试

```bash
cd genesis_v2
uv run pytest -v
# 24 passed in ~15s
```

测试覆盖：

- 单元测试：Genome mutate / crossover / forward / graph
- 集成测试：Breeder / Reaper / SurvivorBank / Island migration
- 探针测试：Probes / Reputation / SelfMod gate
- 持久化测试：DuckDB store round-trip
- Mock 环境：MockLLMEnv（不消耗真实 token）

---

## 🗺️ 路线图

- [x] 6 阶段进化闭环
- [x] DuckDB 持久化
- [x] Island topology + migration
- [x] Multi-LLM semantic environment
- [x] Self-mod with reputation gate
- [x] Streamlit dashboard
- [x] 24 测试守住回归
- [x] GitHub Actions CI
- [ ] LLM-as-Judge 模式（探针评分完全交给 LLM）
- [ ] 跨实验 lineage 可视化
- [ ] Agent 工具调用 benchmark
- [ ] 多目标优化（能耗 vs 性能 vs 安全）

---

## 📜 License

详见 [LICENSE](LICENSE)。

<div align="center">

**⭐ 如果你也相信"智能来自选择压力" — star 一下 ⭐**

</div>
