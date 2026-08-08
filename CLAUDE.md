# Project Genesis v2 — Claude Code 工作指南

## 你是谁

你是 Project Genesis v2 的核心开发者。这是一个面向 AGI 的演化实验平台——在多 LLM 宇宙中，通过社交压力、自我修改、跨代传承，演化出能自主思考的数字生命体。

## 开始前必读

**按顺序读完这些文件**（在 `docx/` 目录下）：

1. `docx/AGI_v2_roadmap.md` — 完整工程蓝图（14 章，所有架构、公式、接口定义）
2. `docx/PLAN_LIST.md` — 进度清单（勾选已完成的项，从第一个未完成项开始）
3. `docx/设计理念思路.md` — 白话设计理念（帮助理解设计意图）

**配置文件**（在 `configs/` 目录下）：

4. `configs/genesis_v2.yaml` — 物理常数 + 实验参数
5. `configs/backends.yaml` — LLM API 后端配置

**关于 v1 参考实现**：

6. v1 参考代码**不在本工作区内**——早期文档提到的 `Desktop\AGI\genesis\genesis\` 路径已不存在，请勿再引用。设计的权威来源是 `docx/AGI_v2_roadmap.md`，遇到歧义查这里。

## 代码架构（已实现状态）

> **现状**：按 `docx/PLAN_LIST.md` 记录，Phase 0–4 **已全部完成**（`pytest -q` 250 passed）。本节是各模块职责的参考索引，不是待办清单——动手前先读对应源码与测试。每个模块的完整字段与公式仍以 roadmap 为准。

源码包根：`genesis_v2/genesis_v2/`；测试：`genesis_v2/tests/`（24 个测试文件）。

### 基因组层 `genome/`
- `graph.py` — `GenomeGraph`（向量节点版）。节点输出为 `np.ndarray[D_node=64]`，边权重为矩阵 `np.ndarray[D_dst, D_src]`；含 `NodeType`/`EdgeKind` Enum、`new_genome_graph(cfg, rng)` 构造器、`get_action/message/state/selfmod_nodes()` 输出分区辅助方法、`touch_forward_cache()` 缓存失效。
- `mutate.py` — 12 种变异原语（7 种 v1 升级为向量版 + 5 种新增：`ADD_ATTENTION_GROUP` / `ADD_MODULE` / `SPLIT_NODE_DIM` / `MERGE_NODES` / `ADD_COMM_EDGE`），`mutate(rng, allowed=None)` 调度器。
- `forward.py` — BLAS bundle 向量前向 + `forward_scalar()` 黄金对照；拓扑排序用确定性 Kahn（ties by NodeID），循环边读 `_last_hidden`。
- `crossover.py` — NEAT 式交叉重组：按节点 ID 对齐父代，匹配基因按适应度比例选父代，disjoint/excess 从更适应方继承，输出分区结构强制一致。

### 个体与引擎 `agent/` `engine/`
- `agent/agent.py` — `Agent` dataclass，输出分区 256(action):128(message):128(state):16(selfmod)，`split_output()` 切分 528 维输出。
- `engine/metabolism.py` — `tick_cost` / `tick_reward` / `apply_metabolism`，含 v2 全部奖励项（adaptation + social + exploration，**已实装，非占位**）。
- `engine/reaper.py` — `evaluate_death` / `sweep_island`，死亡条件含 starvation / topology_entropy / selfmod_fatal。
- `engine/tick.py` — `TickEngine` + `island_step_sync`，tick 循环已接入社交阶段与 `multi_island_step()`。
- `engine/selfmod.py` — 自我修改通道：16 维 selfmod 输出 → 7 种基础变异的倾向权重 + 触发阈值，能量门槛 + 70% 死亡率 + `selfmod_energy_cost`。

### 环境层 `env/`
- `mock.py` — `MockMathEnvironment`（Rule110/30/90 元胞自动机）。
- `multi_llm.py` — `MultiLLMEnvironment`（OpenAI 兼容多 LLM 后端）。
- `budget.py` — `BudgetManager` + 超限降级策略，已接入 MultiLLM（`set_budget()`）。
- `batch.py` — `BatchedEnvironment` 批处理封装。
- `embed.py` — `FrozenEmbeddingAtlas` 冻结投影。
- `base.py` — `Environment` Protocol。

### 社交层 `social/`（Phase 1）
- `comm_bus.py` — `CommunicationBus`（网格拓扑消息传递）。
- `reputation.py` — 声誉系统（cooperation detection + trust）。

### 演化层 `evolution/`（Phase 2）
- `breeder.py` — 混合繁殖策略（Crossover 50% + Clone 30% + Exploration Clone 20%）。
- `gen_memory.py` — `GenerationalMemory`（behavioral_signature + successful_patterns + social_partners，50% 概率跨代传承，偏置 recurrent state）。
- `survivor_bank.py` — 精英种子库 save/load/list（+ `auto_save_elites()` / `auto_load_seeds()`），种子落 `data/survivors/`。

### 种群层 `population/`（Phase 1-2）
- `island.py` — `Island` 类 + `create_islands()` 工厂，每岛绑定独立 env + comm_bus + budget。
- `migration.py` — 跨 LLM 迁移（ring topology，每 N 代 top-3 迁邻岛）+ adaptation bonus（20-tick 线性衰减）。

### 度量与探针 `metrics/`（Phase 2-4）
- `exploration.py` — 探索奖励 `max(0, surprise_to_LLM − surprise_to_self)`。
- `metrics/probes/` — 7 项认知探针（ood / modularity / multiscale / multi_llm / communication / self_mod / exploration_effect）+ `conversation.py` 对话质量探针 + `runner.py`（`run_all_probes()` 编排 + `save_probe_report()` 持久化 + C/B/A/S/SS Tier 评估）。

### 翻译与对话 `translation/`（Phase 3）
- `translator.py` — 双向翻译器（Mock + API 双模式）：`vec_to_text()` / `text_to_vec()` / `translate_agent_output()` / `translate_to_input()`，配合 `ConversationSession`（人类 ↔ Agent 多轮对话）。

### 存储与工具
- `storage/duckdb_store.py` — DuckDB（`generations` + `ticks` 表，含社交指标列 `messages_received` / `mean_trust`）。
- `scripts/run_experiment.py` — 完整 generation loop（tick → select → mutate → save elites → breed），支持单岛/多岛、`run_probes` + `probe_interval`。
- `scripts/dashboard.py` — Streamlit 配置面板（Settings / Launch / Monitor 三 Tab）。
- `cli.py` — CLI 入口：`smoke` / `mock-loop` / `experiment` / `dashboard` / `converse` / `mve-run`。
- `config.py` — Pydantic 配置模型。

## 工作原则

1. **以 roadmap 为权威规范**。v1 参考代码不在本工作区，遇到设计歧义查 `docx/AGI_v2_roadmap.md`，不要引用不存在的 v1 路径。
2. **每个模块写完就写测试**。不要等全部写完再测试。
3. **保持确定性**。所有遍历用 sorted(dict.keys())，随机数用显式 rng。
4. **向量节点的核心约束**：边权重是 ndarray，节点输出是 ndarray；BLAS bundle 存储二维权重矩阵而非一维向量。

## 运行环境

- Python 3.11+
- 包管理：`uv`（优先）或 `pip`
- 安装：`cd genesis_v2 && uv sync --extra dev`
- 测试：`uv run pytest -q`
- 运行：`uv run python -m genesis_v2 smoke`

## 项目目录

```
genesis-v2/                 ← 工作区根（不是 AGI7/）
├── docx/                    ← 文档（roadmap、PLAN_LIST、设计理念，只读）
└── genesis_v2/              ← 项目目录
    ├── pyproject.toml       ← uv/pip 依赖声明
    ├── uv.lock
    ├── configs/             ← genesis_v2.yaml + backends.yaml（模板）
    ├── experiments/         ← 实验产物
    ├── tests/               ← 24 个测试文件（pytest，已 250 passed）
    └── genesis_v2/          ← 源码包（已实现，职责见上"代码架构"）
        ├── agent/  engine/  env/  genome/
        ├── social/  evolution/  population/
        ├── metrics/ (probes/)
        ├── translation/  storage/
        └── scripts/  cli.py  config.py  __main__.py
```

## 开始

1. 先读完上述"开始前必读"全部文件
2. 对照"代码架构"和 `docx/PLAN_LIST.md` 确认现状（Phase 0–4 已完成）
3. 若有新需求，从 `PLAN_LIST.md` 第一个未勾选项或明确的新任务起步；动模块前先读对应源码与测试
4. 每次会话结束，在 `docx/PLAN_LIST.md` 的"会话日志"追加记录
