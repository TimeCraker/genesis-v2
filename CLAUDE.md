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

**参考实现**（v1 已验证的代码，在上层目录）：

6. `C:\Users\TimeCraker\Desktop\AGI\genesis\genesis\` — v1 完整实现，以下是关键参考文件：
   - `genome/graph.py` — GenomeGraph 数据结构（v2 需要从标量升级为向量）
   - `genome/mutate.py` — 7 种变异原语（v2 需要扩展到 12 种）
   - `genome/forward.py` — 前向计算 BLAS bundle（v2 需要改为矩阵乘法）
   - `agent/agent.py` — Agent dataclass（v2 需要添加输出分区）
   - `engine/metabolism.py` — 能量公式（v2 需要新增 API 成本、社交奖励、探索奖励）
   - `engine/reaper.py` — 死亡判定（v2 需要新增自我修改致死）
   - `engine/tick.py` + `tick_core.py` — Tick 循环（v2 需要加入社交阶段）
   - `population/island.py` + `migration.py` — 岛屿模型
   - `env/mock.py` — Mock 环境
   - `env/embed.py` — 冻结 Embedding 投影
   - `storage/duckdb_store.py` — DuckDB 存储
   - `cli.py` — CLI 入口
   - `config.py` — Pydantic 配置模型

## 当前任务：Phase 0 — 骨架

**目标**：`pytest -q` 全绿 + `python -m genesis_v2` 能跑一个 10 个体 × 100 tick 的不崩进程。

**严格按以下顺序实现**（每个子任务完成后再进入下一个）：

### P0.1 项目结构 + CLI 存根

- 完善 `genesis_v2/cli.py`：实现 `python -m genesis_v2` 入口，支持 `smoke` 和 `mock-loop` 子命令
- 添加 `genesis_v2/__main__.py`
- 运行 `uv sync --extra dev` 安装依赖
- 验收：`python -m genesis_v2` 能打印版本

### P0.2 GenomeGraph（向量节点版）

核心变化：每个节点从标量 `float` 升级为向量 `np.ndarray[D_node=64]`，边权重从 `float` 升级为矩阵 `np.ndarray[D_dst, D_src]`。

文件：`genesis_v2/genome/graph.py`

- `NodeType` Enum（INPUT, OUTPUT, HIDDEN, GATING）
- `EdgeKind` Enum（FORWARD, SHORTCUT, RECURRENT）
- `Node` dataclass（id, type, dim=D_node）
- `Edge` dataclass（id, src, dst, kind, weight: np.ndarray, gated_by）
- `GraphConfig` pydantic 模型（node_dim, input_nodes, output_nodes 各区数量, initial_hidden_nodes, initial_edge_density）
- `GenomeGraph` 类：
  - `nodes: dict[int, Node]`、`edges: dict[int, Edge]`、`input_nodes`、`output_nodes`
  - `_last_hidden: dict[int, np.ndarray]`（循环边残留，64 维向量）
  - `_forward_cache_revision`（缓存失效计数器）
  - `new_genome_graph(cfg, rng)` 构造器
  - `node_count()` / `edge_count()` / `hidden_count()` / `copy()` / `entropy()`
  - 输出分区辅助方法：`get_action_nodes()` / `get_message_nodes()` / `get_state_nodes()` / `get_selfmod_nodes()`

关键设计：
- INPUT 节点的 dim = D_node（64），但输入会被 reshape 为 N_input × D_node 的矩阵
- OUTPUT 节点总数 = (256+128+128+16) / 64 = 9 个输出节点（4 action + 2 message + 2 state + 1 selfmod）
- copy() 保留 node/edge ID 以追踪谱系，清零 _last_hidden
- 确定性：所有遍历用 sorted(dict.keys())

测试：`tests/test_genome_graph.py`

### P0.3 变异原语（12 种）

文件：`genesis_v2/genome/mutate.py`

先实现 7 种 v1 升级版（向量化改造）：
0. `ADD_FORWARD_EDGE` — 随机 (src, dst) 对，weight = rng.standard_normal((dst_dim, src_dim)) * 0.3
1. `ADD_SHORTCUT_EDGE` — 同上，kind=SHORTCUT
2. `ADD_RECURRENT_EDGE` — 同上，kind=RECURRENT
3. `ADD_HIDDEN_NODE` — NEAT 边分裂：选一条非 recurrent 边 u→v，替换为 u→h→v，保留近似函数
4. `ADD_GATING_NODE` — 创建 GATING 节点，对随机边做元素级 sigmoid 门控
5. `PERTURB_WEIGHT` — 对随机边的权重矩阵加高斯噪声（sigma=0.1）
6. `DELETE_RANDOM_EDGE` — 删边（拒绝删最后一条边）

再实现 5 种新增：
7. `ADD_ATTENTION_GROUP` — 创建一组 Q/K/V 节点 + 输出节点（简化注意力）
8. `ADD_MODULE` — 注入 2-5 节点的子图
9. `SPLIT_NODE_DIM` — 高维节点拆分为多个低维节点（暂不实现，留接口）
10. `MERGE_NODES` — 多个节点合并为高维节点（暂不实现，留接口）
11. `ADD_COMM_EDGE` — 连接到消息区输出节点的边

`mutate(rng, allowed=None)` 调度器，从 allowed 集合中随机选一个执行。

测试：`tests/test_genome_mutate.py`（含 hypothesis 属性测试）

### P0.4 前向计算（向量版）

文件：`genesis_v2/genome/forward.py`

- `forward(input: np.ndarray) -> np.ndarray`：BLAS bundle 向量版
  - 拓扑排序（Kahn，FORWARD+SHORTCUT 边，ties by NodeID）
  - 每个节点的聚合：`z = W @ x_stack + b`（矩阵-向量乘法）
  - 激活：hidden=tanh, gating=sigmoid, output=linear
  - 循环边读 `_last_hidden`（上一 tick 的 64 维向量）
  - 输出：所有 OUTPUT 节点的激活值拼接为 528 维向量
- `forward_scalar()` 黄金对照（纯 Python 循环版）
- `_rebuild_forward_bundles()` — 缓存编译（权重矩阵堆叠）
- `touch_forward_cache()` — 拓扑/权重变更时失效
- `reset_state()` — 清零 _last_hidden

测试：`tests/test_genome_forward.py`（逐 tick 对齐 BLAS 和 scalar，无 NaN/Inf，循环跨 tick 漂移）

### P0.5 Agent 本体

文件：`genesis_v2/agent/agent.py`

- `Agent` dataclass（见 roadmap §3.1 的完整字段）
- `new_agent(id, genome, initial_energy)` 构造器
- 输出分区常量：`D_ACTION=256, D_MESSAGE=128, D_STATE=128, D_SELFMOD=16, D_OUT=528`
- `split_output(output_vec)` → (action, message, state, selfmod)

测试：`tests/test_agent.py`

### P0.6 能量结算 & 死亡判定

文件：`genesis_v2/engine/metabolism.py`

- `tick_cost(phy, agent)` — 成本公式（含 v2 新项 epsilon·API_cost + zeta·messages）
- `tick_reward(phy, agent, pop_mean)` — 奖励公式（含 v2 新项 adaptation + social + exploration）
  - Phase 0 中 v2 新项暂返回 0，后续 Phase 补全
- `apply_metabolism(phy, agent, pop_mean)` — 计算 delta_energy 并更新 agent.energy/fitness

文件：`genesis_v2/engine/reaper.py`

- `evaluate_death(agent, phy)` — 检查死亡条件（starvation, topology_entropy, selfmod_fatal）
- `sweep_island(agents, phy)` — 扫描并杀死死亡 Agent

测试：`tests/test_metabolism.py`, `tests/test_reaper.py`

### P0.7 存储层

文件：`genesis_v2/storage/duckdb_store.py`

- 单文件嵌入式 DuckDB
- `generations` 表（generation, island_id, alive_count, mean_fitness, mean_energy, ...）
- `ticks` 表（tick, agent_id, energy, fitness, prediction_error, compression, ...）
- `record_tick(agent)` / `flush()` API

测试：`tests/test_duckdb_store.py`

### P0.8 Tick 引擎（Phase 0 简化版）

文件：`genesis_v2/engine/tick.py` + `tick_core.py`

Phase 0 的 tick 循环（简化版，不含社交）：
```
observe → forward(all) → interact(pop_mean) → metabolize → reap
```

- `island_step_sync(agents, env, phy, reaper, store, rng)` — 核心单帧函数
- `TickEngine` class — run_ticks(n, hz=100)

文件：`genesis_v2/env/mock.py`

- `MockMathEnvironment` — 多规则 CA（Rule110 / Rule30 / Rule90）
- `observe()` → float32 向量
- `interact(action)` → float32 反馈
- `true_distribution(history)` → 预测目标

### P0.9 CLI + 验收

- `python -m genesis_v2 smoke` — 单 Agent，10 forward passes
- `python -m genesis_v2 mock-loop --agents 10 --ticks 100 --dim 64` — Mock 闭环
- `pytest -q` 全绿

## 工作原则

1. **先读 v1 参考代码，再写 v2 代码**。v1 的逻辑、测试模式、命名规范都值得继承。
2. **每个模块写完就写测试**。不要等全部写完再测试。
3. **保持确定性**。所有遍历用 sorted(dict.keys())，随机数用显式 rng。
4. **向量节点的核心改变**：边权重从 float → ndarray，节点输出从 float → ndarray。BLAS bundle 从存储一维权重向量 → 存储二维权重矩阵。
5. **Phase 0 不实现**：社交层、多 LLM 环境、Crossover、GenerationalMemory、自我修改、探索奖励、翻译层。这些留到 Phase 1-3。
6. **Phase 0 中 v2 新的奖励项（adaptation, social, exploration）暂返回 0**，只搭好接口。

## 运行环境

- Python 3.11+
- 包管理：`uv`（优先）或 `pip`
- 安装：`cd genesis_v2 && uv sync --extra dev`
- 测试：`uv run pytest -q`
- 运行：`uv run python -m genesis_v2 smoke`

## 项目目录

```
AGI7/
├── docx/                    ← 文档（已写好，只读）
├── genesis_v2/
│   ├── pyproject.toml       ← 已写好
│   ├── configs/             ← 已写好
│   ├── experiments/         ← 已写好
│   ├── genesis_v2/          ← 代码（你来写）
│   ├── tests/               ← 测试（你来写）
│   └── scripts/             ← 后续写
```

## 开始

1. 先读完上述所有文件
2. 从 P0.1 开始
3. 每完成一个子任务，更新 `docx/PLAN_LIST.md` 勾选对应项
4. 每次会话结束，在 `docx/PLAN_LIST.md` 的"会话日志"追加记录
