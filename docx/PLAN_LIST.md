# Project Genesis v2 进度清单（PLAN LIST）

> 本文件与 [AGI_v2_roadmap.md](./AGI_v2_roadmap.md) 的里程碑一一对应。
> 每次完成一项，勾上 `[x]`；每次会话末尾，在"会话日志"追加一条。

**实现技术栈**：Python 3.12 / NumPy / SciPy / Pydantic / NetworkX / DuckDB / httpx / Streamlit / pytest + hypothesis。

---

## Phase 0：骨架（目标：1 周）

> 目的：让 `pytest -q` 全绿，并能用 `python -m genesis_v2` 跑一个 10 个体 × 100 tick 的不崩进程。

### P0.1 项目结构
- [x] `genesis_v2/pyproject.toml`（运行依赖 + `[dev]` extra）
- [x] 目录树完整创建
- [x] `configs/genesis_v2.yaml`（物理常数模板）
- [x] `configs/backends.yaml`（LLM 后端配置模板）
- [x] `genesis_v2/cli.py` 存根（`python -m genesis_v2` 能打印版本 + 跑 smoke）
- [x] `uv sync --extra dev` 完成（uv 0.11.11 + Python 3.12.13）

### P0.2 GenomeGraph 向量节点版
- [x] `genesis_v2/genome/graph.py`：`NodeType` / `EdgeKind` Enum、`Node` / `Edge` dataclass（向量权重）、`GraphConfig`、`GenomeGraph` 类
- [x] 节点状态从标量 → `np.ndarray[D_node=64]`
- [x] 边权重从 `float` → `np.ndarray[D_dst, D_src]`
- [x] 构造器 `new_genome_graph(cfg, rng)`
- [x] `node_count()` / `edge_count()` / `hidden_count()` / `copy()` / `entropy()`
- [x] 基础单元测试

### P0.3 变异原语（12 种）
- [x] v1 的 7 种升级为向量版
- [x] `ADD_ATTENTION_GROUP`：Q/K/V 向量节点 + softmax 注意力
- [x] `ADD_MODULE`：注入子图
- [x] `SPLIT_NODE_DIM`：高维节点拆分（留接口）
- [x] `MERGE_NODES`：多节点合并（留接口）
- [x] `ADD_COMM_EDGE`：连接到消息输出通道
- [x] `mutate()` 调度器
- [x] 变异不变式测试

### P0.4 前向计算（向量版）
- [x] `forward(input)`：BLAS bundle 向量版（矩阵-向量乘法）
- [x] `forward_scalar` 黄金对照实现
- [x] 拓扑/权重变更触发 `touch_forward_cache()`
- [x] 确定性 Kahn 拓扑排序（继承 v1）
- [x] 循环边读取 `_last_hidden`（64 维向量残留）
- [x] 门控节点：向量元素级 sigmoid
- [x] 激活函数：hidden=tanh、gating=sigmoid、output=linear
- [x] `reset_state()`
- [x] 前向测试（无 NaN/Inf、确定性、循环跨 tick 漂移）

### P0.5 Agent 本体（输出分区版）
- [x] `genesis_v2/agent/agent.py`：`Agent` dataclass（输出分区 256:128:128:16）
- [x] `new_agent(id, genome, initial_energy)` 构造器
- [x] `split_output()` 辅助函数
- [x] 基础测试

### P0.6 能量结算 & 死亡判定
- [x] `genesis_v2/engine/metabolism.py`：`tick_cost` / `tick_reward` / `apply_metabolism`（含 v2 新项）
- [x] `genesis_v2/engine/reaper.py`：`evaluate_death` / `sweep_island`（含自我修改致死）
- [x] 测试

### P0.7 存储层
- [x] `genesis_v2/storage/duckdb_store.py`：DuckDB（generations + ticks 表）
- [x] 端到端写读测试

### P0.8 精英种子库 + Mock 环境 + Tick 引擎
- [x] `genesis_v2/evolution/survivor_bank.py`：save/load/list top survivors
- [x] `genesis_v2/env/mock.py`：MockMathEnvironment（多规则 CA: Rule110/30/90）
- [x] `genesis_v2/engine/tick.py`：TickEngine + island_step_sync

### P0.9 验收
- [x] `pytest -q` 全绿（81 passed）
- [x] `python -m genesis_v2 smoke` — 单 Agent smoke
- [x] `python -m genesis_v2 mock-loop --agents 10 --ticks 100` — Mock 闭环（10/10 存活）

---

## Phase 1：社交 + 多 LLM 环境（1-2 周）

- [x] `genesis_v2/social/comm_bus.py`：CommunicationBus（网格拓扑）（Phase 0 已实现，Phase 1 接入）
- [x] `genesis_v2/social/reputation.py`：声誉系统（Phase 0 已实现，Phase 1 接入）
- [x] `genesis_v2/env/base.py`：Environment Protocol（Phase 0 已实现）
- [x] `genesis_v2/env/multi_llm.py`：MultiLLMEnvironment（OpenAI 兼容 + BudgetManager 集成）
- [x] `genesis_v2/env/budget.py`：BudgetManager + 降级策略（已实现 + 接入 MultiLLM）
- [x] `genesis_v2/env/batch.py`：BatchedEnvironment（Phase 0 已实现）
- [x] `genesis_v2/env/embed.py`：FrozenEmbeddingAtlas（Phase 0 已实现）
- [x] `genesis_v2/engine/tick.py`：新 tick 循环（comm_bus 接入 TickEngine + multi_island_step）
- [x] `genesis_v2/population/island.py`：Island 类 + create_islands 工厂
- [x] 社交选择压力（cooperation detection + reputation + social reward，接入 tick 循环）
- [x] 多岛架构（4 岛并行，per-island env/comm_bus/budget）
- [x] `scripts/dashboard.py`：Streamlit 配置面板 v1（Phase 0 已实现）
- [x] 验收：Agent 产生非随机消息（mean_msg_norm ~35）+ 多岛运行 400/400 存活

---

## Phase 2：演化升级 + 探索机制（1-2 周）

- [x] `genesis_v2/genome/crossover.py`：NEAT 式交叉重组
- [x] `genesis_v2/evolution/breeder.py`：繁殖策略（Crossover 50% + Clone 30% + Explore 20%）
- [x] `genesis_v2/evolution/gen_memory.py`：GenerationalMemory（behavioral_signature + successful_patterns + 跨代传承）
- [x] `genesis_v2/evolution/survivor_bank.py`：精英种子库（已实现基础版）
- [x] `genesis_v2/population/island.py`：Island + tick_all（已实现，Phase 1）
- [x] `genesis_v2/population/migration.py`：跨 LLM 迁移 + 适应性奖励（ring topology, 20-tick linear decay）
- [x] `genesis_v2/metrics/exploration.py`：探索奖励（surprise_to_LLM − surprise_to_self）
- [x] 自我修改通道（`engine/selfmod.py`：16-dim selfmod → mutation, 70% death rate, energy gate）
- [x] 验收：跨代 fitness 上升（2254→3636→6308）+ pytest 193 passed

---

## Phase 3：翻译层 + 对话（1-2 周）

- [x] `genesis_v2/translation/translator.py`：多 LLM 翻译器（Mock + API 双模式，vec↔text 双向映射）
- [x] 对话协议（`ConversationSession` — 人类 ↔ Agent 双向多轮对话，历史上下文传递）
- [x] 对话质量探针（`metrics/probes/conversation.py` — response_diversity, semantic_similarity, multi_turn_coherence, cross_llm_consistency）
- [x] CLI `converse` 命令（`python -m genesis_v2 converse --probe` — 加载精英 Agent 进行对话 + 探针评估）
- [x] 验收：`pytest -q` 全绿（227 passed），`converse` CLI 端到端运行成功，探针输出有效数值

---

## Phase 4：深度实验 + 报告（2 周）

- [x] 全部 7 项探针（ood, modularity, multiscale, multi_llm, communication, self_mod, exploration_effect）
- [x] ProbeReport 统一报告 + `run_all_probes()` 编排器 + `save_probe_report()` 持久化
- [x] 探针集成到实验循环（`run_experiment.py` 每代运行探针，probe_interval 可配）
- [x] 对照组消融 CLI（`python -m genesis_v2 mve-run --preset A --compare` 支持 8 个预设 A-H）
- [x] Tier 评估（C/B/A/S/SS 分级，基于 7 项探针通过数）
- [x] 验收：`pytest -q` 全绿（250 passed），`mve-run` 端到端运行成功，探针输出有意义数值

---

## 精英种子库（Survivor Bank）

**目录**：`genesis_v2/data/survivors/`

优秀且合格的 agents 会被保存为 JSON 文件到此目录。文件命名格式：
`{agent_id}_gen{generation}_fit{fitness:.2f}.json`

**用途**：
- 作为未来变异的蓝图（clone + mutate）
- 跨代传承的种子（breeder 从此目录加载 top agents）
- 实验复现的检查点

**API**：
- `save_agent(agent)` — 保存单个 agent
- `load_agent(path)` — 加载单个 agent
- `load_top_survivors(n)` — 加载 top N 精英
- `list_survivors()` — 列出所有种子（按 fitness 降序）

---

## 会话日志（Session Log）

> 时间倒序，最新在最上。

### Session 7 — 2026-05-09：Phase 4 完成 — 深度实验 + 探针 + 消融

- **7 项认知探针**（全部实现于 `metrics/probes/` 目录）：
  1. `ood.py` — OOD 泛化：换 CA 规则，测 KL 比率（<2=泛化，>10=记忆）
  2. `modularity.py` — 拓扑模块度：简化 Louvain 社区检测（Q>0.4=结构化）
  3. `multiscale.py` — 多尺度预测一致性：{1,4,16} horizon KL 比值（<3=世界模型）
  4. `multi_llm.py` — 多 LLM 适应性：跨环境 KL 比率 + 适应速度
  5. `communication.py` — 沟通涌现：互信息 I(msg; action) + 消息-行为相关性
  6. `self_mod.py` — 自我修改效率：存活率 + fitness 变化
  7. `exploration_effect.py` — 探索效果：探索比例 + fitness 相关性
- **ProbeReport 统一报告**：`runner.py` — `run_all_probes()` 编排全部探针，`save_probe_report()` 持久化 JSON
- **Tier 评估**：基于 7 项探针通过数的 C/B/A/S/SS 分级
- **消融 CLI**：`python -m genesis_v2 mve-run --preset A` 运行单个预设，`--compare` 运行全部 8 个预设
- **实验循环集成**：`run_experiment.py` 新增 `run_probes` + `probe_interval` 参数，每代自动运行探针
- **新测试**：`test_probes.py` — 23 个用例覆盖全部探针 + runner
- **验收**：`pytest -q` 全绿（250 passed），`mve-run --preset A` 2 代运行成功，3/7 探针通过（B-tier）

### Session 6 — 2026-05-09：Phase 3 完成 — 翻译层 + 对话

- **Translator**：`genesis_v2/translation/translator.py` — 双向翻译器（Mock + API 双模式）
  - `vec_to_text()`：Agent action 向量 → 伪文本（Mock 模式用 hash 确定性映射，API 模式调 LLM 生成）
  - `text_to_vec()`：自然语言 → Agent 输入向量（embedding + P^T 投影）
  - `translate_agent_output()` / `translate_to_input()`：完整管线辅助函数
- **ConversationSession**：多轮对话协议 — 人类输入文本 → embed → agent forward → translate to text，维护历史上下文
- **对话质量探针**：`metrics/probes/conversation.py` — 4 项指标
  - `response_diversity`：响应多样性（Shannon 熵）
  - `semantic_similarity`：prompt-response 余弦相似度
  - `multi_turn_coherence`：多轮连贯性（Jaccard 词重叠）
  - `cross_llm_consistency`：跨翻译一致性（同一 prompt 不同翻译 → 输出相似度）
- **CLI converse 命令**：`python -m genesis_v2 converse --probe` — 加载精英种子库 Agent，交互式对话 + 探针评估
- **新测试**：2 个新测试文件（test_translator, test_conversation_probes），34 个新用例
- **验收**：`pytest -q` 全绿（227 passed），`converse` CLI 端到端运行成功，探针输出有效数值

### Session 5 — 2026-05-09：Phase 2 完成 — 演化升级 + 探索机制

- **NEAT Crossover**：`genesis_v2/genome/crossover.py` — 按节点 ID 对齐两个父代基因，匹配基因按适应度比例随机选父代，disjoint/excess 从更适应的父代继承，输出分区结构强制一致
- **混合繁殖策略**：`genesis_v2/evolution/breeder.py` — Crossover 50% + Clone 30% + Exploration Clone 20%（更多变异），替代旧的纯 clone+mutate
- **GenerationalMemory**：`genesis_v2/evolution/gen_memory.py` — 记录每代 top agents 的 behavioral_signature、successful_patterns、social_partners，50% 概率传递给子代（偏置 recurrent state）
- **跨 LLM 迁移**：`genesis_v2/population/migration.py` — Ring topology 迁移，每 migration_interval_generations 代 top-3 迁移到邻岛，adaptation bonus 线性衰减 20 ticks
- **探索奖励**：`genesis_v2/metrics/exploration.py` — max(0, surprise_to_LLM − surprise_to_self)，驱动从"学习已知"到"发现未知"的阶段转变
- **自我修改通道**：`genesis_v2/engine/selfmod.py` — 16-dim selfmod 输出 → 7 种基础变异倾向权重 + 参数 + 触发阈值，能量门槛 + 70% 死亡率 + 每次消耗 selfmod_energy_cost
- **tick.py 集成**：island_step_sync 新增 exploration bonus 计算、selfmod 执行、exploration 传入 metabolism
- **run_experiment.py 重构**：删除旧 breed_generation，改用 breed_generation_v2 + GenerationalMemoryBank + MigrationTracker
- **新测试**：6 个新测试文件（crossover, breeder, gen_memory, migration, exploration, selfmod），63 个新用例
- **验收**：`pytest -q` 全绿（193 passed），`experiment --multi-island` 4 岛 3 代运行成功，跨代 best_fit 上升（2254→3636→6308）

### Session 4 — 2026-05-09：Phase 1 完成 — 社交层 + 多岛架构

- **Island Manager**：`genesis_v2/population/island.py` — `Island` 类 + `create_islands()` 工厂，每个岛绑定独立 env + comm_bus + budget
- **TickEngine 社交接入**：`TickEngine` 新增 `comm_bus` 参数，`multi_island_step()` 函数支持多岛并行
- **实验运行器重构**：`run_experiment.py` 支持多岛模式（`_run_multi_island`）+ 单岛模式（`_run_single_island`），per-island 繁殖 + 隔离
- **BudgetManager 集成**：`MultiLLMEnvironment.set_budget()` — API 调用前检查预算，超限返回零向量，成功调用后记录成本
- **DuckDB 社交指标**：`ticks` 表新增 `messages_received` + `mean_trust` 列（含迁移逻辑）
- **CLI 社交开关**：`--social` / `--no-social` 和 `--multi-island` / `--single-island` 标志
- **新测试**：5 个新测试文件（test_comm_bus, test_reputation, test_island, test_multi_llm, test_social_integration）
- **验收**：`pytest -q` 全绿（130 passed），`mock-loop --social` 消息范数 ~35，`experiment --multi-island` 4 岛 400/400 存活

### Session 3 — 2026-05-08：实验运行器 + Dashboard + 自动演化

- **实验运行器**：`genesis_v2/scripts/run_experiment.py` — 完整的 generation loop（tick → select → mutate → save elites → breed next gen）
- **自动演化**：每代自动选择 top-k 精英 → clone+mutate 产生后代 → 下一代
- **自动种子库**：每代自动保存 top 10% 到 `data/survivors/`，新实验自动加载为种子
- **Streamlit Dashboard**：`genesis_v2/scripts/dashboard.py` — 3 个 Tab
  - Settings: API keys（password 输入）+ 物理常数 + 基因参数 + 演化参数
  - Launch: 种群大小/代数/ticks 滑块 + 启动/停止按钮 + 预设选择
  - Monitor: 实时 KPI 卡片 + fitness/energy/pred_err/alive 折线图 + Top agents 排行榜
- **CLI 新命令**：
  - `python -m genesis_v2 experiment --agents 10 --generations 5 --ticks 100`
  - `python -m genesis_v2 dashboard`（启动 Streamlit）
- **survivor_bank 扩展**：新增 `auto_save_elites()` + `auto_load_seeds()`
- **Status JSON**：`data/experiment_status.json` — Dashboard 实时读取
- **DuckDB 遥测**：每代记录到 `data/experiments/*.duckdb`
- **验证**：`pytest -q` 全绿（81 passed），experiment 命令运行成功，survivors 自动保存

### Session 2 — 2026-05-08：Phase 0 完整实现

- **安装**：uv 0.11.11 + Python 3.12.13，依赖全部安装成功
- **P0.1**：CLI smoke + mock-loop 子命令实现并验证
- **P0.2**：GenomeGraph v2 向量节点版（D_node=64，矩阵权重），含完整测试
- **P0.3**：12 种变异原语（7 v1 升级 + 5 v2 新增），含测试
- **P0.4**：BLAS bundle 前向计算 + forward_scalar 黄金对照，含测试
- **P0.5**：Agent dataclass（输出分区 256:128:128:16）+ split_output，含测试
- **P0.6**：能量结算（含 v2 新项 epsilon·API_cost + zeta·messages）+ 死亡判定（含 selfmod_fatal），含测试
- **P0.7**：DuckDB 存储层（generations + ticks 表），含测试
- **P0.8**：TickEngine + MockMathEnvironment（多规则 CA）+ survivor_bank（精英种子库），含测试
- **P0.9 验收**：`pytest -q` 全绿（81 passed），`python -m genesis_v2 smoke` OK，`mock-loop --agents 10 --ticks 100` 10/10 存活
- **精英种子库**：`genesis_v2/data/survivors/` 目录已创建，API 在 `evolution/survivor_bank.py`

### Session 1 — 2026-05-08：项目初始化

- **项目创建**：AGI7/ 目录，genesis_v2/ 项目骨架
- **文档**：`docx/AGI_v2_roadmap.md`（完整工程蓝图）
- **配置**：`configs/genesis_v2.yaml` + `configs/backends.yaml`
- **pyproject.toml**：依赖声明
- **目录树**：全部模块 `__init__.py` 创建完成
