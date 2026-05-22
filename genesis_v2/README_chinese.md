# Genesis v2

**多 LLM 语义荒野 — 自主演化 AGI 实验平台**

数字生命体在由多个大模型驱动的多元宇宙中演化。通过能量约束、社交压力、自我修改、跨代传承，Agent 自主发展出认知结构——没有训练、没有硬编码、没有先验。

---

## 快速开始

```bash
cd genesis_v2
uv sync --extra dev                  # 安装依赖
uv run pytest -q                     # 验证（250 个测试，约 4 秒）
uv run python -m genesis_v2 smoke    # 单 Agent 快速检查
```

## 运行实验

```bash
# 控制面板（推荐）— 配置、启动、监控一体化
uv run python -m genesis_v2 dashboard

# CLI — Mock 环境（零成本，纯数学模拟）
uv run python -m genesis_v2 mock-loop --agents 10 --ticks 100

# CLI — 完整多岛实验
uv run python -m genesis_v2 experiment --agents 10 --generations 10 --ticks 100

# 与最强演化 Agent 对话
uv run python -m genesis_v2 converse --probe

# 消融实验（8 组对照预设 A-H）
uv run python -m genesis_v2 mve-run --preset A
```

---

## 控制面板

Streamlit 可视化控制台——配置、启动、监控实验的主要入口。

```bash
uv run python -m genesis_v2 dashboard    # 打开 http://localhost:8501
```

| 标签页 | 功能 |
|---|---|
| **Config / 配置** | 每岛独立选择 LLM 后端，API Key 一键测试连通性，硬件预设，物理/基因/演化参数微调 |
| **Run / 启动** | 启动预设（Quick / Standard / Deep / Marathon），精英种子库选择（自动或手动），启停实验 |
| **Monitor / 监控** | 实时 KPI 卡片、tick 级进度条、fitness/energy 折线图、精英排行榜——每 5 秒自动刷新，无需刷页面 |

**核心特性：**

- **每岛独立后端** — 4 座岛屿各自选择 LLM（DeepSeek / Gemini / MiMo / OpenAI / Mock），不同岛屿可以用不同模型
- **实时监控** — 基于 `st.fragment` 的局部刷新，不闪屏、不丢滚动位置，数据每 5 秒静默更新
- **可靠的停止按钮** — PID 持久化到状态文件，刷新页面后仍然能停止实验
- **僵尸检测** — Dashboard 启动时自动检测崩溃的实验进程，标记为已结束
- **预算安全** — $50 硬上限，预算耗尽自动降级到 Mock 环境
- **中英双语** — 所有界面标签和帮助文本均为中英对照

---

## LLM 配置

API Key 通过环境变量或控制面板设置：

```bash
export GENESIS_DEEPSEEK_KEY=sk-xxx
export GENESIS_GEMINI_KEY=AIzaSy-xxx
export GENESIS_MIMO_KEY=xxx
export GENESIS_OPENAI_KEY=sk-xxx
```

后端定义在 `configs/backends.yaml`——新增模型只需加一条配置，零代码改动。

---

## 架构

```
环境层（多 LLM：DeepSeek / Gemini / MiMo / OpenAI / Mock-CA）
    |  observe / interact / true_distribution
    v
Agent 种群（4 座岛屿，每岛绑定独立 LLM 后端）
    ├── GenomeGraph（向量节点，D=64，矩阵权重）
    │   ├── 12 种变异原语（NEAT 分裂、注意力组、模块注入……）
    │   ├── 前向计算（BLAS 矩阵乘法，Kahn 拓扑排序）
    │   └── Crossover（NEAT 式有性繁殖）
    ├── 输出分区（528 维）
    │   ├── [0:256]   动作区  → 通过投影矩阵 P 映射到环境
    │   ├── [256:384]  消息区  → 通过通信总线传递给邻居
    │   ├── [384:512]  状态区  → 工作记忆（反馈回路）
    │   └── [512:528]  自我修改区 → 变异指令
    ├── 能量系统（成本：token/节点/边/API/消息）
    └── 选择压力（饿死 / 熵溢出 / 自我修改致死）
```

---

## 核心概念

| 概念 | 作用 |
|---|---|
| **多 LLM 岛屿** | 4 座岛屿，各绑定不同大模型。Agent 定期跨岛迁移，迫使发展通用理解力而非死记硬背。 |
| **能量 = 生命** | 每 tick 消耗能量（token、节点、边）。预测准的 Agent 赚取能量。能量归零即死亡。 |
| **社交层** | Agent 分布在二维网格上，向邻居发送 128 维向量消息。通信成本激励高效信号。 |
| **自我修改** | 输出向量最后 16 维 = 变异指令。70% 死亡率。需高能量才能执行。驱动自我模型涌现。 |
| **探索奖励** | 奖励 LLM 没预测到、但 Agent 自己能预测的行为。阶段转变：学生 → 探险家。 |
| **跨代记忆** | 精英 Agent 的行为签名 50% 概率传递给后代，类似"家训"。 |
| **认知探针** | 7 项探针区分"理解"与"过拟合"：OOD 泛化、模块度、多尺度、多 LLM 适应性、沟通涌现、自我修改效率、探索效果。 |

---

## 项目结构

```
genesis_v2/
├── genesis_v2/
│   ├── cli.py                 # CLI 入口
│   ├── config.py              # Pydantic 配置模型
│   ├── agent/agent.py         # Agent 实体（输出分区版）
│   ├── genome/
│   │   ├── graph.py           # GenomeGraph（向量节点，D=64）
│   │   ├── mutate.py          # 12 种变异原语
│   │   ├── forward.py         # BLAS 前向计算
│   │   └── crossover.py       # NEAT 交叉重组
│   ├── engine/
│   │   ├── tick.py            # Tick 循环（单岛 + 多岛）
│   │   ├── metabolism.py      # 能量结算
│   │   ├── reaper.py          # 死亡判定
│   │   └── selfmod.py         # 自我修改通道
│   ├── env/
│   │   ├── mock.py            # Mock CA 环境（零成本）
│   │   ├── multi_llm.py       # 多 LLM 环境
│   │   ├── budget.py          # 预算管理 + 自动降级
│   │   └── embed.py           # 冻结 Embedding 投影
│   ├── social/
│   │   ├── comm_bus.py        # 通信总线（网格拓扑）
│   │   └── reputation.py      # 声誉系统
│   ├── population/
│   │   ├── island.py          # 岛屿类
│   │   └── migration.py       # 跨 LLM 迁移
│   ├── evolution/
│   │   ├── breeder.py         # 繁殖策略（50% 交叉 / 30% 克隆 / 20% 探索）
│   │   ├── gen_memory.py      # 跨代记忆库
│   │   └── survivor_bank.py   # 精英种子持久化（JSON）
│   ├── metrics/probes/        # 7 项认知探针 + 编排器
│   ├── translation/translator.py  # 向量 <-> 自然语言双向翻译
│   ├── storage/duckdb_store.py    # DuckDB 遥测存储
│   └── scripts/
│       ├── dashboard.py       # Streamlit 控制面板
│       └── run_experiment.py  # 实验运行器
├── configs/
│   ├── genesis_v2.yaml        # 物理常数 + 实验参数
│   └── backends.yaml          # LLM 后端定义
├── data/
│   ├── survivors/             # 精英 Agent JSON 文件
│   └── experiments/           # DuckDB 实验数据库
└── tests/                     # 250 个测试，26 个文件
```

---

## 配置

物理常数和实验参数在 `configs/genesis_v2.yaml`：

```yaml
physics:
  alpha: 0.01        # token 成本权重
  beta: 0.005        # 节点数成本权重
  w_pred: 1.0        # 预测奖励权重
  w_explore: 0.2     # 探索奖励权重
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
```

LLM 后端在 `configs/backends.yaml`，新增模型零代码改动。

---

## 测试

```bash
uv run pytest -q                                 # 250 passed，约 4 秒
uv run pytest tests/test_genome_graph.py -v      # 单模块测试
uv run pytest tests/test_probes.py -v            # 探针测试
```

---

## 当前进度

| 阶段 | 状态 | 关键结果 |
|---|---|---|
| Phase 0：骨架 | 完成 | 81 个测试，10 Agent x 100 tick 运行不崩 |
| Phase 1：社交 + 多 LLM | 完成 | 130 个测试，消息范数 ~35，4 岛 400/400 存活 |
| Phase 2：演化 + 探索 | 完成 | 193 个测试，跨代 fitness 上升（2254 -> 6308） |
| Phase 3：翻译 + 对话 | 完成 | 227 个测试，人类-Agent 对话可用 |
| Phase 4：深度实验 + 探针 | 完成 | 250 个测试，7 项探针，B-tier（3/7 通过） |

**当前评级：B-tier** — 演化信号已确认。冲击 A-tier 需再通过 1 项探针。完整设计规格见 `docx/AGI_v2_roadmap.md`。

---

## 技术栈

Python 3.12 / NumPy / SciPy / Pydantic / NetworkX / DuckDB / httpx / Streamlit / pytest + hypothesis

## 许可证

研究项目，详见仓库条款。
