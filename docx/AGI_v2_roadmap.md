# Project Genesis v2.0

## 多LLM语义荒野 — 自主演化AGI实验平台（Multi-LLM Semantic Wilderness）

> **Project Genesis v2.0** 工程蓝图：基于 v1 的工程验证，面向"从学习人类知识到自主超越"的完整路径——六层架构、多LLM岛屿、社交涌现、自我修改、跨代传承。

---

# 零、项目定位

## 0.1 本质

> **Project Genesis v2 是一个从零开始、面向真正 AGI 的演化实验平台。**
>
> 在由多个 LLM 充当"多元物理法则"的信息宇宙里，通过结构演化、能量守恒、社交压力、自我修改，演化出**对该宇宙原生适应的、能自主思考的、具备自我优化能力的数字生命体**。

## 0.2 与 v1 的核心区别

| 维度 | v1 | v2 |
|---|---|---|
| 节点计算 | 标量（1 float/node） | 向量（64-dim/node） |
| 边权重 | 标量 float | 矩阵 W ∈ R^{D×D} |
| Agent 交互 | 完全独立，共享 pop_mean | 局部通信（网格邻居消息传递） |
| 环境 | 单一 LLM 或 Mock | 多 LLM 岛屿（每个岛绑定不同大模型） |
| 演化 | 无性繁殖（clone + mutate） | + Crossover（NEAT 式性繁殖） |
| 选择压力 | 预测 + 压缩 + 行为多样性 | + 社交奖励 + 探索奖励 + 迁移适应性 |
| 自我修改 | 无 | 16 维输出通道 → 变异指令 |
| 跨代记忆 | 无（recurrent state 清零） | GenerationalMemory（经验摘要传承） |
| 最终目标 | 指标曲线 | 与 Agent 自然语言对话 |

## 0.3 核心哲学：从"学习人类"到"超越人类"

**四阶段涌现路径**（能量梯度自然驱动，非硬编码）：

```
Phase 1 学习期：  预测奖励主导 → Agent 学习 LLM 中的人类知识
   ↓ （预测奖励边际收益递减）
Phase 2 内化期：  压缩奖励主导 → Agent 压缩知识为结构化表征
   ↓ （压缩趋于极限）
Phase 3 超越期：  探索奖励主导 → Agent 探索 LLM 知识之外的领域
   ↓ （Agent 发展出自我模型）
Phase 4 自主期：  自我修改涌现 → Agent 能优化自身拓扑
```

## 0.4 范式红线（继承 v1 并扩展）

**系统内部永远禁止**：
- SFT / RLHF / 监督学习
- Prompt Engineering / Chain-of-Thought 注入
- Agent Workflow / 工具调用 / LangChain 类组装
- 手写任何"认知模块"（Memory / Planner / Attention / S1-S2）
- 任何 `if token == "hello"` 类的业务逻辑硬编码

**系统唯一允许**：
- 定义**物理常数**（能量换算率、突变概率、投影矩阵）
- 定义**物理接口**（输出分区：动作/消息/状态/自我修改）
- 定义**纯数学奖惩**（KL 散度、MDL、互信息、探索奖励公式）
- 定义**物理规则的稳定化**（LLM temperature = 0）

---

# 一、世界观（Worldview）

## 1.1 宇宙基本设定

```text
┌─────────────────────────────────────────────────────────┐
│              Multi-LLM Environment                       │
│         (多元物理法则引擎 — 冷酷、决定论)                  │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │ DeepSeek │  │  Gemini  │  │   MiMo   │  │  Mock-CA ││
│  │ (Island-A)│  │(Island-B)│  │(Island-C)│  │(Island-D)││
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘│
└──────────────────────▲───────────────────────────────────┘
                       │
               action / feedback (token 语义流)
                       │
              [ Interaction Bus + Communication Bus ]
                       │
┌──────────────────────┴───────────────────────────────────┐
│              Agent Population                             │
│         (数字原生生物 — 向量节点 GenomeGraph)              │
│  - 拓扑 + 能量 + 社交 + 工作记忆 + 自我修改能力          │
│  - 无预设任务 / 模块 / 目标                               │
│  - 必须在多 LLM 环境中生存 → 驱动通用理解涌现            │
└──────────────────────────────────────────────────────────┘
```

## 1.2 Agent 的感知方式

- Agent 的世界由**定长向量 `[]float32`** 构成
- Agent 向外抛 action 向量 → 通过冻结投影矩阵 P 映射到 token → LLM 按因果律回应 token → 通过 P^T 映射回 Agent 输入向量
- Agent 之间的通信**直接在向量空间传递**（不经过 LLM，更高效）
- 这是宇宙中两种基本"物理相互作用"：与环境的语义交互 + 与同类的向量通信

## 1.3 多 LLM 岛屿：最强认知压力

**核心洞察**：如果 Agent 只面对一个 LLM，它可以靠记忆行为模式生存。但跨多个 LLM 迁移时，Agent **必须**学会：
- 识别不同 LLM 共享的**深层结构**（语言规律、逻辑一致性）
- 发展**可迁移的推理能力**（通用因果推理，而非 pattern matching）
- 建立**环境自适应的内部模型**（区分"物理法则"和"表面特征"）

这就像一个孩子在多个文化中长大——被迫学会"通用社交规则"而非死记某个文化的行为模式。

---

# 二、六大物理基石

## 2.1 空间法则（向量 I/O + 输出分区）

### 2.1.1 内部维度

- **节点维度**：`D_node = 64`（每个隐藏节点计算一个 64 维向量）
- **输入维度**：`D_in = D_node × N_input_nodes`（由输入节点数决定）
- **输出分区**：`D_out = D_action + D_message + D_state + D_selfmod = 256 + 128 + 128 + 16 = 528`

### 2.1.2 输出分区（物理接口定义）

```text
┌─────────────────────────────────────────────────────────────┐
│  Agent Output Vector (528-dim)                               │
│                                                              │
│  [0:256]    动作区  → 通过投影矩阵 P 映射到 LLM token 空间   │
│  [256:384]  消息区  → 通过通信总线传递给网格邻居 Agent        │
│  [384:512]  状态区  → 回读为下一 tick 的额外输入（工作记忆）  │
│  [512:528]  自我修改区 → 映射到图谱变异指令                   │
└─────────────────────────────────────────────────────────────┘
```

**合法性论证**：输出分区是**宇宙的物理接口**（如同 USB 针脚定义），不是 Agent 的业务逻辑。Agent 内部的 GenomeGraph 仍然零先验、零预训练。

### 2.1.3 嵌入与投影

```text
┌─── 宇宙的物理常数（全程冻结） ──────────────────────────────────┐
│  E  ∈ R^{|V| × D_embed}   LLM 原生 Embedding 矩阵              │
│  P  ∈ R^{D_action × D_embed}  项目启动时固定随机投影矩阵（seed 固定）│
└────────────────────────────────────────────────────────────────┘

Agent → LLM:
    action_vec (256) ──P──▶ act_embed (D_embed)
                              │
                     余弦相似度最近邻
                              ▼
                        token ∈ V ──▶ API 调用 LLM

LLM → Agent:
    token ∈ V ──E[token]──▶ fb_embed (D_embed) ──P^T──▶ feedback_vec (D_in)
```

**物理连贯性**：Agent 输出向量的微小扰动 → 映射到 embedding 空间相近位置 → LLM 端产生语义相近的激发。

## 2.2 时间法则（权威 Tick Engine）

- **权威服务器架构**：`asyncio` 权威 Ticker + `ThreadPoolExecutor`，全局统一时钟
- **Tick 频率**：
  - Mock 环境：**100 Hz**（纯数学函数，零 API 成本）
  - Cloud LLM：**0.5-2 Hz**（受 API 延迟限制，asyncio 批量化优化）
- **单帧操作**：

```text
perceive → forward → deliver_messages → forward_2(可选)
→ interact(→ LLM API) → self_modify → metabolize → reap → social_settle
```

- **无例外**：每 tick 必须闭环，不允许跨 tick 的业务缓存

## 2.3 能量与热力学法则

### 2.3.1 代谢消耗（Cost per Tick）

```math
Cost = α · T_usage + β · |V_nodes| · log|V_nodes| + γ · L_latency + δ · |E_edges|
     + ε · API_cost + ζ · |messages_sent|
```

| 符号 | 含义 | 初始值 | 作用 |
|---|---|---|---|
| T_usage | 本 tick 消耗的 token 数 | - | 惩罚冗长输出 |
| V_nodes | 拓扑节点数 | - | 惩罚网络臃肿 |
| L_latency | 本 tick 计算耗时（ms） | - | 惩罚低效拓扑 |
| E_edges | 边数 | - | 惩罚连接泛滥 |
| API_cost | 本 tick 的 API 调用美元成本 | - | **v2 新增**：惩罚低效 API 使用 |
| messages_sent | 发送的消息向量数 | - | **v2 新增**：惩罚通信 spam |
| α, β, γ, δ, ε, ζ | 物理常数 | 配置文件 | 唯一的"超参数" |

### 2.3.2 能量获取（Reward per Tick）

```math
Reward = w_p · (−ΔKL) + w_c · ΔCompression + w_b · BehavioralVariance
       + w_a · MigrationAdaptation + w_s · SocialCooperation
       + w_e · ExplorationBonus
```

| 项 | 定义 | 复杂度 | 哲学定位 |
|---|---|---|---|
| −ΔKL | 预测误差下降量（KL 散度） | O(V) | FEP 核心（继承 v1） |
| ΔCompression | 描述长度压缩增量（MDL） | O(V+E) | Kolmogorov 复杂度代理（继承 v1） |
| BehavioralVariance | ‖action − μ_pop‖₂ | O(D) | 反坍缩项（继承 v1） |
| **MigrationAdaptation** | 迁移到新 LLM 后 KL 下降速度 | O(1) | **v2 新增**：奖励通用理解力 |
| **SocialCooperation** | 合作成功率 × 互惠评分 | O(N) | **v2 新增**：驱动社交行为涌现 |
| **ExplorationBonus** | max(0, surprise_LLM − surprise_self) | O(V) | **v2 新增**：驱动超越人类知识 |

### 2.3.3 探索奖励详解

```math
ExplorationBonus = w_e · max(0, surprise_to_LLM − surprise_to_self)
```

- `surprise_to_LLM` = LLM 对 Agent 行为的惊讶程度（预测概率的负对数）
- `surprise_to_self` = Agent 对自己行为的预测误差

**含义**：当 Agent 做出 LLM 认为不可能但 Agent 自己完全能预测的行为时，获得探索奖励。这是从"学习已知"到"发现未知"的自然过渡。

**阶段转变机制**：
- 初期：预测奖励边际收益高，探索奖励低 → Agent 专注学习人类知识
- 中期：预测奖励趋于饱和，探索奖励相对上升 → Agent 开始探索
- 后期：Agent 发展出自洽的内部模型，探索成为主要收益来源

### 2.3.4 最终能量微分

```math
ΔEnergy_t = Reward_t − Cost_t
```

## 2.4 死亡与达尔文镰刀（The Reaper）

**死亡触发条件**（任一即死）：

| 条件 | 含义 |
|---|---|
| Energy ≤ 0 | 饿死 |
| API 调用连续失败 > N | 环境不可达（宇宙"毁灭"） |
| ContextOverflow | 神经崩溃 |
| TopologyEntropy ≥ Θ | 癌变（拓扑熵超标） |
| Self-Modify 致命 | 自我修改导致内部状态崩溃 |

死亡 = `GenomeGraph` 实例销毁 + 从 `AgentPool` 移除 + **GenerationalMemory 保留**。无复活机制。

## 2.5 自我修改机制（通向 AGI 的关键通道）

### 2.5.1 自我修改通道

Agent 的输出向量的最后 16 维 `[512:528]` 被解释为自我修改指令：

```text
[0:7]   → 7 种基础变异的倾向权重（哪个变异更可能发生）
[7:12]  → 变异强度/方向参数
[12:16] → 触发阈值（超过阈值时才执行修改）
```

### 2.5.2 安全性机制

| 机制 | 规则 | 原因 |
|---|---|---|
| 能量门槛 | 能量 > 初始能量 × 2 才能执行 | 只有"富余"的 Agent 才应冒险 |
| 能量消耗 | 每次自我修改消耗大量能量 | 昂贵 = 谨慎使用 |
| 高死亡率 | 自我修改后 ~70% 概率恶化 | 模拟脑部手术的风险 |
| 不可遗传 | 子代的 selfmod 能力归零 | 每代需自己发展 |

### 2.5.3 为什么这能通向 AGI

一个能理解"我哪里做得不好 → 我应该怎么改"的 Agent，本质上在进行**科学推理和自我优化**：
- 它需要建立**自我模型**（理解自己的拓扑结构和行为模式）
- 它需要**因果推理**（预测修改的后果）
- 它需要**风险评估**（权衡收益和死亡风险）

这比盲目随机变异快几个数量级——是从"自然选择"到"智能设计"的跳变。

## 2.6 "预测 vs 理解" 判别探针（继承 v1 并扩展）

### 2.6.1 v1 继承的三项探针

| 探针 | 定义 | 真理解 | 纯拟合 |
|---|---|---|---|
| OOD 泛化 | LLM system_prompt 换成语域，测 KL 变化 | 增幅 < 2× | 增幅 > 10× |
| 拓扑模块度 Q | Louvain 社区检测 | Q > 0.4 | Q < 0.2 |
| 多尺度预测一致性 | {1, 4, 16} token 时间窗 KL 比值 | < 3.0 | > 20.0 |

### 2.6.2 v2 新增的四项探针

| 探针 | 定义 | 真理解 | 纯拟合 |
|---|---|---|---|
| **多 LLM 适应性** | 跨模型 KL 比率 + 迁移速度 | KL 比 < 2.0，< 50 tick 适应 | KL 比 > 10.0，> 500 tick |
| **沟通涌现** | 互信息 I(msg; action) + 组合性 | 互信息 > 0.5 bits | 互信息 ≈ 0 |
| **自我修改效率** | 自我修改存活率 + fitness 变化 | 存活率 > 0.3，fitness ↑ | 存活率 < 0.05 |
| **探索奖励效果** | 探索行为占比 + 后续 fitness | 探索后 fitness ↑ | 探索后 fitness ↓ |

### 2.6.3 反自我麻醉条款（继承 v1）

**如果 Phase 4 跑完，七项探针全部为负**：
- 诚实承认系统是**"神经进化 + 压缩拟合"**，不是**"认知结构涌现"**
- 公布负结果（本身就是可发表的 null result）
- 不把曲线好看包装成"类 AGI 涌现"

---

# 三、Agent 本体（Digital Organisms）

## 3.1 数据结构

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Agent:
    id: str
    generation: int = 0
    island_id: int = 0

    genome: "GenomeGraph" = None          # 唯一内部结构（向量节点版）
    energy: float = 0.0
    tick_alive: int = 0
    is_alive: bool = True

    # === 输出分区缓存（每 tick 更新）===
    last_action: np.ndarray = None        # [256] 对环境的动作
    last_message: np.ndarray = None       # [128] 对其他 Agent 的消息
    last_state: np.ndarray = None         # [128] 内部工作记忆
    state_buffer: np.ndarray = None       # [128] 上一 tick 的工作记忆（回读）
    last_selfmod: np.ndarray = None       # [16] 自我修改指令

    # === 评估缓存 ===
    prediction_error: float = 0.0
    compression: float = 0.0
    behavioral_variance: float = 0.0
    exploration_bonus: float = 0.0
    fitness: float = 0.0

    # === 社交状态 ===
    inbox: list = field(default_factory=list)
    social_memory: dict = field(default_factory=dict)  # {agent_id: trust_score}

    # === 跨代记忆 ===
    birth_mem: dict = field(default_factory=dict)

    # === 自我修改 ===
    selfmod_enabled: bool = False
    selfmod_count: int = 0
    selfmod_survived: int = 0
```

## 3.2 不具备的东西（继承 v1）

- 状态机 / 行为树 / 规则引擎 / 状态变量
- "记忆池" / "经验回放" / "目标列表"
- 任何人类认知学概念的数据结构

## 3.3 只具备

- 一个**可变拓扑图 GenomeGraph**（向量节点版）
- 一个**能量标量 Energy**
- 一个**收件箱 inbox**（来自邻居的向量消息）
- 一个**状态缓冲区 state_buffer**（工作记忆）
- 一个**自我修改通道**（16 维输出 → 变异指令）

---

# 四、基因与计算图（GenomeGraph v2）

## 4.1 节点类型（4 种原语，与 v1 相同）

```python
class NodeType(IntEnum):
    INPUT  = 0   # 输入端点（固定数量）
    OUTPUT = 1   # 输出端点（固定数量，528 个）
    HIDDEN = 2   # 隐藏节点（可增删，每个计算 64 维向量）
    GATING = 3   # 门控节点（向量元素级 sigmoid 门控）
```

## 4.2 边类型（3 种，与 v1 相同）

```python
class EdgeKind(IntEnum):
    FORWARD   = 0   # 前馈
    SHORTCUT  = 1   # 跳层
    RECURRENT = 2   # 循环（跨 tick 记忆）
```

**v2 升级**：边的权重从标量 `float` 变为矩阵 `np.ndarray[D_dst_node, D_src_node]`。

## 4.3 变异原语（7 → 12 种）

```python
class MutationKind(IntEnum):
    # === v1 继承（向量化升级）===
    ADD_FORWARD_EDGE   = 0   # 前馈边（矩阵权重 W）
    ADD_SHORTCUT_EDGE  = 1   # 跳层边
    ADD_RECURRENT_EDGE = 2   # 循环边
    ADD_HIDDEN_NODE    = 3   # NEAT 分裂：向量版
    ADD_GATING_NODE    = 4   # 门控：向量元素级 sigmoid
    PERTURB_WEIGHT     = 5   # 矩阵权重高斯扰动
    DELETE_RANDOM_EDGE = 6   # 删边

    # === v2 新增 ===
    ADD_ATTENTION_GROUP = 7   # 创建 Q/K/V 向量节点 + softmax 注意力
    ADD_MODULE          = 8   # 注入 2-5 节点的子图（预制功能模块）
    SPLIT_NODE_DIM      = 9   # 高维节点拆分为多个低维节点
    MERGE_NODES         = 10  # 多个节点合并为高维节点
    ADD_COMM_EDGE       = 11  # 连接到消息输出通道的边
```

**绝对禁止**出现 `add_memory_module()`、`add_attention_layer()` 这类面向人类认知的变异。`ADD_ATTENTION_GROUP` 是通用的 Q/K/V 计算原语，不是"人类注意力机制"。

## 4.4 前向计算（向量版）

**v1 的标量计算**：
```python
z = sum(w_i * x_i)        # 标量求和
output = tanh(z)
```

**v2 的向量计算**：
```python
z = W @ x_stack             # 矩阵-向量乘法：W ∈ R^{D_node × (D_node × n_inputs)}
output = np.tanh(z)         # 逐元素激活
```

- **拓扑调度**：对 FORWARD ∪ SHORTCUT 子图做确定性 Kahn 拓扑排序（继承 v1）
- **边聚合**：对每个目的节点，入边预编译为 BLAS bundle（权重矩阵堆叠），用 `numpy.dot` 一次完成
- **循环边**：读取 `_last_hidden`（上一 tick 的 64 维向量残留）
- **门控**：向量元素级 sigmoid 乘性调制
- **激活函数**：hidden = tanh、gating = sigmoid、output = linear
- **黄金对照**：保留 `forward_scalar` 参考实现，逐 tick 对齐 BLAS 路径

---

# 五、环境接口（Multi-LLM Environment）

## 5.1 统一接口

```python
class Environment(Protocol):
    def observe(self) -> np.ndarray:
        """当前环境状态 → Agent 输入向量"""
        ...

    def interact(self, action: np.ndarray) -> np.ndarray:
        """Agent 的动作 → 环境的反馈"""
        ...

    def true_distribution(self, history: np.ndarray) -> np.ndarray:
        """给定历史，返回宇宙真实的下一步分布"""
        ...

    def close(self) -> None: ...
```

## 5.2 四种实现（分层启用）

### 5.2.1 MockMathEnvironment（Phase 0，零成本）

- 环境 = 多规则元胞自动机（Rule110 / Rule30 / Rule90 可切换）
- **成本**：零（纯 CPU）
- **速度**：100+ Hz
- **目的**：验证 TickEngine + 变异 + 能量结算闭环

### 5.2.2 MultiLLMEnvironment（Phase 1-2，主力环境）

- 通过 API 调用多个大模型（DeepSeek / Gemini / MiMo 等）
- 每个岛屿绑定一个 LLM 后端
- 统一 OpenAI 兼容格式（`/v1/chat/completions`）
- Agent 的动作通过投影矩阵映射到 token，发送给 LLM，响应 token 映射回向量

```yaml
# configs/backends.yaml — 只需改这里即可切换模型
backends:
  deepseek:
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-chat
    cost_per_1m_tokens: 0.14
  gemini:
    base_url: https://generativelanguage.googleapis.com/v1beta
    api_key_env: GEMINI_API_KEY
    model: gemini-2.0-flash
    cost_per_1m_tokens: 0.075
  mimo:
    base_url: https://api.xiaomi.com/v1
    api_key_env: MIMO_API_KEY
    model: mimo-7b
    cost_per_1m_tokens: 0.05
```

### 5.2.3 BatchedEnvironment（包装层）

- 把同一 LLM 后端的 Agent 请求聚合成 micro-batch
- `batch_size = 16`，`max_wait_ms = 100`
- 不同后端通过 `asyncio.gather` 并行发送

### 5.2.4 BudgetManager（预算安全网）

```python
class BudgetManager:
    total_budget: float = 50.0   # $50 硬上限
    spent: float = 0.0

    def check_budget(self, island_id, estimated_cost) -> bool:
        """超限自动降级到 Mock 环境"""

    def get_cost_pressure(self, agent_id) -> float:
        """API 成本作为代谢成本的一部分"""
```

**降级策略**：预算充足 → Cloud LLM → 预算紧张 → Mock Math（平滑降级，Agent 无需感知）

---

# 六、社交层（Social Layer — v2 全新）

## 6.1 通信总线（CommunicationBus）

```python
class CommunicationBus:
    grid_size: tuple[int, int]           # (rows, cols) 网格大小
    agent_positions: dict[str, tuple[int, int]]  # Agent 在网格上的位置
    comm_radius: int = 2                  # 通信半径（L2 距离）

    def deliver(self, sender_id, message_vec):
        """将消息投递给通信半径内的所有邻居"""

    def get_inbox(self, agent_id) -> list[np.ndarray]:
        """获取收件箱（邻居消息平均池化）"""
```

**网格布局**：每个岛屿的 Agent 排列在二维网格上（如 10×12 = 120 位置），只能与 L2 距离 ≤ 2 的邻居通信。

## 6.2 社交选择压力

| 压力 | 机制 | 预期涌现 |
|---|---|---|
| **资源共享** | 有限资源池，Agent 的 action 影响资源分配。合作获取更多资源 | 合作行为 |
| **信息不对称** | 每个 Agent 只能看到环境的部分观测。通信扩展信息范围 | 信息传递 |
| **声誉系统** | 成功合作过的 Agent 对获得额外能量奖励（互惠） | 信任 / 互惠 |
| **沟通成本** | 发送消息消耗能量（消息向量范数惩罚） | 高效信号 |
| **集体任务** | 某些高价值资源需要 ≥ 2 个 Agent 同时行动才能获取 | 协调行为 |

**为什么社交压力是语言涌现的前提**：
- 没有沟通需求 → 不可能涌现语言能力
- 地球上语言的起源正是社交压力（合作狩猎、信息共享、社会联盟）
- Agent 之间的向量通信 = "原始发声"，通过演化逐渐编码有意义的信息

---

# 七、种群与演化（Population & Evolution）

## 7.1 岛屿配置（4 岛并行，每岛绑定不同 LLM）

| 岛屿 | 种群 | 突变率 | LLM 后端 | 目的 |
|---|---|---|---|---|
| Island-Explorer | 100 | 0.30 | DeepSeek | 高突变探索 |
| Island-Exploiter | 100 | 0.05 | Gemini | 低突变精细利用 |
| Island-Recurrent | 100 | 0.15 | MiMo | 只允许 recurrent 变异 |
| Island-Shortcut | 100 | 0.15 | Mock-CA | 只允许 shortcut 变异 |

**总种群**：400 个体。

## 7.2 Crossover（NEAT 式性繁殖 — v2 新增）

```python
def crossover(parent_a: GenomeGraph, parent_b: GenomeGraph, rng) -> GenomeGraph:
    # 1. 按 NodeID 对齐两个父代的节点（匹配基因）
    # 2. 匹配的节点/边：随机选一个父代的权重（倾向适应度更高的）
    # 3. 不匹配的（disjoint/excess）：继承适应度更高的父代
    # 4. 输出分区结构强制一致
```

## 7.3 繁殖策略

| 策略 | 比例 | 说明 |
|---|---|---|
| **Crossover** | 50% | 随机配对两个精英，交叉重组 + 1 次变异 |
| **Clone** | 30% | 精英直接复制 + 多次变异 |
| **Migration** | 20% | 从邻岛引入精英（跨 LLM 检验） |

## 7.4 多层级选择

```
个体层：  能量 ≤ 0 → 死亡
群体层：  合作成功率高的 Agent 对获得集体能量奖励
岛屿层：  演化停滞的岛屿触发"灾变"（随机淘汰 30%，注入新突变）
跨代层：  精英基因库 → 下一代种子
迁移层：  跨 LLM 适应性 → 额外能量奖励
```

## 7.5 迁移规则（跨 LLM 迁移 — v2 核心创新）

每 `G = 50` 代，每岛 Top-3 个体随机迁移到邻岛（环形拓扑）。

**迁移时的适应性奖励**：
```python
def migration_adaptation_bonus(agent, ticks_since_migration):
    if ticks_since_migration > 20:  # 前 20 tick
        return 0.0
    kl_improvement_rate = agent.prev_kl - agent.prediction_error
    return w_adapt * kl_improvement_rate * (1.0 - ticks_since_migration / 20)
```

Agent 首次进入新 LLM 环境时，KL 下降越快，能量奖励越大——直接奖励"通用理解力"。

## 7.6 跨代记忆（GenerationalMemory — v2 新增）

```python
@dataclass
class GenerationalMemory:
    agent_id: str
    generation: int
    fitness: float
    behavioral_signature: np.ndarray      # 平均行为向量（128-dim）
    successful_patterns: list[np.ndarray] # 能量增长最大的 tick 的行为模式
    social_partners: list[str]            # 成功合作过的 Agent ID
    env_adaptation_scores: dict[str, float]  # 对不同 LLM 的适应分数
```

**继承规则**：
- 50% 概率继承父代的 `behavioral_signature` 作为初始偏置
- `successful_patterns` 可初始化部分 recurrent state
- `env_adaptation_scores` 清零（每代重新证明自己）

---

# 八、Python 代码骨架

## 8.1 项目结构

```text
AGI7/
├── docx/
│   └── AGI_v2_roadmap.md              # 本文档
├── genesis_v2/
│   ├── pyproject.toml                  # 依赖 + 构建配置
│   ├── genesis_v2/
│   │   ├── __init__.py
│   │   ├── cli.py                      # 入口（python -m genesis_v2）
│   │   ├── config.py                   # pydantic 配置模型
│   │   ├── engine/
│   │   │   ├── __init__.py
│   │   │   ├── tick.py                 # TickEngine
│   │   │   ├── tick_core.py            # 核心单帧循环
│   │   │   ├── reaper.py               # 死亡判定
│   │   │   └── metabolism.py           # 能量结算
│   │   ├── genome/
│   │   │   ├── __init__.py
│   │   │   ├── graph.py                # GenomeGraph（向量节点版）
│   │   │   ├── mutate.py               # 12 种变异原语
│   │   │   ├── forward.py              # 前向计算（BLAS bundle 向量版）
│   │   │   └── crossover.py            # NEAT 式交叉重组
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   └── agent.py                # Agent 实体（输出分区版）
│   │   ├── population/
│   │   │   ├── __init__.py
│   │   │   ├── island.py               # 岛屿
│   │   │   └── migration.py            # 迁移（含跨 LLM 适应性奖励）
│   │   ├── social/
│   │   │   ├── __init__.py
│   │   │   ├── comm_bus.py             # 通信总线（网格拓扑）
│   │   │   └── reputation.py           # 声誉系统
│   │   ├── env/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # Environment 协议
│   │   │   ├── mock.py                 # MockMathEnvironment（多规则 CA）
│   │   │   ├── multi_llm.py            # MultiLLMEnvironment
│   │   │   ├── batch.py                # BatchedEnvironment
│   │   │   ├── budget.py               # BudgetManager
│   │   │   └── embed.py                # 冻结 Embedding + 投影 P
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── kl.py                   # KL 散度
│   │   │   ├── compression.py          # MDL 压缩
│   │   │   ├── bvar.py                 # BehavioralVariance
│   │   │   ├── exploration.py          # 探索奖励
│   │   │   └── probes/
│   │   │       ├── __init__.py
│   │   │       ├── ood.py              # OOD 泛化
│   │   │       ├── modularity.py       # Louvain 模块度
│   │   │       ├── multiscale.py       # 多尺度一致性
│   │   │       ├── multi_llm.py        # 多 LLM 适应性探针
│   │   │       ├── communication.py    # 沟通涌现探针
│   │   │       ├── self_mod.py         # 自我修改效率探针
│   │   │       └── conversation.py     # 对话质量探针
│   │   ├── evolution/
│   │   │   ├── __init__.py
│   │   │   ├── breeder.py              # 繁殖策略（含 Crossover）
│   │   │   ├── gen_memory.py           # GenerationalMemory
│   │   │   └── survivor_bank.py        # 精英种子库（序列化/反序列化）
│   │   ├── translation/
│   │   │   ├── __init__.py
│   │   │   └── translator.py           # 多 LLM 翻译器（向量 ↔ 自然语言）
│   │   └── storage/
│   │       ├── __init__.py
│   │       └── duckdb_store.py         # DuckDB 存储
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_genome_graph.py
│   │   ├── test_genome_mutate.py
│   │   ├── test_genome_forward.py
│   │   ├── test_genome_crossover.py
│   │   ├── test_agent.py
│   │   ├── test_metabolism.py
│   │   ├── test_reaper.py
│   │   ├── test_comm_bus.py
│   │   ├── test_social.py
│   │   ├── test_embed.py
│   │   ├── test_multi_llm.py
│   │   ├── test_budget.py
│   │   ├── test_batch_env.py
│   │   ├── test_probes.py
│   │   ├── test_evolution.py
│   │   ├── test_translation.py
│   │   └── test_stress.py
│   ├── configs/
│   │   ├── genesis_v2.yaml             # 物理常数配置
│   │   └── backends.yaml               # LLM 后端配置
│   ├── experiments/
│   │   └── ablation.yaml               # 对照组实验预设
│   ├── scripts/
│   │   ├── dashboard.py                # Streamlit 可视化 + 配置面板
│   │   └── run_mve.py                  # MVE 实验运行器
│   └── data/
│       └── ood_prompts.txt             # OOD 探针用 prompt
```

## 8.2 物理常数配置（`configs/genesis_v2.yaml`）

```yaml
physics:
  alpha: 0.01           # token cost
  beta: 0.005           # node cost
  gamma: 0.001          # latency cost
  delta: 0.002          # edge cost
  epsilon: 0.1          # API cost multiplier (v2)
  zeta: 0.01            # message cost (v2)

  w_pred: 1.0           # prediction reward weight
  w_comp: 0.5           # compression reward weight
  w_bvar: 0.3           # behavioral variance weight
  w_adapt: 0.5          # migration adaptation weight (v2)
  w_social: 0.3         # social cooperation weight (v2)
  w_explore: 0.2        # exploration bonus weight (v2)

  death_penalty: 500.0
  initial_energy: 1000.0
  topology_entropy_threshold: 5.0

  # 自我修改参数（v2）
  selfmod_energy_threshold: 2000.0     # 能量 > 2000 才能自我修改
  selfmod_energy_cost: 500.0           # 每次修改消耗 500 能量
  selfmod_death_rate: 0.7              # 70% 概率恶化

evolution:
  tick_rate: 2            # Hz（Cloud LLM，受 API 延迟限制）
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
    - name: RecurrentOnly
      size: 100
      mutation_rate: 0.15
      backend: mimo
      allowed_mutations: [AddRecurrentEdge, PerturbWeight]
    - name: ShortcutOnly
      size: 100
      mutation_rate: 0.15
      backend: mock
      allowed_mutations: [AddShortcutEdge, PerturbWeight]

genome:
  node_dim: 64                      # 每个隐藏节点的向量维度（v2）
  input_nodes: 8                    # 输入节点数
  output_nodes_action: 4            # 动作区输出节点数（4×64=256）
  output_nodes_message: 2           # 消息区输出节点数（2×64=128）
  output_nodes_state: 2             # 状态区输出节点数（2×64=128）
  output_nodes_selfmod: 1           # 自我修改区输出节点数（1×64→取前16维）
  initial_hidden_nodes: 4
  initial_edge_density: 0.2

environment:
  type: multi_llm                   # mock | multi_llm
  embed_dim: 1536                   # LLM embedding 维度
  projection_seed: 42               # P 矩阵固定种子
  batch_size: 16
  batch_wait_ms: 100
  context_window: 64
  temperature: 0.0
  top_p: 0.0

  # 社交参数（v2）
  comm_radius: 2                    # 通信半径
  grid_rows: 10                     # 岛屿网格行数
  grid_cols: 10                     # 岛屿网格列数

  # 预算参数（v2）
  total_budget_usd: 50.0
  per_island_budget_usd: 15.0
  fallback_to_mock: true            # 预算耗尽时降级到 Mock

evaluation:
  probe_interval_generations: 10
  ood_prompt_file: ./data/ood_prompts.txt
  modularity_algo: louvain
  multiscale_horizons: [1, 4, 16]
```

---

# 九、硬件与成本预算（RTX 4060 Laptop）

| 组件 | VRAM | CPU | 策略 |
|---|---|---|---|
| Agent 前向计算（400 × 64 维向量节点） | 0 | 中 | ThreadPool 并行 + numpy BLAS |
| 消息路由 | ~10MB | 低 | 稀疏矩阵操作 |
| LLM API 调用 | 0 | 低 (IO) | asyncio 并发 + 批量化 |
| Embedding 投影 | ~12MB | 低 | 矩阵乘法 |
| DuckDB 存储 | 0 | 低 | 异步写入 |

**核心设计**：Agent 前向计算纯 CPU numpy，LLM 仅通过 API 调用（IO-bound，高效异步）。不部署本地模型。

**预计成本**（400 Agent × 1000 tick）：

| 阶段 | 环境 | 成本 |
|---|---|---|
| Phase 0 | Mock | **$0** |
| Phase 1-2 | 经济型 API（DeepSeek/MiMo） | **$5-10** |
| Phase 3 | 中端 API（验证最佳 Agent） | **$10-20** |
| Phase 4 | 混合（长程实验） | **$10-20** |
| **总计** | | **≤ $50** |

---

# 十、前端配置面板

## 10.1 Streamlit 面板页面

```text
┌──────────────────────────────────────────────────────┐
│                Genesis v2 控制台                       │
│                                                       │
│  [1. API 配置]  [2. 实验参数]  [3. 岛屿分配]         │
│  [4. 预算控制]  [5. 启动实验]  [6. 实时监控]         │
│                                                       │
│  ──────────── API 配置 ────────────                   │
│  DeepSeek API Key: [••••••••]  [测试连通性 ✓]         │
│  Gemini API Key:   [••••••••]  [测试连通性 ✓]         │
│  MiMo API Key:     [••••••••]  [测试连通性 ✓]         │
│                                                       │
│  ──────────── 岛屿分配 ────────────                   │
│  Island-Explorer  → DeepSeek    [✓]                   │
│  Island-Exploiter → Gemini      [✓]                   │
│  Island-Recurrent → MiMo        [✓]                   │
│  Island-Shortcut  → Mock(CA)    [✓]                   │
│                                                       │
│  ──────────── 预算控制 ────────────                   │
│  总预算: [$50.00]  已用: [$0.00]  剩余: [$50.00]     │
│  [启动实验 ▶]                                         │
└──────────────────────────────────────────────────────┘
```

---

# 十一、分阶段落地（Milestones）

## Phase 0：骨架（1 周）

> 目的：向量节点 GenomeGraph 能跑通，`pytest -q` 全绿。

- [ ] `genesis_v2/pyproject.toml`（运行依赖 + `[dev]` extra）
- [ ] 目录树完整创建
- [ ] `configs/genesis_v2.yaml` + `configs/backends.yaml`
- [ ] `genome/graph.py`：向量节点 GenomeGraph（D_node=64，矩阵权重）
- [ ] `genome/mutate.py`：12 种变异原语
- [ ] `genome/forward.py`：BLAS bundle 向量版前向计算 + `forward_scalar` 黄金对照
- [ ] `agent/agent.py`：Agent dataclass（输出分区 256:128:128:16）
- [ ] `engine/metabolism.py`：能量结算（含 v2 新项：API 成本、消息成本）
- [ ] `engine/reaper.py`：死亡判定（含自我修改致死）
- [ ] `env/mock.py`：MockMathEnvironment（多规则 CA）
- [ ] `storage/duckdb_store.py`：DuckDB 存储
- [ ] `cli.py`：`python -m genesis_v2` 能打印版本 + 跑 smoke
- [ ] `pytest -q` 全绿

## Phase 1：社交 + 多 LLM 环境（1-2 周）

- [ ] `social/comm_bus.py`：CommunicationBus（网格拓扑，comm_radius=2）
- [ ] `social/reputation.py`：声誉系统
- [ ] `env/multi_llm.py`：MultiLLMEnvironment（OpenAI 兼容格式）
- [ ] `env/budget.py`：BudgetManager + 降级策略
- [ ] `env/batch.py`：BatchedEnvironment
- [ ] `env/embed.py`：FrozenEmbeddingAtlas（向量 ↔ token）
- [ ] `engine/tick.py` + `tick_core.py`：新 tick 循环（含社交阶段）
- [ ] 社交选择压力（资源共享 + 信息不对称 + 集体任务）
- [ ] `scripts/dashboard.py`：Streamlit 配置面板 v1
- [ ] 验收：Agent 产生非随机消息 + 能在多个 LLM 间切换

## Phase 2：演化升级 + 探索机制（1-2 周）

- [ ] `genome/crossover.py`：NEAT 式交叉重组
- [ ] `evolution/breeder.py`：繁殖策略（Crossover 50% + Clone 30% + Migration 20%）
- [ ] `evolution/gen_memory.py`：GenerationalMemory + 跨代传承
- [ ] `evolution/survivor_bank.py`：精英种子库（JSON/DuckDB 序列化）
- [ ] `population/migration.py`：跨 LLM 迁移 + 适应性奖励
- [ ] `metrics/exploration.py`：探索奖励计算
- [ ] 自我修改通道（16 维输出 → 变异指令）
- [ ] 验收：跨代 fitness 上升 + 跨 LLM 适应速度提升 + 自我修改存活 Agent 出现

## Phase 3：翻译层 + 对话（1-2 周）

- [ ] `translation/translator.py`：多 LLM 翻译器（向量 ↔ 自然语言）
- [ ] 对话协议（人类 ↔ Agent 双向）
- [ ] 对话质量探针
- [ ] `scripts/dashboard.py`：Streamlit 面板 v2（对话界面 + 实时监控）
- [ ] 验收：能与 top-1% Agent 进行有意义的对话

## Phase 4：深度实验 + 报告（2 周）

- [ ] 因果推理探针 + 干预环境
- [ ] 沟通涌现探针
- [ ] 多 LLM 适应性探针
- [ ] 自我修改效率探针
- [ ] 探索奖励效果探针
- [ ] 3 种子 × 长程实验
- [ ] 对照组（无社交 / 无 crossover / 无探索奖励 / 无自我修改 / 单 LLM vs 多 LLM）
- [ ] 验收报告

---

# 十二、成功判据（MVE v2）

## 12.1 第一层：演化信号（必要条件）

| 现象 | 量化判据 |
|---|---|
| **压缩涌现** | 某岛屿平均 Compression 随代数单调上升，p < 0.05 |
| **预测涌现** | 某岛屿平均 PredictionError 随代数单调下降 |
| **沟通涌现** | 互信息 I(msg; action) > 0.5 bits，持续 > 100 ticks |
| **探索涌现** | 探索奖励为正的 Agent 占比 > 10%，且 fitness 高于平均 |
| **自我修改涌现** | 自我修改存活率 > 0.3，且修改后 fitness 显著提升 |

## 12.2 第二层：认知探针（判别"理解" vs "拟合"）

| 探针 | 判据 | 通过说明 |
|---|---|---|
| OOD 泛化 | OOD KL / 训练域 KL < 2.0 | 可迁移结构，非记忆 |
| 拓扑模块度 | Louvain Q > 0.4 | 功能分区涌现 |
| 多尺度一致性 | KL(16) / KL(1) < 3.0 | 世界模型，非一步预测 |
| **多 LLM 适应性** | 跨模型 KL 比 < 2.0，< 50 tick 适应 | 通用理解力 |
| **沟通有效性** | 参考游戏准确率 > 70%（随机基线 25%） | 有意义的通信协议 |
| **自我修改效率** | 修改后 fitness 变化 > 0 | 有效的自我优化 |

## 12.3 第三层：对话验证（最终目标）

| 指标 | 判据 |
|---|---|
| 输出困惑度 | < 随机阈值（输出非乱码） |
| 语义相关性 | prompt-response cosine similarity > 0.3 |
| 多轮连贯性 | 3 轮对话主题保持率 > 50% |
| 跨 LLM 一致性 | 同一问题在不同翻译下语义相似度 > 0.5 |

## 12.4 分级结论

| 等级 | 含义 | 判据 |
|---|---|---|
| **C-tier** | 系统跑通 | §12.1 ≥ 1 项 |
| **B-tier** | 演化成立 | §12.1 ≥ 3 项 |
| **A-tier** | 结构涌现 | §12.1 ≥ 3 项且 §12.2 ≥ 2 项 |
| **S-tier** | 可发论文 | §12.1 ≥ 3 项且 §12.2 ≥ 4 项，2 种子可复现 |
| **SS-tier** | 对话 AGI | S-tier + §12.3 全部通过 |

## 12.5 明确不承诺

- 不承诺 AGI 一定能在当前架构下涌现
- 不承诺对话能力在预算内可达
- **即使只达到 B-tier，演化信号本身也是有价值的科学发现**
- **如果全部为负，诚实公布 null result，不包装为"涌现"**

---

# 十三、风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| API 调用成本超预算 | 中 | BudgetManager 硬限 + 降级到 Mock |
| 400 Agent 全部饿死 | **高** | 初代给予充足 initial_energy=5000，α 调小 |
| 向量节点计算太慢 | 中 | D_node=64 平衡速度和表征能力；numpy BLAS 优化 |
| Agent 发不出有意义的 token | **高** | 嵌入投影矩阵的条件数检查；增大 projection_seed 搜索 |
| 跨 LLM 迁移全部失败 | 中 | 先从相似模型迁移（如 DeepSeek ↔ Qwen），再逐步扩大差异 |
| 社交层全是噪声 | 中 | 增大 w_social 权重；增加集体任务奖励 |
| 自我修改全是自毁 | 中 | 降低 selfmod_death_rate；增大能量门槛 |
| 探索奖励被 reward hack | 中 | 探索奖励上限 cap；定期重算 surprise_to_self |
| 语言翻译质量太差 | **高** | 增大投影维度；用多个 projection_seed 取最优 |
| 跑了一周无任何涌现信号 | **中** | Null result 也是结果，写清楚写入报告 |
| 看到结果后开始幻想"它懂了" | **极高** | 必须对照七项探针，全负就诚实说"只是拟合" |

---

# 十四、开工清单

立即可做的 5 件事：

1. 创建 `genesis_v2/` Python 项目：`pyproject.toml` + `uv sync --extra dev`
2. 写 `genome/graph.py`（向量节点） + `genome/mutate.py`（12 种变异） + 对应 `tests/`
3. 写 `genome/forward.py`（BLAS bundle 向量版） + `forward_scalar` 黄金对照
4. 写 `agent/agent.py`（输出分区 dataclass） + `engine/metabolism.py`（含 v2 新项）
5. 写 `env/mock.py`（多规则 CA） + `cli.py`（smoke 命令）

**Phase 0 完成的标志**：`pytest -q` 全绿，且 `python -m genesis_v2` 能跑一个 10 个体 × 100 tick 的不崩进程。

---

# 一句话总结

> 本规格规定：Agent 在多 LLM 物理法则中，通过社交压力、探索奖励、自我修改、跨代传承，在向量语义空间中自主演化认知结构；并通过冻结投影矩阵与自然语言空间对接，最终实现可对话的自主智能体。
>
> 它不声称 AGI，但作为**可跑、可证伪**的完整实验蓝图与工程验收依据。
