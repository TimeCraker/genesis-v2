"""Genesis v2 Dashboard — Bilingual control panel for AGI evolution experiments.
Genesis v2 控制台 — AGI 演化实验的中英对照控制面板。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import streamlit as st

# ─── Paths ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATUS_FILE = DATA_DIR / "experiment_status.json"
CONFIGS_DIR = PROJECT_ROOT / "configs"
BACKENDS_FILE = CONFIGS_DIR / "backends.yaml"
SURVIVORS_DIR = DATA_DIR / "survivors"
SEEDS_FILE = DATA_DIR / "selected_seeds.json"
ENV_KEYS_FILE = DATA_DIR / "api_keys.env"

# ─── Backend Presets / 后端预设 ─────────────────────────────────────

BACKEND_PRESETS: dict[str, dict] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-v4-flash",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "api_key_env": "GENESIS_DEEPSEEK_KEY",
    },
    "gemini": {
        "label": "Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "api_key_env": "GENESIS_GEMINI_KEY",
    },
    "mimo": {
        "label": "MiMo",
        "base_url": "https://api.xiaomi.com/v1",
        "default_model": "MiMo-7B",
        "models": ["MiMo-7B"],
        "api_key_env": "GENESIS_MIMO_KEY",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        "api_key_env": "GENESIS_OPENAI_KEY",
    },
    "mock": {
        "label": "Mock (Free / 免费)",
        "base_url": "",
        "default_model": "",
        "models": [],
        "api_key_env": "",
    },
}

# ─── Hardware Presets / 硬件配置预设 ────────────────────────────────

HARDWARE_PRESETS: dict[str, dict] = {
    "low": {
        "label": "Low / 低配 (Quick test, ~2min / 快速测试)",
        "agents": 10,
        "generations": 5,
        "ticks": 50,
        "top_fraction": 0.3,
        "mutation_rate": 0.2,
        "node_dim": 32,
        "input_nodes": 4,
        "hidden_nodes": 2,
        "edge_density": 0.15,
        "out_action": 2,
        "out_message": 1,
        "out_state": 1,
        "out_selfmod": 1,
        "initial_energy": 3000.0,
        "gen_ticks": 50,
    },
    "standard": {
        "label": "Standard / 标准 (Normal experiment, ~10min / 标准实验)",
        "agents": 20,
        "generations": 10,
        "ticks": 100,
        "top_fraction": 0.25,
        "mutation_rate": 0.15,
        "node_dim": 64,
        "input_nodes": 8,
        "hidden_nodes": 4,
        "edge_density": 0.2,
        "out_action": 4,
        "out_message": 2,
        "out_state": 2,
        "out_selfmod": 1,
        "initial_energy": 5000.0,
        "gen_ticks": 100,
    },
    "high": {
        "label": "High / 高配 (Deep evolution, ~30min / 深度演化)",
        "agents": 50,
        "generations": 30,
        "ticks": 200,
        "top_fraction": 0.2,
        "mutation_rate": 0.15,
        "node_dim": 64,
        "input_nodes": 8,
        "hidden_nodes": 8,
        "edge_density": 0.25,
        "out_action": 4,
        "out_message": 2,
        "out_state": 2,
        "out_selfmod": 1,
        "initial_energy": 8000.0,
        "gen_ticks": 200,
    },
}

# ─── Page Config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Genesis v2 Console",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* === Polish only — colors come from .streamlit/config.toml === */

/* Shape: rounded inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 8px !important;
}
div[data-baseweb="select"] > div {
    border-radius: 8px !important;
}

/* Shape: rounded tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

/* Shape: rounded buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
}

/* Shape: rounded cards */
div[data-testid="stMetric"] {
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetric"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

/* Shape: rounded expanders */
div[data-testid="stExpander"] {
    border-radius: 10px !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 500 !important;
}

/* Shape: rounded containers (API cards) */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px !important;
}

/* Shape: rounded alerts / dataframe */
.stAlert, div[data-baseweb="notification"] {
    border-radius: 10px !important;
}
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Shape: multiselect tags */
span[data-baseweb="tag"] {
    border-radius: 6px !important;
}

/* Metric help text */
.mh { font-size: 0.75rem; color: #86868b; margin-top: -0.3rem; margin-bottom: 0.5rem; line-height: 1.35; }
.mh .up { color: #34c759; font-weight: 600; }
.mh .down { color: #ff3b30; font-weight: 600; }

/* Info box custom */
.ib { background: #f5f5f7; border: 1px solid #e8e8ed; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.8rem; font-size: 0.85rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #d2d2d7; border-radius: 3px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────

def _load_yaml() -> dict:
    import yaml
    p = CONFIGS_DIR / "genesis_v2.yaml"
    return yaml.safe_load(p.read_text("utf-8")) if p.exists() else {}


def _save_yaml(cfg: dict) -> None:
    import yaml
    (CONFIGS_DIR / "genesis_v2.yaml").write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True), "utf-8"
    )


def _load_backends() -> dict:
    import yaml
    return yaml.safe_load(BACKENDS_FILE.read_text("utf-8")) if BACKENDS_FILE.exists() else {}


def _save_backends(cfg: dict) -> None:
    import yaml
    BACKENDS_FILE.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True), "utf-8")


def _save_api_keys(keys: dict[str, str]) -> None:
    """Persist API keys to data/api_keys.env so subprocess can read them."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in sorted(keys.items()) if v]
    ENV_KEYS_FILE.write_text("\n".join(lines) + "\n" if lines else "", "utf-8")


def _load_api_keys_to_env() -> None:
    """Load persisted API keys into os.environ on startup."""
    if not ENV_KEYS_FILE.exists():
        return
    for line in ENV_KEYS_FILE.read_text("utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            if v:
                os.environ[k.strip()] = v.strip()


# Load persisted API keys on import
_load_api_keys_to_env()


def _load_status() -> dict | None:
    if not STATUS_FILE.exists():
        return None
    try:
        return json.loads(STATUS_FILE.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _get_running_pid() -> int | None:
    """Return PID of a running experiment. Checks session_state first, then status file."""
    # Direct launch from this session
    pid = st.session_state.get("exp_pid")
    if pid:
        try:
            os.kill(pid, 0)  # check alive
            return pid
        except (OSError, ProcessLookupError):
            st.session_state.pop("exp_pid", None)

    # Fallback: read from status file
    status = _load_status()
    if status and status.get("running") and status.get("pid"):
        pid = status["pid"]
        try:
            os.kill(pid, 0)
            return pid
        except (OSError, ProcessLookupError):
            pass
    return None


def _mark_stale_if_dead() -> None:
    """If status says running but PID is dead, mark as finished."""
    status = _load_status()
    if not status or not status.get("running"):
        return
    pid = status.get("pid")
    if pid:
        try:
            os.kill(pid, 0)
            return  # still alive
        except (OSError, ProcessLookupError):
            pass
    # Process is dead — mark finished
    status["running"] = False
    status["note"] = "Process ended (detected on dashboard load)"
    STATUS_FILE.write_text(json.dumps(status, indent=2), "utf-8")


# Clean up stale status on import
_mark_stale_if_dead()


def _count_survivors() -> int:
    return len(list(SURVIVORS_DIR.glob("*.json"))) if SURVIVORS_DIR.exists() else 0


def _top_survivor_fitness() -> float | None:
    if not SURVIVORS_DIR.exists():
        return None
    best = 0.0
    found = False
    for f in SURVIVORS_DIR.glob("*.json"):
        try:
            fit = float(f.stem.split("_fit")[-1])
            best = max(best, fit)
            found = True
        except (IndexError, ValueError):
            pass
    return best if found else None


def _parse_survivor(fp: Path) -> dict:
    stem = fp.stem
    info = {"path": str(fp), "id": stem, "fitness": 0.0, "gen": "?"}
    try:
        info["fitness"] = float(stem.split("_fit")[-1])
    except (IndexError, ValueError):
        pass
    m = re.search(r"_gen(\d+)_", stem)
    if m:
        info["gen"] = m.group(1)
    return info


def _list_survivors(n: int = 50) -> list[dict]:
    if not SURVIVORS_DIR.exists():
        return []
    items = [_parse_survivor(f) for f in SURVIVORS_DIR.glob("*.json")]
    items.sort(key=lambda x: x["fitness"], reverse=True)
    return items[:n]


def _group_survivors_by_date() -> list[dict]:
    """Group survivors by date, newest first. Returns list of groups with stats."""
    if not SURVIVORS_DIR.exists():
        return []
    files = list(SURVIVORS_DIR.glob("*.json"))
    if not files:
        return []

    from collections import defaultdict
    day_groups: dict[str, list[tuple[Path, datetime]]] = defaultdict(list)
    for f in files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        day_groups[mtime.strftime("%Y-%m-%d")].append((f, mtime))

    groups = []
    for day_str in sorted(day_groups.keys(), reverse=True):
        day_entries = day_groups[day_str]
        day_files = [e[0] for e in day_entries]
        mtimes = [e[1] for e in day_entries]
        agents = [_parse_survivor(f) for f in day_files]
        agents.sort(key=lambda x: x["fitness"], reverse=True)
        fitnesses = [a["fitness"] for a in agents]
        gens = [a["gen"] for a in agents if a["gen"] != "?"]
        t_min = min(mtimes).strftime("%H:%M")
        t_max = max(mtimes).strftime("%H:%M")
        groups.append({
            "date": day_str,
            "time_range": f"{t_min}–{t_max}",
            "count": len(agents),
            "best_fitness": max(fitnesses) if fitnesses else 0.0,
            "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
            "max_gen": max(int(g) for g in gens) if gens else 0,
            "agents": agents,
        })
    return groups


def _breed_top_survivors(
    seed_paths: list[str],
    n_children: int,
    mutation_rate: float = 0.15,
) -> list[str]:
    """Breed top survivors via NEAT crossover + mutation. Returns paths to new seed files."""
    from genesis_v2.agent.agent import new_agent
    from genesis_v2.evolution.survivor_bank import load_agent, save_agent
    from genesis_v2.genome.crossover import crossover
    from genesis_v2.genome.mutate import mutate

    cfg = _load_yaml()
    gcfg = cfg.get("genome", {})
    pcfg = cfg.get("physics", {})

    loaded = []
    for p in seed_paths:
        try:
            a = load_agent(p)
            loaded.append(a)
        except Exception:
            pass

    if len(loaded) < 2:
        return []

    loaded.sort(key=lambda a: a.fitness, reverse=True)
    rng = np.random.default_rng(42)
    new_paths: list[str] = []

    for i in range(n_children):
        # Pick two random parents biased toward fitter
        idx = rng.choice(len(loaded), size=2, replace=False)
        pa, pb = loaded[int(idx[0])], loaded[int(idx[1])]

        # NEAT crossover
        child_genome = crossover(
            pa.genome, pb.genome,
            rng=rng,
            fitness_a=pa.fitness,
            fitness_b=pb.fitness,
        )

        # Mutate
        if rng.random() < mutation_rate:
            try:
                mutate(child_genome, rng)
            except Exception:
                pass

        child = new_agent(
            id=f"bred-{i}",
            genome=child_genome,
            initial_energy=float(pcfg.get("initial_energy", 5000.0)),
            generation=max(pa.generation, pb.generation) + 1,
        )
        fp = save_agent(child, SURVIVORS_DIR)
        new_paths.append(str(fp))

    return new_paths


def _help(cn: str, en: str, arrow: str = "") -> None:
    """Small bilingual note under an input. arrow: 'up' | 'down' | ''."""
    a = ""
    if arrow == "up":
        a = " <span class='up'>&uarr; Higher=Better</span>"
    elif arrow == "down":
        a = " <span class='down'>&darr; Lower=Better</span>"
    st.markdown(f"<div class='mh'>{cn} / <em>{en}</em>{a}</div>", unsafe_allow_html=True)


# ─── Page Title ─────────────────────────────────────────────────────

st.title("Genesis v2")
st.caption("Multi-LLM AGI Evolution Platform / 多 LLM 语义荒野 AGI 演化平台")

tab_cfg, tab_run, tab_mon = st.tabs([
    "⚙️ Config / 配置",
    "\U0001f680 Run / 启动",
    "\U0001f4ca Monitor / 监控",
])


# ================================================================
# TAB 1: Config / 配置
# ================================================================
with tab_cfg:
    cfg = _load_yaml()
    physics = cfg.get("physics", {})
    genome = cfg.get("genome", {})
    evolution = cfg.get("evolution", {})
    population = cfg.get("population", {})
    islands_cfg = population.get("islands", [])

    # ── Hardware Preset / 硬件预设 ──
    st.subheader("Hardware Preset / 硬件预设")

    hw_key = st.radio(
        "Select preset / 选择预设",
        options=["low", "standard", "high"],
        format_func=lambda k: HARDWARE_PRESETS[k]["label"],
        horizontal=True,
        label_visibility="collapsed",
    )
    hw = HARDWARE_PRESETS[hw_key]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents / 代理数", hw["agents"])
    c2.metric("Generations / 代数", hw["generations"])
    c3.metric("Ticks/Gen / 每代tick", hw["ticks"])
    c4.metric("Node Dim / 节点维度", hw["node_dim"])

    st.divider()

    # ────────────────────────────────────────────────────────────
    # Island Config / 岛屿配置 (BEFORE API config)
    # ────────────────────────────────────────────────────────────
    st.subheader("Islands / 岛屿配置")
    st.caption(
        "Each island independently selects its LLM backend. "
        "Different islands can use different models. / "
        "每个岛屿独立选择 LLM 后端。不同岛屿可以使用不同的模型。"
    )

    island_configs: list[dict] = []
    island_backends: list[str] = []

    if islands_cfg:
        cols = st.columns(min(len(islands_cfg), 4))
        for i, isl in enumerate(islands_cfg):
            with cols[i % 4]:
                name = isl.get("name", f"Island-{i}")
                st.markdown(f"### {name}")

                # Backend dropdown — each island independent
                backend_names = list(BACKEND_PRESETS.keys())
                cur_backend = isl.get("backend", "mock")
                if cur_backend not in BACKEND_PRESETS:
                    cur_backend = "mock"
                backend = st.selectbox(
                    "Backend / 后端",
                    options=backend_names,
                    index=backend_names.index(cur_backend),
                    format_func=lambda k: BACKEND_PRESETS[k]["label"],
                    key=f"isl_backend_{i}",
                )

                size = st.number_input(
                    "Pop / 种群",
                    value=int(isl.get("size", hw["agents"])),
                    min_value=2, max_value=1000, step=5,
                    key=f"isl_size_{i}",
                )
                mut = st.slider(
                    "Mut Rate / 变异率",
                    min_value=0.01, max_value=1.0,
                    value=float(isl.get("mutation_rate", hw.get("mutation_rate", 0.15))),
                    step=0.05,
                    key=f"isl_mut_{i}",
                )

                if backend == "mock":
                    st.caption("Free / 免费 — math CA environment")
                else:
                    bp = BACKEND_PRESETS[backend]
                    st.caption(f"API: {bp['base_url'][:40]}...")

                island_configs.append({
                    "name": name,
                    "size": size,
                    "mutation_rate": mut,
                    "backend": backend,
                })
                island_backends.append(backend)
    else:
        st.warning("No islands in config. / 配置中无岛屿。")

    st.divider()

    # ────────────────────────────────────────────────────────────
    # API Keys — Auto-generated based on unique backends
    # API 密钥 — 根据岛屿选择的后端自动生成配置卡片
    # ────────────────────────────────────────────────────────────
    unique_backends = sorted(set(b for b in island_backends if b != "mock"))
    mock_count = sum(1 for b in island_backends if b == "mock")

    st.subheader("API Keys / API 密钥配置")

    # Always initialize so Save button works even when all islands are mock
    api_values: dict[str, dict] = {}

    if not unique_backends:
        st.info(
            "All islands use Mock (free). No API keys needed. / "
            "所有岛屿使用 Mock（免费）。不需要 API 密钥。"
        )
    else:
        st.caption(
            f"{len(unique_backends)} backend(s) selected across islands. "
            f"Configure API keys below. / "
            f"岛屿共选择了 {len(unique_backends)} 个后端，请在下方配置 API 密钥。"
        )

        for idx, backend_key in enumerate(unique_backends):
            bp = BACKEND_PRESETS[backend_key]
            env_key = bp["api_key_env"]
            # Which islands use this backend?
            using_islands = [
                island_configs[j]["name"]
                for j, b in enumerate(island_backends)
                if b == backend_key
            ]
            island_names_str = ", ".join(using_islands)

            with st.container(border=True):
                st.markdown(f"**{bp['label']}** — used by: {island_names_str}")

                ac1, ac2 = st.columns(2)
                with ac1:
                    api_url = st.text_input(
                        "API URL / API 地址",
                        value=bp["base_url"],
                        key=f"api_url_{backend_key}",
                    )
                    _help(
                        "兼容 OpenAI 格式的 API 地址，已自动填充",
                        "OpenAI-compatible endpoint, auto-filled",
                    )
                with ac2:
                    # Text input — user can type any model name
                    model_hint = ", ".join(bp["models"]) if bp["models"] else ""
                    api_model = st.text_input(
                        "Model / 模型",
                        value=bp["default_model"],
                        key=f"api_model_{backend_key}",
                        placeholder=f"e.g. {model_hint}" if model_hint else "model-name",
                    )
                    _help(
                        "手动输入模型名称，如 deepseek-chat、gpt-4o 等",
                        "Type any model name, e.g. deepseek-chat, gpt-4o, etc.",
                    )

                api_key_val = st.text_input(
                    "API Key / API 密钥",
                    value=os.environ.get(env_key, ""),
                    type="password",
                    key=f"api_key_{backend_key}",
                )
                _help(
                    "输入你的 API 密钥，使用此后端的所有岛屿共享此密钥",
                    "API key for this backend. All islands using this backend share it.",
                )

                # Store for later use
                api_values[backend_key] = {
                    "url": api_url,
                    "model": api_model,
                    "key": api_key_val,
                }

                # Set env var
                if api_key_val:
                    os.environ[env_key] = api_key_val

                # Test Connection / 测试连通性
                if st.button(
                    "\U0001f50c Test Connection / 测试连通",
                    key=f"test_conn_{backend_key}",
                    use_container_width=True,
                ):
                    if not api_key_val:
                        st.error("Please enter API Key first / 请先输入 API Key")
                    else:
                        with st.spinner("Testing... / 测试中..."):
                            try:
                                import httpx
                                test_url = api_url.rstrip("/") + "/chat/completions"
                                headers = {
                                    "Authorization": f"Bearer {api_key_val}",
                                    "Content-Type": "application/json",
                                }
                                body = {
                                    "model": api_model,
                                    "messages": [{"role": "user", "content": "hi"}],
                                    "max_tokens": 5,
                                    "temperature": 0,
                                }
                                resp = httpx.post(
                                    test_url, json=body, headers=headers, timeout=15.0,
                                )
                                if resp.status_code == 200:
                                    st.success(
                                        f"Connected! Model: {api_model} / "
                                        f"连通成功！模型: {api_model}"
                                    )
                                else:
                                    detail = ""
                                    try:
                                        detail = resp.json().get("error", {}).get("message", resp.text[:200])
                                    except Exception:
                                        detail = resp.text[:200]
                                    st.error(
                                        f"Failed / 连接失败 — HTTP {resp.status_code}: {detail}"
                                    )
                            except httpx.ConnectError:
                                st.error(
                                    "Cannot reach server / 无法连接服务器。"
                                    "Check URL and network / 请检查 URL 和网络"
                                )
                            except httpx.TimeoutException:
                                st.error(
                                    "Timeout / 连接超时。"
                                    "Server too slow / 服务器响应过慢"
                                )
                            except Exception as e:
                                st.error(f"Error / 错误: {e}")

    # Mock environment info
    if mock_count > 0:
        with st.expander("About Mock Environment / 关于 Mock 环境", expanded=False):
            st.markdown("""
**Mock Environment / Mock 环境**

A self-contained mathematical environment using cellular automata (Rule110 / Rule30 / Rule90).
No external API calls — completely free to run.

一个自包含的数学环境，使用元胞自动机（Rule110 / Rule30 / Rule90）。
不需要外部 API 调用 — 完全免费运行。

**How it works / 工作原理:**
- Agents observe a binary CA pattern each tick / 代理每 tick 观测一个二值 CA 模式
- They predict the next state of the CA / 代理预测 CA 的下一个状态
- Prediction accuracy earns energy; errors cost energy / 预测准确赚取能量；失误消耗能量
- Death when energy reaches 0 / 能量归零即死

**Pros / 优点:** Free, fast, deterministic — great for testing and baselines.
免费、快速、确定性 — 适合测试和基线实验。

**Cons / 缺点:** Agents can only learn mathematical patterns, not language.
代理只能学习数学模式，无法学习语言。

**Use cases / 使用场景:**
- Debug and validate your setup / 调试和验证配置
- Baseline comparison / 基线对比实验
- Cost-free evolution / 零成本演化
            """)

    st.divider()

    # ── Advanced Parameters / 高级参数 (collapsible) ──
    # Initialize defaults from config (used when expander is collapsed)
    alpha = float(physics.get("alpha", 0.01))
    beta = float(physics.get("beta", 0.005))
    gamma = float(physics.get("gamma", 0.001))
    delta = float(physics.get("delta", 0.002))
    epsilon = float(physics.get("epsilon", 0.1))
    zeta = float(physics.get("zeta", 0.01))
    initial_energy = float(physics.get("initial_energy", hw["initial_energy"]))
    entropy_thresh = float(physics.get("topology_entropy_threshold", 5.0))
    w_pred = float(physics.get("w_pred", 1.0))
    w_comp = float(physics.get("w_comp", 0.5))
    w_bvar = float(physics.get("w_bvar", 0.3))
    w_explore = float(physics.get("w_explore", 0.2))
    node_dim = int(genome.get("node_dim", hw["node_dim"]))
    input_nodes = int(genome.get("input_nodes", hw["input_nodes"]))
    hidden_nodes = int(genome.get("initial_hidden_nodes", hw["hidden_nodes"]))
    edge_density = float(genome.get("initial_edge_density", hw["edge_density"]))
    out_action = int(genome.get("output_nodes_action", hw["out_action"]))
    out_message = int(genome.get("output_nodes_message", hw["out_message"]))
    out_state = int(genome.get("output_nodes_state", hw["out_state"]))
    out_selfmod = int(genome.get("output_nodes_selfmod", hw["out_selfmod"]))
    gen_ticks = int(evolution.get("generation_ticks", hw["gen_ticks"]))
    migration_interval = int(evolution.get("migration_interval_generations", 50))

    with st.expander("Advanced Parameters / 高级参数 (可微调)", expanded=False):

        st.markdown("**Physics / 物理常数**")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            alpha = st.number_input(
                "α token cost", value=alpha,
                format="%.4f", step=0.001, key="ph_a",
            )
            _help("token 消耗成本", "Token consumption cost", "down")
            beta = st.number_input(
                "β node cost", value=beta,
                format="%.4f", step=0.001, key="ph_b",
            )
            _help("节点维护成本", "Node maintenance cost", "down")
        with pc2:
            gamma = st.number_input(
                "γ latency", value=gamma,
                format="%.4f", step=0.0001, key="ph_g",
            )
            _help("前向计算深度惩罚", "Forward pass depth penalty", "down")
            delta = st.number_input(
                "δ edge cost", value=delta,
                format="%.4f", step=0.001, key="ph_d",
            )
            _help("突触连接成本", "Synaptic edge cost", "down")
        with pc3:
            epsilon = st.number_input(
                "ε API cost", value=epsilon,
                format="%.3f", step=0.01, key="ph_e",
            )
            _help("API 调用能量倍率", "API call energy multiplier", "down")
            zeta = st.number_input(
                "ζ msg cost", value=zeta,
                format="%.4f", step=0.001, key="ph_z",
            )
            _help("消息发送成本", "Message sending cost", "down")
        with pc4:
            initial_energy = st.number_input(
                "Init Energy / 初始能量",
                value=initial_energy,
                step=100.0, key="ph_ie",
            )
            _help("出生能量，归零即死", "Energy at birth, die at 0")
            entropy_thresh = st.number_input(
                "Entropy Thresh / 熵阈值",
                value=entropy_thresh,
                step=0.5, key="ph_et",
            )
            _help("拓扑熵超过此值即死", "Die when topology entropy exceeds this")

        st.markdown("**Reward Weights / 奖励权重**")
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            w_pred = st.number_input(
                "w_pred / 预测", value=w_pred,
                step=0.1, key="w_p",
            )
            _help("预测准确性奖励", "Prediction accuracy reward", "up")
        with wc2:
            w_comp = st.number_input(
                "w_comp / 压缩", value=w_comp,
                step=0.1, key="w_c",
            )
            _help("信息压缩效率奖励", "Compression efficiency reward", "up")
        with wc3:
            w_bvar = st.number_input(
                "w_bvar / 多样性", value=w_bvar,
                step=0.1, key="w_b",
            )
            _help("行为差异化奖励", "Behavioral diversity reward", "up")
        with wc4:
            w_explore = st.number_input(
                "w_explore / 探索", value=w_explore,
                step=0.1, key="w_e",
            )
            _help("新颖可预测行为奖励", "Novel-but-predictable behavior reward", "up")

        st.markdown("**Genome / 基因组**")
        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            node_dim = st.number_input(
                "Node Dim / 维度",
                value=node_dim,
                min_value=8, max_value=256, key="gn_d",
            )
            _help("每个神经元的向量维度", "Vector dimension per neuron")
            input_nodes = st.number_input(
                "Input Nodes / 输入节点",
                value=input_nodes,
                min_value=1, key="gn_i",
            )
            _help("接收环境观测的节点数", "Nodes receiving environment observations")
        with gc2:
            hidden_nodes = st.number_input(
                "Hidden Nodes / 隐藏节点",
                value=hidden_nodes,
                min_value=0, key="gn_h",
            )
            _help("初始隐藏层节点数", "Initial hidden layer nodes")
            edge_density = st.slider(
                "Edge Density / 边密度", 0.05, 1.0,
                edge_density,
                0.05, key="gn_ed",
            )
            _help("初始连接密度比例", "Initial connection density ratio")
        with gc3:
            out_action = st.number_input(
                "Action Nodes / 动作节点",
                value=out_action,
                min_value=1, key="gn_oa",
            )
            _help("输出到环境的动作节点", "Action output nodes to environment")
            out_message = st.number_input(
                "Msg Nodes / 消息节点",
                value=out_message,
                min_value=1, key="gn_om",
            )
            _help("发送给邻居的消息节点", "Message output nodes to neighbors")
        with gc4:
            out_state = st.number_input(
                "State Nodes / 状态节点",
                value=out_state,
                min_value=1, key="gn_os",
            )
            _help("反馈到下一 tick 的工作记忆", "Working memory fed back to next tick")
            out_selfmod = st.number_input(
                "SelfMod Nodes / 自修改节点",
                value=out_selfmod,
                min_value=1, key="gn_omod",
            )
            _help(
                "自我变异指令节点，需能量>10000",
                "Self-mutation instruction nodes, needs energy>10000",
            )

        st.markdown("**Evolution / 演化**")
        ec1, ec2 = st.columns(2)
        with ec1:
            gen_ticks = st.number_input(
                "Ticks/Gen / 每代tick",
                value=gen_ticks,
                min_value=10, step=10, key="ev_gt",
            )
            _help("每代运行的 tick 数", "Ticks per generation")
        with ec2:
            migration_interval = st.number_input(
                "Migration / 迁移间隔(代)",
                value=migration_interval,
                min_value=0, key="ev_mi",
            )
            _help(
                "精英在岛屿间迁移的间隔，0=不迁移",
                "Elite migration interval between islands, 0=no migration",
            )

    st.divider()

    # ── Save / 保存 ──
    if st.button("\U0001f4be Save Config / 保存配置", type="primary", use_container_width=True):
        # Update backends.yaml with API configs from cards
        bcfg = _load_backends()
        if "backends" not in bcfg:
            bcfg["backends"] = {}

        for backend_key, vals in api_values.items():
            bcfg["backends"][backend_key] = {
                "base_url": vals["url"],
                "api_key_env": BACKEND_PRESETS[backend_key]["api_key_env"],
                "model": vals["model"],
                "cost_per_1m_tokens": 0.0,
                "max_tokens": 64,
                "timeout_sec": 30.0,
            }
        _save_backends(bcfg)

        # Persist API keys to env file
        _save_api_keys({BACKEND_PRESETS[k]["api_key_env"]: v["key"] for k, v in api_values.items() if v.get("key")})

        # Update islands in config
        if island_configs:
            population["islands"] = island_configs

        new_cfg = {
            "physics": {
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
                "epsilon": epsilon, "zeta": zeta,
                "w_pred": w_pred, "w_comp": w_comp, "w_bvar": w_bvar,
                "w_adapt": physics.get("w_adapt", 0.5),
                "w_social": physics.get("w_social", 0.3),
                "w_explore": w_explore,
                "death_penalty": physics.get("death_penalty", 500.0),
                "initial_energy": initial_energy,
                "topology_entropy_threshold": entropy_thresh,
                "selfmod_energy_threshold": physics.get("selfmod_energy_threshold", 10000.0),
                "selfmod_energy_cost": physics.get("selfmod_energy_cost", 1000.0),
                "selfmod_death_rate": physics.get("selfmod_death_rate", 0.7),
            },
            "evolution": {
                "tick_rate": evolution.get("tick_rate", 2),
                "generation_ticks": gen_ticks,
                "migration_interval_generations": migration_interval,
            },
            "population": population,
            "genome": {
                "node_dim": node_dim, "input_nodes": input_nodes,
                "output_nodes_action": out_action, "output_nodes_message": out_message,
                "output_nodes_state": out_state, "output_nodes_selfmod": out_selfmod,
                "initial_hidden_nodes": hidden_nodes, "initial_edge_density": edge_density,
            },
            "environment": cfg.get("environment", {}),
            "evaluation": cfg.get("evaluation", {}),
        }
        _save_yaml(new_cfg)
        st.success("Config saved / 配置已保存")


# ================================================================
# TAB 2: Run / 启动
# ================================================================
with tab_run:
    st.subheader("Launch Experiment / 启动实验")

    # ── Preset Selector / 预设选择 ──
    run_presets = {
        "custom": {"label": "Custom / 自定义"},
        "quick": {"label": "Quick / 快速 (~2min)", "agents": 10, "gens": 3, "ticks": 30, "top_frac": 0.3, "mut": 0.2},
        "standard": {"label": "Standard / 标准 (~10min)", "agents": 20, "gens": 10, "ticks": 100, "top_frac": 0.25, "mut": 0.15},
        "deep": {"label": "Deep / 深度 (~30min)", "agents": 50, "gens": 30, "ticks": 200, "top_frac": 0.2, "mut": 0.15},
        "marathon": {"label": "Marathon / 马拉松 (~2h)", "agents": 100, "gens": 100, "ticks": 200, "top_frac": 0.15, "mut": 0.12},
    }
    run_preset_key = st.selectbox(
        "Preset / 预设",
        options=list(run_presets.keys()),
        format_func=lambda k: run_presets[k]["label"],
    )
    rp = run_presets[run_preset_key]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_agents = st.number_input(
            "Agents / 代理数",
            value=rp.get("agents", 20), min_value=2, max_value=1000, step=5,
        )
    with c2:
        total_gens = st.number_input(
            "Generations / 代数",
            value=rp.get("gens", 10), min_value=1, max_value=10000, step=5,
        )
    with c3:
        ticks_per = st.number_input(
            "Ticks / Gen",
            value=rp.get("ticks", 100), min_value=10, max_value=10000, step=50,
        )
    with c4:
        top_frac = st.slider(
            "Elite % / 精英比例",
            0.05, 0.5, rp.get("top_frac", 0.25), 0.05,
        )
    with c5:
        mut_rate = st.slider(
            "Mut Rate / 变异率",
            0.01, 1.0, rp.get("mut", 0.15), 0.05,
        )

    seed = st.number_input("Random Seed / 随机种子", value=42, min_value=0)

    st.divider()

    # ── Elite Seed Bank / 精英种子库 ──
    n_surv = _count_survivors()
    best_fit = _top_survivor_fitness()

    c1, c2, c3 = st.columns(3)
    c1.metric("Seeds / 种子数", n_surv)
    c2.metric("Best Fitness / 最高适应度", f"{best_fit:.1f}" if best_fit else "N/A")
    c3.metric("Est. Time / 预估时间", f"~{max(1, n_agents * total_gens * ticks_per // 50000)}min")

    # Auto-determine seed count: ~10-30% of population, min 3, max 30
    auto_seed_n = max(3, min(30, n_agents // 3))

    seed_mode = st.radio(
        "Seed Mode / 种子模式",
        ["Auto / 自动加载最优", "Manual / 手动选择", "None / 无种子"],
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_seed_paths: list[str] = []

    if seed_mode == "Manual / 手动选择":
        seed_groups = _group_survivors_by_date()

        if not seed_groups:
            st.info("No survivors yet. Run an experiment first. / 暂无种子，请先跑实验。")
        else:
            st.markdown(
                f"**Select seeds / 选择种子** — "
                f"click a date on the left, pick agents on the right / "
                f"左侧点选日期，右侧选择个体"
            )

            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown("**实验日期 / Runs**")
                for i, grp in enumerate(seed_groups):
                    btn_label = (
                        f"📅 {grp['date']} {grp['time_range']}  "
                        f"({grp['count']} agents, "
                        f"best={grp['best_fitness']:.0f})"
                    )
                    if st.button(btn_label, key=f"seed_grp_{i}", use_container_width=True):
                        st.session_state["seed_selected_group"] = i

            sel_grp_idx = st.session_state.get("seed_selected_group", 0)
            sel_grp_idx = min(sel_grp_idx, len(seed_groups) - 1)
            selected_group = seed_groups[sel_grp_idx]

            with col_right:
                st.markdown(
                    f"**{selected_group['date']}** — "
                    f"{selected_group['count']} agents, "
                    f"best fitness = **{selected_group['best_fitness']:.1f}**"
                )

                # Quick-select buttons row
                qc1, qc2, qc3 = st.columns(3)
                with qc1:
                    if st.button(
                        f"🏆 Select Top {auto_seed_n} / 选最优 {auto_seed_n} 个",
                        key="btn_top_n", use_container_width=True,
                    ):
                        top_agents = selected_group["agents"][:auto_seed_n]
                        st.session_state["seed_manual_picks"] = [a["path"] for a in top_agents]
                with qc2:
                    if st.button(
                        f"🧬 Breed {auto_seed_n} New Seeds / 繁殖 {auto_seed_n} 个新种子",
                        key="btn_breed", use_container_width=True,
                    ):
                        top_for_breed = selected_group["agents"][:min(10, len(selected_group["agents"]))]
                        breed_paths = [a["path"] for a in top_for_breed]
                        with st.spinner("Breeding... / 繁殖中..."):
                            new_paths = _breed_top_survivors(
                                breed_paths, n_children=auto_seed_n,
                                mutation_rate=mut_rate,
                            )
                        if new_paths:
                            st.session_state["seed_manual_picks"] = new_paths
                            st.success(
                                f"Bred {len(new_paths)} new seeds / "
                                f"成功繁殖 {len(new_paths)} 个新种子"
                            )
                        else:
                            st.warning("Need ≥2 seeds to breed / 至少需要 2 个种子才能繁殖")
                with qc3:
                    if st.button("Clear / 清空选择", key="btn_clear_picks", use_container_width=True):
                        st.session_state["seed_manual_picks"] = []

                # Agent table with checkboxes
                st.markdown("---")
                current_picks = set(st.session_state.get("seed_manual_picks", []))

                for j, ag in enumerate(selected_group["agents"][:50]):
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                    with c1:
                        checked = st.checkbox(
                            ag["id"],
                            value=ag["path"] in current_picks,
                            key=f"seed_chk_{i}_{j}",
                        )
                    with c2:
                        st.caption(f"fitness: {ag['fitness']:.1f}")
                    with c3:
                        st.caption(f"gen: {ag['gen']}")
                    with c4:
                        mtime = datetime.fromtimestamp(Path(ag["path"]).stat().st_mtime)
                        st.caption(mtime.strftime("%H:%M"))

                    if checked:
                        current_picks.add(ag["path"])
                    else:
                        current_picks.discard(ag["path"])

                st.session_state["seed_manual_picks"] = list(current_picks)

            selected_seed_paths = st.session_state.get("seed_manual_picks", [])

    elif seed_mode == "Auto / 自动加载最优":
        survivors = _list_survivors(auto_seed_n)
        if survivors:
            selected_seed_paths = [s["path"] for s in survivors]
            st.caption(
                f"Will load {len(survivors)} top seeds / "
                f"将加载 {len(survivors)} 个最优种子 "
                f"(≈10-30% of population / 约为种群的 10-30%)"
            )

    st.divider()

    # ── Launch / 启动 ──
    c1, c2 = st.columns(2)
    with c1:
        if st.button("\U0001f680 Start / 开始实验", type="primary", use_container_width=True):
            # Write initial status so Monitor shows data immediately
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            STATUS_FILE.write_text(json.dumps({
                "running": True,
                "mode": "starting",
                "generation": 0,
                "total_generations": total_gens,
                "tick": 0,
                "alive_count": 0,
                "total_agents": n_agents * 4,
                "islands": [],
                "history": [],
                "note": "Initializing... / 初始化中...",
            }, indent=2), "utf-8")

            seeds_arg = None
            if selected_seed_paths:
                SEEDS_FILE.write_text(json.dumps(selected_seed_paths, indent=2), "utf-8")
                seeds_arg = str(SEEDS_FILE)

            cmd = [
                sys.executable, "-m", "genesis_v2", "experiment",
                "--agents", str(n_agents),
                "--generations", str(total_gens),
                "--ticks", str(ticks_per),
                "--top-fraction", str(top_frac),
                "--mutation-rate", str(mut_rate),
                "--seed", str(seed),
            ]
            if seeds_arg:
                cmd.extend(["--seeds-file", seeds_arg])

            # Reload persisted API keys into env so subprocess inherits them
            _load_api_keys_to_env()

            log_file = DATA_DIR / "experiment.log"
            log_fh = open(log_file, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT),
                stdout=log_fh, stderr=subprocess.STDOUT, text=True,
            )
            # Keep file handle alive in session state to prevent GC
            st.session_state["exp_log_fh"] = log_fh
            st.session_state["exp_pid"] = proc.pid
            st.session_state["exp_log_file"] = str(log_file)
            st.success(f"Started! PID: {proc.pid} / 已启动！")
            st.caption(f"Log: `{log_file}`")

    with c2:
        if st.button("⏹ Stop / 停止", type="secondary", use_container_width=True):
            pid = _get_running_pid()
            if pid:
                try:
                    os.kill(pid, 9)
                    st.warning(f"Killed PID {pid}")
                except (OSError, ProcessLookupError):
                    st.info("Already finished / 已结束")
                # Mark status as finished
                status = _load_status()
                if status:
                    status["running"] = False
                    status["note"] = "Stopped by user"
                    STATUS_FILE.write_text(json.dumps(status, indent=2), "utf-8")
                st.session_state.pop("exp_pid", None)
            else:
                st.info("No experiment running / 无运行中的实验")


# ================================================================
# TAB 3: Monitor / 监控
# ================================================================

@st.fragment(run_every=timedelta(seconds=5))
def monitor_panel():
    """Auto-refreshing monitor — only this fragment re-renders, no full page reload."""
    import pandas as pd

    _mark_stale_if_dead()
    status = _load_status()

    if status is None:
        st.markdown(
            '<div class="ib">No experiment data yet. Launch an experiment from the Run tab. / '
            '暂无实验数据。请在「启动」标签页启动实验。</div>',
            unsafe_allow_html=True,
        )
        return

    running = status.get("running", False)
    history = status.get("history", [])

    # Inject live data point so charts appear even before first generation completes
    if running and not any(h.get("generation") == status.get("generation", -1) for h in history):
        live_point = {
            "generation": status.get("generation", 0),
            "alive_count": status.get("alive_count", 0),
            "mean_fitness": status.get("mean_fitness", 0),
            "best_fitness": status.get("best_fitness", 0),
            "mean_energy": status.get("mean_energy", 0),
            "mean_pred_err": status.get("mean_pred_err", 0),
        }
        history = [live_point] + history

    # Compute deltas from history for colored arrows
    def _delta(key: str, positive_is_good: bool = True):
        """Return (delta_str, color) for st.metric delta_color."""
        if len(history) < 2:
            return None
        cur = history[0].get(key, 0)
        prev = history[1].get(key, 0)
        d = cur - prev
        if d == 0:
            return None
        # st.metric delta_color="normal": green if delta>0, red if delta<0
        # We need to flip for metrics where decrease is good (e.g. pred_err)
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.2f}"

    def _delta_color(key: str, positive_is_good: bool = True):
        """Return "normal" or "inverse" for st.metric delta_color."""
        if len(history) < 2:
            return "off"
        cur = history[0].get(key, 0)
        prev = history[1].get(key, 0)
        d = cur - prev
        if d == 0:
            return "off"
        return "normal" if positive_is_good else "inverse"

    # ── Status Bar / 状态栏 ──
    if running:
        gen = status.get("generation", 0)
        total_gens = status.get("total_generations", 0)
        tick_in_gen = status.get("tick_in_gen", 0)
        ticks_per_gen = status.get("ticks_per_gen", 0)
        elapsed = status.get("elapsed_seconds", 0)
        st.success(f"▶ Running / 运行中 — Gen {gen}/{total_gens}  |  "
                   f"Tick {tick_in_gen}/{ticks_per_gen}  |  "
                   f"Elapsed {elapsed:.0f}s  |  "
                   f"auto-refresh 5s")
        if ticks_per_gen > 0:
            st.progress(tick_in_gen / ticks_per_gen)
            st.caption("进度条 = 当前代内 tick 完成度 (tick/每代总tick)。满格代表一代结束，进入繁殖和下一代。")
            st.caption("Progress bar = current generation's tick completion (tick / ticks_per_gen). "
                       "Full = generation ends, breeding + next gen starts.")
    else:
        st.info("⏹ Finished / 已结束")

    # ── KPI Row / 核心指标（带增益箭头） ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        gen = status.get("generation", 0)
        total_gens = status.get("total_generations", 0)
        tick_str = ""
        if "tick_in_gen" in status and "ticks_per_gen" in status:
            tick_str = f" (tick {status['tick_in_gen']}/{status['ticks_per_gen']})"
        st.metric(
            "Generation / 代数",
            f"{gen}/{total_gens}{tick_str}",
        )
    with k2:
        alive = status.get("alive_count", 0)
        total = status.get("total_agents", 0)
        st.metric(
            "Alive / 存活",
            f"{alive}/{total}",
            delta=_delta("alive_count", positive_is_good=True),
            delta_color=_delta_color("alive_count", positive_is_good=True),
        )
    with k3:
        st.metric(
            "Best Fitness / 最高适应度",
            f"{status.get('best_fitness', 0):.1f}",
            delta=_delta("best_fitness", positive_is_good=True),
            delta_color=_delta_color("best_fitness", positive_is_good=True),
        )
    with k4:
        st.metric(
            "Mean Energy / 平均能量",
            f"{status.get('mean_energy', 0):.0f}",
            delta=_delta("mean_energy", positive_is_good=True),
            delta_color=_delta_color("mean_energy", positive_is_good=True),
        )

    # ── KPI help ──
    with st.expander("What do these mean? / 这些指标什么意思？", expanded=False):
        st.markdown("""
| Metric / 指标 | Good / 好 | Bad / 差 | Meaning / 含义 |
|---|---|---|---|
| **Alive / 存活** | >60% | <20% | 存活率。骤降=选择压力过大 / Survival rate. Crash=harsh selection |
| **Best Fitness / 最高适应度** | 上升 (绿色) | 下降 (红色) | 精英代理的累计奖励 / Elite agent cumulative reward |
| **Mean Energy / 平均能量** | 稳定/上升 (绿色) | 持续下降 (红色) | 平均生命资源。归零即死 / Avg life resource. 0=death |
| **Generation / 代数** | 进行中 | - | 当前进度 / Current progress |
| **箭头颜色** | 绿色 = 正向变化 | 红色 = 负向变化 | 与上一代的差值 / Delta from previous generation |
        """)

    # ── Experiment Log / 实验日志 ──
    log_path = Path(DATA_DIR / "experiment.log")
    if log_path.exists():
        with st.expander("Experiment Log / 实验日志 (tail)", expanded=False):
            try:
                log_content = log_path.read_text("utf-8")
                lines = log_content.strip().splitlines()
                tail = "\n".join(lines[-50:]) if len(lines) > 50 else log_content
                st.code(tail, language=None)
            except OSError:
                st.caption("Cannot read log / 无法读取日志")

    st.divider()

    # ── Charts / 图表 ──
    if history:
        df = pd.DataFrame(history)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.subheader("Fitness / 适应度")
            if "best_fitness" in df.columns:
                cdf = df[["generation", "mean_fitness", "best_fitness"]].set_index("generation")
                st.line_chart(cdf, use_container_width=True)

        with ch2:
            st.subheader("Energy / 能量")
            if "mean_energy" in df.columns:
                cdf = df[["generation", "mean_energy"]].set_index("generation")
                st.line_chart(cdf, use_container_width=True)

        with st.expander("More Charts / 更多图表", expanded=False):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.subheader("Prediction Error / 预测误差")
                if "mean_pred_err" in df.columns:
                    cdf = df[["generation", "mean_pred_err"]].set_index("generation")
                    st.line_chart(cdf, use_container_width=True)
                _help(
                    "KL散度，越低越好。代理对环境的理解程度",
                    "KL divergence, lower=better. How well agents understand the environment",
                    "down",
                )

            with sc2:
                st.subheader("Alive Pop / 存活种群")
                if "alive_count" in df.columns:
                    cdf = df[["generation", "alive_count"]].set_index("generation")
                    st.line_chart(cdf, use_container_width=True)

            # Per-island fitness
            st.subheader("Per-Island Fitness / 各岛适应度")
            island_names = set()
            for h in history:
                for isl in h.get("islands", []):
                    island_names.add(isl["island"])
            if island_names:
                rows = []
                for h in history:
                    row = {"generation": h["generation"]}
                    for isl in h.get("islands", []):
                        row[isl["island"]] = isl.get("best_fitness", 0)
                    rows.append(row)
                idf = pd.DataFrame(rows).set_index("generation")
                st.line_chart(idf, use_container_width=True)

    st.divider()

    # ── Leaderboard / 精英代理排行榜 ──
    with st.expander("Top Agents / 精英代理排行榜", expanded=True):
        top = status.get("top_agents", [])
        if top:
            leaderboard_df = pd.DataFrame(top)
            leaderboard_df.columns = [
                "排名", "ID", "适应度 (越高越好)", "能量 (越高越好)",
                "存活tick数", "节点数", "边数",
            ]
            st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
            _help(
                "**适应度**: 代理的累计奖励，越高越好。反映预测准确度和能量效率。"
                "**能量**: 当前生命资源，归零即死。"
                "**节点数/边数**: 基因组复杂度，变异导致增长。",
                "**Fitness**: cumulative reward, higher=better. Reflects prediction accuracy & energy efficiency. "
                "**Energy**: current life resource, 0=death. "
                "**Nodes/Edges**: genome complexity, grows via mutation.",
            )
        else:
            st.caption("No data yet / 暂无数据")

    st.divider()

    # ── Experiment Config / 实验配置 ──
    islands_data = status.get("islands", [])
    if islands_data:
        with st.expander("Experiment Config / 实验配置", expanded=True):
            st.subheader("Islands / 岛屿概况")
            for isl in islands_data:
                name = isl.get("island", "?")
                alive = isl.get("alive", 0)
                best_f = isl.get("best_fitness", 0)
                mean_f = isl.get("mean_fitness", 0)
                mean_e = isl.get("mean_energy", 0)

                # Color code: green if alive>50%, red if <20%
                alive_color = "green" if alive >= 50 else ("orange" if alive >= 20 else "red")

                ic1, ic2, ic3, ic4, ic5 = st.columns(5)
                ic1.markdown(f"**{name}**")
                ic2.markdown(f":{alive_color}[存活: {alive}]")
                ic3.metric("最高适应度", f"{best_f:.0f}")
                ic4.metric("平均适应度", f"{mean_f:.0f}")
                ic5.metric("平均能量", f"{mean_e:.0f}")

            # ── Advanced Config / 高级参数 ──
            with st.expander("Advanced Config / 高级参数 (只读)", expanded=False):
                try:
                    import yaml
                    config_path = CONFIGS_DIR / "genesis_v2.yaml"
                    if config_path.exists():
                        with open(config_path, "r", encoding="utf-8") as f:
                            cfg = yaml.safe_load(f)

                        # Physics constants
                        st.markdown("**Physics / 物理常数**")
                        physics = cfg.get("physics", {})
                        phys_rows = []
                        for k, v in physics.items():
                            phys_rows.append({"Parameter / 参数": k, "Value / 值": v})
                        st.dataframe(pd.DataFrame(phys_rows), hide_index=True, use_container_width=True)

                        # Genome params
                        st.markdown("**Genome / 基因组参数**")
                        genome = cfg.get("genome", {})
                        gen_rows = []
                        for k, v in genome.items():
                            gen_rows.append({"Parameter / 参数": k, "Value / 值": v})
                        st.dataframe(pd.DataFrame(gen_rows), hide_index=True, use_container_width=True)

                        # Evolution params
                        st.markdown("**Evolution / 演化参数**")
                        evo = cfg.get("evolution", {})
                        evo_rows = []
                        for k, v in evo.items():
                            evo_rows.append({"Parameter / 参数": k, "Value / 值": v})
                        st.dataframe(pd.DataFrame(evo_rows), hide_index=True, use_container_width=True)

                        # Environment params
                        st.markdown("**Environment / 环境参数**")
                        env = cfg.get("environment", {})
                        env_rows = []
                        for k, v in env.items():
                            env_rows.append({"Parameter / 参数": k, "Value / 值": v})
                        st.dataframe(pd.DataFrame(env_rows), hide_index=True, use_container_width=True)

                        # Population / islands detail
                        st.markdown("**Population / 种群配置**")
                        pop = cfg.get("population", {})
                        for isl_cfg in pop.get("islands", []):
                            st.markdown(
                                f"- **{isl_cfg.get('name', '?')}**: "
                                f"backend={isl_cfg.get('backend', '?')}, "
                                f"size={isl_cfg.get('size', '?')}, "
                                f"mutation_rate={isl_cfg.get('mutation_rate', '?')}"
                            )
                except Exception as e:
                    st.warning(f"Cannot load config / 无法加载配置: {e}")


with tab_mon:
    monitor_panel()
