from __future__ import annotations

from pathlib import Path
import sys
import time
import json
import subprocess
from typing import Any

import streamlit as st
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.a2c_agent import A2CAmazonsAgent
from src.agents.alphazero_mcts import AlphaZeroMCTS, PUCTConfig
from src.agents.alphazero_net import load_checkpoint
from src.agents.alphazero_player import AlphaZeroPlayer
from src.agents.bc_player import BCPolicyPlayer
from src.agents.dqn_agent import DQNAmazonsAgent
from src.agents.heuristic_agent import HeuristicAmazonsAgent
from src.agents.minimax_agent import MinimaxAmazonsAgent
from src.agents.mcts_agent import MCTSAmazonsAgent, MCTSConfig
from src.agents.ppo_agent import PPOAmazonsAgent
from src.agents.q_learning_agent import TabularQLearningAmazonsAgent
from src.agents.random_agent import RandomAmazonsAgent
from src.envs.amazons_env import AmazonsConfig, MiniAmazonsEnv
from src.train.innovation_utils import save_case_record


def render_board_html(board) -> str:
    icon = {0: "", 1: "⚪", 2: "⚫", 3: "✖"}
    bg = {0: "#f0d9b5", 1: "#f0d9b5", 2: "#f0d9b5", 3: "#777777"}
    html = [
        "<table style='border-collapse: collapse; border: 2px solid #333;'>",
    ]
    for row in board:
        html.append("<tr>")
        for v in row:
            cell_bg = bg.get(v, "#f0d9b5")
            piece = icon.get(v, "")
            html.append(
                "<td style='width: 44px; height: 44px; text-align: center; vertical-align: middle;"
                f"border: 1px solid #333; background:{cell_bg}; font-size: 26px;'>{piece}</td>"
            )
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)


def list_model_paths() -> list[str]:
    candidates = []
    root = ROOT / "results" / "models"
    if not root.exists():
        return candidates
    for ext in ("*.pt", "*.json"):
        for p in root.rglob(ext):
            try:
                candidates.append(str(p.relative_to(ROOT)).replace("\\", "/"))
            except ValueError:
                candidates.append(str(p))
    return sorted(set(candidates))


def model_matches_type(agent_type: str, model_path: str) -> bool:
    t = agent_type.lower()
    mp = model_path.lower()
    if t in {"random", "heuristic", "minimax_d1", "minimax_d2", "mcts_80"}:
        return False
    if t == "q":
        return mp.endswith(".json") and ("_q" in mp or "qlearning" in mp or "/q" in mp)
    if t == "dqn":
        return mp.endswith(".pt") and "dqn" in mp
    if t == "a2c":
        return mp.endswith(".pt") and "a2c" in mp
    if t == "ppo":
        return mp.endswith(".pt") and "ppo" in mp
    if t == "bc":
        return mp.endswith(".pt") and ("bc" in mp or "policy_from_mcts" in mp)
    if t == "az":
        return mp.endswith(".pt") and ("az_" in mp or "alphazero" in mp)
    return False


def filtered_model_choices(agent_type: str, all_models: list[str]) -> list[str]:
    return [m for m in all_models if model_matches_type(agent_type, m)]


def list_training_logs() -> list[str]:
    root = ROOT / "results"
    if not root.exists():
        return []
    out: list[str] = []
    for p in root.rglob("*log*.csv"):
        try:
            out.append(str(p.relative_to(ROOT)).replace("\\", "/"))
        except ValueError:
            out.append(str(p))
    return sorted(set(out))


def moving_average(series: pd.Series, window: int = 20) -> pd.Series:
    if series.empty:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def list_figure_paths() -> list[str]:
    root = ROOT / "results" / "figures"
    if not root.exists():
        return []
    out: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for p in root.rglob(ext):
            try:
                out.append(str(p.relative_to(ROOT)).replace("\\", "/"))
            except ValueError:
                out.append(str(p))
    return sorted(set(out))


def load_strength_table() -> tuple[dict[str, float], str]:
    """
    Prefer n40 full matrix ranking as strength source.
    Fallback to n30 long-model ranking.
    """
    candidates = [
        ROOT / "results" / "eval_matrix_all_az_n40.csv",
        ROOT / "results" / "eval_matrix_long_models_n30.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                s = df.groupby("agent0")["win_rate_agent0"].mean().to_dict()
                return {str(k): float(v) for k, v in s.items()}, p.name
            except Exception:
                continue
    return {}, "none"


def infer_agent_key(agent_type: str, model_path: str) -> str:
    t = agent_type.lower()
    mp = (model_path or "").lower()
    if t == "heuristic":
        return "heu"
    if t.startswith("minimax") or t == "mcts_80":
        return "mm"
    if t == "dqn":
        return "dqnL"
    if t == "a2c":
        return "a2cL"
    if t == "ppo":
        return "ppoL"
    if t == "bc":
        return "bcL"
    if t == "az":
        if "stage1" in mp:
            return "az1"
        if "stage2" in mp:
            return "az2"
        if "stage3" in mp:
            return "az3"
        return "az2"
    if t == "q":
        return "q0"
    if t == "random":
        return "rnd"
    return t


def build_from_rank_key(key: str, device: str, az_sims: int):
    k = key.lower()
    if k == "rnd":
        return build_agent("random", "", device, az_sims)
    if k == "heu":
        return build_agent("heuristic", "", device, az_sims)
    if k == "mm":
        return build_agent("minimax_d1", "", device, az_sims)
    if k == "q0":
        return build_agent("q", "results/models/agent0_q.json", device, az_sims)
    if k == "dqnl":
        return build_agent("dqn", "results/models/agent0_dqn.pt", device, az_sims)
    if k == "a2cl":
        return build_agent("a2c", "results/models/a2c_long_20260331/agent0_a2c.pt", device, az_sims)
    if k == "ppol":
        return build_agent("ppo", "results/models/ppo_long_20260331/agent0_ppo.pt", device, az_sims)
    if k == "bcl":
        return build_agent("bc", "results/models/bc_long_20260331/bc_policy_from_mcts.pt", device, az_sims)
    if k == "az1":
        return build_agent("az", "results/models/az_stage1_20260331/alphazero_pvnet.pt", device, az_sims)
    if k == "az2":
        return build_agent("az", "results/models/az_stage2_20260331/alphazero_pvnet.pt", device, az_sims)
    if k == "az3":
        return build_agent("az", "results/models/az_stage3_20260331/alphazero_pvnet.pt", device, az_sims)
    raise ValueError(f"Unknown rank key: {key}")


def build_agent(name: str, model_path: str, device: str, az_sims: int):
    n = name.lower()
    if n == "random":
        return RandomAmazonsAgent()
    if n == "heuristic":
        return HeuristicAmazonsAgent()
    if n == "minimax_d1":
        return MinimaxAmazonsAgent(depth=1)
    if n == "minimax_d2":
        return MinimaxAmazonsAgent(depth=2)
    if n == "mcts_80":
        return MCTSAmazonsAgent(config=MCTSConfig(simulations=80, rollout_depth=40))
    if n == "q":
        return TabularQLearningAmazonsAgent.load(model_path)
    if n == "dqn":
        return DQNAmazonsAgent.load(model_path, device=device)
    if n == "a2c":
        return A2CAmazonsAgent.load(model_path, device=device)
    if n == "ppo":
        return PPOAmazonsAgent.load(model_path, device=device)
    if n == "bc":
        net = load_checkpoint(model_path, device=device)
        return BCPolicyPlayer(size=net.size, net=net, device=device)
    if n == "az":
        net = load_checkpoint(model_path, device=device)
        mcts = AlphaZeroMCTS(size=net.size, net=net, cfg=PUCTConfig(simulations=az_sims), device=device)
        return AlphaZeroPlayer(size=net.size, net=net, mcts=mcts)
    raise ValueError(f"Unknown agent type: {name}")


def ensure_state():
    if "env" not in st.session_state:
        st.session_state.env = MiniAmazonsEnv(AmazonsConfig(size=6, max_turns=80))
        st.session_state.obs = st.session_state.env.reset()
        st.session_state.done = False
        st.session_state.info = {"winner": -1}
        st.session_state.log = []
        st.session_state.history = [([row[:] for row in st.session_state.env.board], 0, 0)]
        st.session_state.metric_rows = []
        st.session_state.auto_play = False
        st.session_state.auto_remaining = 0
        st.session_state.auto_delay = 0.5
        st.session_state.board_size = 6
        st.session_state.bg_task = None


def reset_game():
    env = st.session_state.env
    st.session_state.obs = env.reset()
    st.session_state.done = False
    st.session_state.info = {"winner": -1}
    st.session_state.log = []
    st.session_state.history = [([row[:] for row in env.board], 0, 0)]
    st.session_state.metric_rows = []


def rebuild_env(size: int):
    size = int(size)
    st.session_state.env = MiniAmazonsEnv(AmazonsConfig(size=size, max_turns=200 if size >= 12 else 80))
    st.session_state.board_size = size
    st.session_state.obs = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.info = {"winner": -1}
    st.session_state.log = []
    st.session_state.history = [([row[:] for row in st.session_state.env.board], 0, 0)]
    st.session_state.metric_rows = []
    st.session_state.auto_play = False
    st.session_state.auto_remaining = 0


def step_once(agent0: Any, agent1: Any):
    env = st.session_state.env
    if st.session_state.done:
        return
    p = env.current_player
    legal = env.legal_actions(p)
    legal_n = len(legal)
    if not legal:
        st.session_state.done = True
        st.session_state.info = {"winner": 1 - p}
        return

    if p == 0:
        action = agent0.select_action(st.session_state.obs, legal, training=False)
    else:
        action = agent1.select_action(st.session_state.obs, legal, training=False)

    obs, rewards, done, info = env.step(action)
    st.session_state.obs = obs
    st.session_state.done = done
    st.session_state.info = info
    st.session_state.log.append(
        {
            "turn": env.turns,
            "player": p,
            "action": action,
            "reward0": rewards[0],
            "reward1": rewards[1],
            "legal_actions": legal_n,
        }
    )
    st.session_state.history.append(([row[:] for row in env.board], env.turns, env.current_player))
    size = env.cfg.size
    block_n = sum(1 for row in env.board for v in row if v == 3)
    legal0 = len(env.legal_actions(0))
    legal1 = len(env.legal_actions(1))
    st.session_state.metric_rows.append(
        {
            "turn": env.turns,
            "reward0": rewards[0],
            "reward1": rewards[1],
            "cum_reward0": sum(x["reward0"] for x in st.session_state.log),
            "cum_reward1": sum(x["reward1"] for x in st.session_state.log),
            "legal0": legal0,
            "legal1": legal1,
            "mobility_gap": legal0 - legal1,
            "block_ratio": block_n / float(size * size),
        }
    )


def _start_bg_task(task_name: str, cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    st.session_state.bg_task = {
        "name": task_name,
        "cmd": cmd,
        "pid": proc.pid,
        "proc": proc,
        "log_path": str(log_path),
        "started_at": time.time(),
    }
    if "task_history" not in st.session_state:
        st.session_state.task_history = []
    st.session_state.task_history.insert(
        0,
        {
            "name": task_name,
            "pid": proc.pid,
            "log_path": str(log_path),
            "started_at": int(time.time()),
        },
    )


def _render_bg_task_status() -> None:
    task = st.session_state.get("bg_task")
    if not task:
        st.caption("当前没有后台任务")
        return
    proc = task.get("proc")
    status = "running" if proc and proc.poll() is None else f"finished(code={proc.poll() if proc else 'n/a'})"
    st.write(f"任务: `{task['name']}`")
    st.write(f"PID: `{task['pid']}` | 状态: `{status}`")
    st.write(f"日志: `{task['log_path']}`")
    colx, coly = st.columns(2)
    with colx:
        if st.button("刷新任务状态"):
            st.rerun()
    with coly:
        if proc and proc.poll() is None and st.button("停止后台任务"):
            proc.terminate()
            st.warning("已发送终止信号")
            st.rerun()
    hist = st.session_state.get("task_history", [])
    if hist:
        st.caption("最近任务")
        for h in hist[:5]:
            st.text(f"{h['name']} | pid={h['pid']} | log={h['log_path']}")


def main():
    st.set_page_config(page_title="Amazons AI Demo", layout="wide")
    st.title("Mini Amazons 对弈演示")

    ensure_state()

    with st.sidebar:
        st.header("设置")
        device_default = "cuda" if torch.cuda.is_available() else "cpu"
        device = st.selectbox("推理设备", ["cuda", "cpu"], index=0 if device_default == "cuda" else 1)
        az_sims = st.slider("AZ MCTS simulations", min_value=20, max_value=200, value=80, step=20)
        page = st.radio("页面", ["对弈", "日志与回放", "训练监控", "一键实验"], index=0)

        board_size = st.selectbox("棋盘大小", [6, 8, 10, 12, 16, 32, 64], index=[6, 8, 10, 12, 16, 32, 64].index(st.session_state.board_size))
        if board_size != st.session_state.board_size:
            rebuild_env(int(board_size))
            st.info(f"棋盘已切换到 {board_size}x{board_size}。提示：尺寸越大越慢。")
            st.rerun()

        agent_options = [
            "random",
            "heuristic",
            "minimax_d1",
            "minimax_d2",
            "mcts_80",
            "q",
            "dqn",
            "a2c",
            "ppo",
            "bc",
            "az",
        ]
        rank_keys = ["rnd", "heu", "mm", "q0", "dqnL", "a2cL", "ppoL", "bcL", "az1", "az2", "az3"]
        use_preset = st.checkbox("使用排行榜预设（推荐）", value=True)
        a0_rank = st.selectbox("Agent0 预设", rank_keys, index=rank_keys.index("dqnL"))
        a1_rank = st.selectbox("Agent1 预设", rank_keys, index=rank_keys.index("mm"))

        a0_type = st.selectbox("Agent0 自定义类型", agent_options, index=6, disabled=use_preset)
        a1_type = st.selectbox("Agent1 自定义类型", agent_options, index=2, disabled=use_preset)
        non6_safe_rank = {"rnd", "heu", "mm"}
        non6_safe_types = {"random", "heuristic", "minimax_d1", "minimax_d2", "mcts_80"}
        selected_requires_6x6 = (
            (a0_rank not in non6_safe_rank or a1_rank not in non6_safe_rank)
            if use_preset
            else (a0_type not in non6_safe_types or a1_type not in non6_safe_types)
        )
        if selected_requires_6x6 and st.session_state.board_size != 6:
            st.warning("当前选择包含学习型智能体（Q/DQN/A2C/PPO/BC/AZ），仅支持 6x6。")
            if st.button("切回 6x6（推荐）"):
                rebuild_env(6)
                st.rerun()

        model_choices = list_model_paths()
        strength_map, strength_src = load_strength_table()
        default_model = "results/models/agent0_dqn.pt"
        default_model_b = "results/models/az_stage2_20260331/alphazero_pvnet.pt"
        no_model_types = {"random", "heuristic", "minimax_d1", "minimax_d2", "mcts_80"}
        a0_candidates = filtered_model_choices(a0_type, model_choices)
        a1_candidates = filtered_model_choices(a1_type, model_choices)

        if a0_type in no_model_types:
            a0_model = "(none)"
            st.selectbox("Agent0 model path", ["(none)"], index=0, disabled=True)
        else:
            a0_idx = 0
            if default_model in a0_candidates:
                a0_idx = a0_candidates.index(default_model)
            a0_model = st.selectbox(
                "Agent0 model path",
                a0_candidates if a0_candidates else ["(none)"],
                index=a0_idx if a0_candidates else 0,
                disabled=use_preset or (not a0_candidates),
            )

        if a1_type in no_model_types:
            a1_model = "(none)"
            st.selectbox("Agent1 model path", ["(none)"], index=0, disabled=True)
        else:
            a1_idx = 0
            if default_model_b in a1_candidates:
                a1_idx = a1_candidates.index(default_model_b)
            a1_model = st.selectbox(
                "Agent1 model path",
                a1_candidates if a1_candidates else ["(none)"],
                index=a1_idx if a1_candidates else 0,
                disabled=use_preset or (not a1_candidates),
            )
        a0_model = "" if a0_model == "(none)" else a0_model
        a1_model = "" if a1_model == "(none)" else a1_model

        build_btn = st.button("加载智能体")
        if build_btn:
            try:
                if selected_requires_6x6 and st.session_state.board_size != 6:
                    rebuild_env(6)
                    st.info("已自动切换到 6x6（当前智能体组合仅支持 6x6）。")
                if use_preset:
                    st.session_state.agent0 = build_from_rank_key(a0_rank, device, az_sims)
                    st.session_state.agent1 = build_from_rank_key(a1_rank, device, az_sims)
                    st.session_state.agent0_key = a0_rank
                    st.session_state.agent1_key = a1_rank
                else:
                    if a0_type not in no_model_types and not a0_model:
                        raise ValueError(f"Agent0 类型 {a0_type} 当前没有可匹配模型文件")
                    if a1_type not in no_model_types and not a1_model:
                        raise ValueError(f"Agent1 类型 {a1_type} 当前没有可匹配模型文件")
                    st.session_state.agent0 = build_agent(a0_type, a0_model, device, az_sims)
                    st.session_state.agent1 = build_agent(a1_type, a1_model, device, az_sims)
                    st.session_state.agent0_key = infer_agent_key(a0_type, a0_model)
                    st.session_state.agent1_key = infer_agent_key(a1_type, a1_model)
                st.success("智能体加载成功")
            except Exception as e:
                st.error(
                    "加载失败。可能是“类型与模型不匹配”（例如用 DQN 模型去加载 AZ/BC）。\n"
                    f"详细错误: {e}\n"
                    "建议：勾选“使用排行榜预设（推荐）”。"
                )

        st.markdown("---")
        st.subheader("模型强度参考")
        if strength_map:
            rank_df = (
                pd.DataFrame([{"agent": k, "score": v} for k, v in strength_map.items()])
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )
            rank_df.index = rank_df.index + 1
            st.caption(f"来源: {strength_src}（row mean win_rate）")
            st.dataframe(rank_df, use_container_width=True, height=250)

            k0 = st.session_state.get("agent0_key", a0_rank if use_preset else infer_agent_key(a0_type, a0_model))
            k1 = st.session_state.get("agent1_key", a1_rank if use_preset else infer_agent_key(a1_type, a1_model))
            s0 = strength_map.get(k0)
            s1 = strength_map.get(k1)
            st.write(f"A(agent0) 估计强度: `{k0}` -> `{s0:.3f}`" if s0 is not None else f"A(agent0) 估计强度: `{k0}` -> 无数据")
            st.write(f"B(agent1) 估计强度: `{k1}` -> `{s1:.3f}`" if s1 is not None else f"B(agent1) 估计强度: `{k1}` -> 无数据")
        else:
            st.caption("未找到强度数据文件（需要 results/eval_matrix_all_az_n40.csv 或 n30 矩阵）")

        if st.button("新开一局"):
            reset_game()

    if "agent0" not in st.session_state or "agent1" not in st.session_state:
        st.info("请先在左侧点击“加载智能体”。")
        return

    if page == "对弈":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("棋盘")
            st.markdown(render_board_html(st.session_state.env.board), unsafe_allow_html=True)
            st.write(f"turns: {st.session_state.env.turns}")
            st.write(f"current_player: {st.session_state.env.current_player}")
            st.write(f"done: {st.session_state.done}")
            if st.session_state.done:
                w = st.session_state.info.get("winner", -1)
                if w == -1:
                    st.warning("平局")
                else:
                    st.success(f"赢家: {'A(agent0)' if w == 0 else 'B(agent1)'}")

        with col2:
            st.subheader("控制")
            if st.button("走一步"):
                step_once(st.session_state.agent0, st.session_state.agent1)
                st.rerun()

            auto_steps = st.number_input("自动走步数（可见速度）", min_value=1, max_value=1000, value=20, step=1)
            step_delay = st.slider("每步间隔秒数", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            if st.button("按设定速度自动对弈"):
                st.session_state.auto_remaining = int(auto_steps)
                st.session_state.auto_delay = float(step_delay)
                st.session_state.auto_play = True
                st.rerun()

            if st.button("一步到位（直到终局）"):
                while not st.session_state.done:
                    step_once(st.session_state.agent0, st.session_state.agent1)
                st.rerun()

            if st.button("停止自动"):
                st.session_state.auto_play = False
                st.session_state.auto_remaining = 0
                st.rerun()
    elif page == "日志与回放":
        st.subheader("日志与回放")
        history_len = len(st.session_state.history)
        st.write(f"总步数记录: {max(0, history_len - 1)}")
        if history_len == 0:
            st.info("当前没有可回放数据，请先开局或走一步。")
        elif history_len == 1:
            board, turn, cp = st.session_state.history[0]
            st.markdown(render_board_html(board), unsafe_allow_html=True)
            st.write(f"回放 turn={turn}, current_player={cp}")
        else:
            max_idx = history_len - 1
            idx = st.slider("回放步号", min_value=0, max_value=max_idx, value=max_idx, step=1)
            board, turn, cp = st.session_state.history[idx]
            st.markdown(render_board_html(board), unsafe_allow_html=True)
            st.write(f"回放 turn={turn}, current_player={cp}")
        st.markdown("---")
        st.subheader("最近20条动作日志")
        for row in st.session_state.log[-20:]:
            st.text(str(row))
        if st.session_state.metric_rows:
            mdf = pd.DataFrame(st.session_state.metric_rows)
            st.markdown("---")
            st.subheader("对战曲线")
            st.line_chart(mdf.set_index("turn")[["cum_reward0", "cum_reward1"]], height=220)
            st.line_chart(mdf.set_index("turn")[["legal0", "legal1"]], height=220)
            st.line_chart(mdf.set_index("turn")[["mobility_gap", "block_ratio"]], height=220)

        st.markdown("---")
        st.subheader("案例追溯")
        case_name = st.text_input("案例文件名", value="latest_case.json")
        if st.button("保存案例到 results/cases"):
            out = ROOT / "results" / "cases" / case_name
            case_id = save_case_record(
                str(out),
                metadata={
                    "board_size": st.session_state.board_size,
                    "winner": st.session_state.info.get("winner", -1),
                    "total_turns": st.session_state.env.turns,
                },
                steps=st.session_state.log,
            )
            st.success(f"案例已保存，case_id={case_id}")
    elif page == "训练监控":
        st.subheader("训练监控")
        logs = list_training_logs()
        if not logs:
            st.info("未发现训练日志，请先执行训练任务生成 *log*.csv 文件。")
        else:
            selected = st.selectbox("训练日志文件", logs, index=0)
            path = ROOT / selected
            try:
                df = pd.read_csv(path)
                st.caption(f"数据源: {selected}")
                st.dataframe(df.tail(20), use_container_width=True, height=220)
                if "episode" in df.columns:
                    x = df.set_index("episode")
                    if {"reward_0", "reward_1"}.issubset(df.columns):
                        chart_df = pd.DataFrame(
                            {
                                "reward0_ma20": moving_average(x["reward_0"], 20),
                                "reward1_ma20": moving_average(x["reward_1"], 20),
                            }
                        )
                        st.line_chart(chart_df, height=220)
                    if "winner" in df.columns:
                        win = (df["winner"] == 0).astype(float)
                        st.line_chart(pd.DataFrame({"win_rate0_ma50": moving_average(win, 50)}), height=220)
                    eps_cols = [c for c in ["epsilon_0", "epsilon_1"] if c in df.columns]
                    if eps_cols:
                        st.line_chart(x[eps_cols], height=220)
                    loss_cols = [c for c in ["loss_0", "loss_1"] if c in df.columns]
                    if loss_cols:
                        st.line_chart(x[loss_cols], height=220)
                    ent_cols = [c for c in ["entropy_0", "entropy_1"] if c in df.columns]
                    if ent_cols:
                        st.line_chart(x[ent_cols], height=220)
                    if "turns" in df.columns:
                        st.line_chart(pd.DataFrame({"turns_ma20": moving_average(x["turns"], 20)}), height=220)
            except Exception as e:
                st.error(f"读取训练日志失败: {e}")
        st.markdown("---")
        st.subheader("图像结果浏览")
        figures = list_figure_paths()
        if not figures:
            st.info("未发现图像，请先运行绘图任务。")
        else:
            keyword = st.text_input("按关键词过滤（文件名）", value="")
            fig_items: list[tuple[str, float]] = []
            for rel in figures:
                if keyword and keyword.lower() not in rel.lower():
                    continue
                p = ROOT / rel
                try:
                    fig_items.append((rel, p.stat().st_mtime))
                except OSError:
                    fig_items.append((rel, 0.0))
            fig_items.sort(key=lambda x: x[1], reverse=True)
            filtered_figs = [x[0] for x in fig_items]
            if not filtered_figs:
                st.warning("没有匹配该关键词的图像。")
            else:
                default_figs = filtered_figs[: min(6, len(filtered_figs))]
                selected_figs = st.multiselect("选择要显示的图", filtered_figs, default=default_figs)
                col_n = st.slider("每行显示列数", min_value=1, max_value=4, value=2, step=1)
                if selected_figs:
                    cols = st.columns(col_n)
                    for i, rel in enumerate(selected_figs):
                        p = ROOT / rel
                        with cols[i % col_n]:
                            st.image(str(p), caption=rel, use_container_width=True)
            if st.button("刷新图像列表"):
                st.rerun()
    else:
        st.subheader("一键实验")
        st.caption("所有任务都在后台执行，完成后可在训练监控和 results 目录查看产物。")
        _render_bg_task_status()

        st.markdown("---")
        st.write("**1) 一键创新训练（DQN）**")
        dqn_episodes = st.number_input("训练局数", min_value=50, max_value=5000, value=600, step=50)
        if st.button("启动创新DQN训练"):
            cmd = [
                sys.executable,
                "-m",
                "src.main",
                "train-dqn",
                "--episodes",
                str(int(dqn_episodes)),
                "--size",
                "6",
                "--max-turns",
                "200",
                "--device",
                "cuda" if torch.cuda.is_available() else "cpu",
                "--use-per",
                "--n-step",
                "3",
                "--reward-mobility-weight",
                "0.05",
                "--reward-center-weight",
                "0.02",
                "--prune-top-k",
                "24",
                "--prune-keep-ratio",
                "0.5",
                "--model-dir",
                "results/models/dqn_ui_run",
                "--log-csv",
                "results/train_dqn_ui_run.csv",
                "--metadata-json",
                "results/runs/train_dqn_ui_run_meta.json",
            ]
            _start_bg_task("dqn_train_ui", cmd, ROOT / "results" / "runs" / "dqn_train_ui.log")
            st.success("已启动后台训练")
            st.rerun()

        st.markdown("---")
        st.write("**2) 一键 Arena 对战**")
        arena_games = st.number_input("Arena 对战局数", min_value=20, max_value=50000, value=300, step=20)
        if st.button("启动 Arena 对战（rnd/heu/mm/dqn）"):
            cmd = [
                sys.executable,
                "-m",
                "src.main",
                "run-arena",
                "--agent",
                "rnd:random",
                "--agent",
                "heu:heuristic",
                "--agent",
                "mm:minimax:1",
                "--agent",
                "dqn:dqn:results/models/dqn_innovation_run/agent0_dqn.pt",
                "--games",
                str(int(arena_games)),
                "--mode",
                "pool",
                "--size",
                "6",
                "--max-turns",
                "200",
                "--device",
                "cuda" if torch.cuda.is_available() else "cpu",
                "--out-games-csv",
                "results/arena_games_ui.csv",
                "--out-summary-json",
                "results/arena_summary_ui.json",
            ]
            _start_bg_task("arena_ui", cmd, ROOT / "results" / "runs" / "arena_ui.log")
            st.success("已启动后台 Arena")
            st.rerun()

        st.markdown("---")
        st.write("**3) 一键生成 Arena 分析图**")
        fig_prefix = st.text_input("图名前缀", value="arena_ui")
        if st.button("生成 Arena 图（柱状图/箱线图/滚动胜率）"):
            cmd = [
                sys.executable,
                "scripts/plot_arena_results.py",
                "--games-csv",
                "results/arena_games_ui.csv",
                "--summary-json",
                "results/arena_summary_ui.json",
                "--out-dir",
                "results/figures",
                "--prefix",
                fig_prefix,
                "--rolling-window",
                "30",
            ]
            try:
                out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True)
                st.success("图表生成完成")
                st.code(out.stdout.strip() or "ok")
            except subprocess.CalledProcessError as e:
                st.error("图表生成失败")
                st.code((e.stdout or "") + "\n" + (e.stderr or ""))
        st.markdown("---")
        st.write("**4) 单按钮全流程（训练 -> Arena -> 自动出图）**")
        p_episodes = st.number_input("全流程训练局数", min_value=100, max_value=5000, value=400, step=50)
        p_games = st.number_input("全流程 Arena 局数", min_value=50, max_value=50000, value=200, step=50)
        p_prefix = st.text_input("全流程图名前缀", value="arena_pipeline")
        if st.button("启动全流程"):
            cmd = [
                sys.executable,
                "scripts/run_full_pipeline.py",
                "--episodes",
                str(int(p_episodes)),
                "--arena-games",
                str(int(p_games)),
                "--device",
                "cuda" if torch.cuda.is_available() else "cpu",
                "--prefix",
                p_prefix,
            ]
            _start_bg_task("full_pipeline_ui", cmd, ROOT / "results" / "runs" / "full_pipeline_ui.log")
            st.success("已启动全流程后台任务")
            st.rerun()

        st.markdown("---")
        st.write("**结果快捷入口**")
        st.code(
            "\n".join(
                [
                    "results/train_dqn_ui_run.csv",
                    "results/arena_games_ui.csv",
                    "results/arena_summary_ui.json",
                    f"results/figures/{fig_prefix}_winrate_bars.png",
                    f"results/figures/{fig_prefix}_turns_boxplot.png",
                    f"results/figures/{fig_prefix}_rolling_winrate.png",
                ]
            )
        )

    if (
        st.session_state.auto_play
        and not st.session_state.done
        and st.session_state.auto_remaining > 0
    ):
        step_once(st.session_state.agent0, st.session_state.agent1)
        st.session_state.auto_remaining -= 1
        time.sleep(max(0.05, float(st.session_state.auto_delay)))
        if st.session_state.auto_remaining <= 0:
            st.session_state.auto_play = False
        st.rerun()


if __name__ == "__main__":
    main()
