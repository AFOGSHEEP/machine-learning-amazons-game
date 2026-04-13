from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import random
import time
from typing import Any

from src.agents.a2c_agent import A2CAmazonsAgent
from src.agents.alphazero_mcts import AlphaZeroMCTS, PUCTConfig
from src.agents.alphazero_net import load_checkpoint
from src.agents.alphazero_player import AlphaZeroPlayer
from src.agents.bc_player import BCPolicyPlayer
from src.agents.dqn_agent import DQNAmazonsAgent
from src.agents.heuristic_agent import HeuristicAmazonsAgent
from src.agents.minimax_agent import MinimaxAmazonsAgent
from src.agents.ppo_agent import PPOAmazonsAgent
from src.agents.q_learning_agent import TabularQLearningAmazonsAgent
from src.agents.random_agent import RandomAmazonsAgent
from src.envs.amazons_env import AmazonsConfig, MiniAmazonsEnv


@dataclass
class ArenaAgentSpec:
    name: str
    kind: str
    model: str | None = None
    depth: int = 2
    az_sims: int = 80
    device: str | None = None


@dataclass
class ArenaConfig:
    games: int = 200
    size: int = 6
    max_turns: int = 200
    seed: int = 42
    mode: str = "pool"  # pool | fixed
    out_games_csv: str = "results/arena_games.csv"
    out_summary_json: str = "results/arena_summary.json"


def parse_agent_spec(raw: str, device: str | None = None, az_sims: int = 80) -> ArenaAgentSpec:
    """
    name:kind[:model_or_depth]
    kind: random|heuristic|minimax|mcts|q|dqn|a2c|ppo|bc|az
    """
    parts = raw.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid --agent format: {raw}")
    name = parts[0].strip()
    kind = parts[1].strip().lower()
    if kind == "minimax":
        depth = int(parts[2]) if len(parts) >= 3 and parts[2].strip() else 2
        return ArenaAgentSpec(name=name, kind=kind, depth=depth, az_sims=az_sims, device=device)
    model = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None
    return ArenaAgentSpec(name=name, kind=kind, model=model, depth=2, az_sims=az_sims, device=device)


def build_agent(spec: ArenaAgentSpec) -> Any:
    k = spec.kind
    if k == "random":
        return RandomAmazonsAgent()
    if k == "heuristic":
        return HeuristicAmazonsAgent()
    if k == "minimax":
        return MinimaxAmazonsAgent(depth=spec.depth)
    if k == "mcts":
        from src.agents.mcts_agent import MCTSAmazonsAgent, MCTSConfig

        return MCTSAmazonsAgent(config=MCTSConfig(simulations=120, rollout_depth=40))
    if k == "q":
        if not spec.model:
            raise ValueError(f"{spec.name}: q needs model path")
        return TabularQLearningAmazonsAgent.load(spec.model)
    if k == "dqn":
        if not spec.model:
            raise ValueError(f"{spec.name}: dqn needs model path")
        return DQNAmazonsAgent.load(spec.model, device=spec.device)
    if k == "a2c":
        if not spec.model:
            raise ValueError(f"{spec.name}: a2c needs model path")
        return A2CAmazonsAgent.load(spec.model, device=spec.device)
    if k == "ppo":
        if not spec.model:
            raise ValueError(f"{spec.name}: ppo needs model path")
        return PPOAmazonsAgent.load(spec.model, device=spec.device)
    if k == "bc":
        if not spec.model:
            raise ValueError(f"{spec.name}: bc needs model path")
        net = load_checkpoint(spec.model, device=spec.device)
        return BCPolicyPlayer(size=net.size, net=net, device=spec.device)
    if k == "az":
        if not spec.model:
            raise ValueError(f"{spec.name}: az needs model path")
        net = load_checkpoint(spec.model, device=spec.device)
        mcts = AlphaZeroMCTS(
            size=net.size,
            net=net,
            cfg=PUCTConfig(simulations=spec.az_sims, seed=0),
            device=spec.device,
        )
        return AlphaZeroPlayer(size=net.size, net=net, mcts=mcts)
    raise ValueError(f"Unsupported kind: {k}")


def _play_single_game(
    env: MiniAmazonsEnv,
    agent0: Any,
    agent1: Any,
) -> tuple[int, int]:
    state = env.reset()
    done = False
    info = {"winner": -1}
    while not done:
        p = env.current_player
        legal = env.legal_actions(p)
        if p == 0:
            action = agent0.select_action(state, legal, training=False)
        else:
            action = agent1.select_action(state, legal, training=False)
        state, _, done, info = env.step(action)
    return int(info.get("winner", -1)), int(env.turns)


def run_arena(specs: list[ArenaAgentSpec], cfg: ArenaConfig) -> dict[str, Any]:
    if len(specs) < 2:
        raise ValueError("Arena needs at least two agents")
    random.seed(cfg.seed)
    env = MiniAmazonsEnv(AmazonsConfig(size=cfg.size, max_turns=cfg.max_turns))

    agent_objs = {s.name: build_agent(s) for s in specs}
    names = [s.name for s in specs]
    stats = {n: {"wins": 0, "losses": 0, "draws": 0, "games": 0, "turns": 0} for n in names}
    rows: list[dict[str, Any]] = []

    start = time.perf_counter()
    for game_id in range(1, cfg.games + 1):
        if cfg.mode == "fixed":
            i = (game_id - 1) % len(names)
            j = (i + 1) % len(names)
            a_name = names[i]
            b_name = names[j]
        else:
            a_name, b_name = random.sample(names, k=2)

        # Swap colors every other game to reduce first-move bias.
        if game_id % 2 == 1:
            white_name, black_name = a_name, b_name
        else:
            white_name, black_name = b_name, a_name

        winner, turns = _play_single_game(env, agent_objs[white_name], agent_objs[black_name])
        if winner == 0:
            winner_name = white_name
            loser_name = black_name
        elif winner == 1:
            winner_name = black_name
            loser_name = white_name
        else:
            winner_name = "draw"
            loser_name = "draw"

        stats[white_name]["games"] += 1
        stats[black_name]["games"] += 1
        stats[white_name]["turns"] += turns
        stats[black_name]["turns"] += turns
        if winner_name == "draw":
            stats[white_name]["draws"] += 1
            stats[black_name]["draws"] += 1
        else:
            stats[winner_name]["wins"] += 1
            stats[loser_name]["losses"] += 1

        rows.append(
            {
                "game_id": game_id,
                "white": white_name,
                "black": black_name,
                "winner": winner_name,
                "turns": turns,
            }
        )

    elapsed = time.perf_counter() - start
    out_games = Path(cfg.out_games_csv)
    out_games.parent.mkdir(parents=True, exist_ok=True)
    with out_games.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["game_id", "white", "black", "winner", "turns"])
        writer.writeheader()
        writer.writerows(rows)

    summary_agents: dict[str, Any] = {}
    for n, s in stats.items():
        games = max(1, s["games"])
        summary_agents[n] = {
            "games": s["games"],
            "wins": s["wins"],
            "losses": s["losses"],
            "draws": s["draws"],
            "win_rate": s["wins"] / games,
            "draw_rate": s["draws"] / games,
            "avg_turns": s["turns"] / games,
        }

    summary = {
        "config": {
            "games": cfg.games,
            "size": cfg.size,
            "max_turns": cfg.max_turns,
            "seed": cfg.seed,
            "mode": cfg.mode,
        },
        "agents": summary_agents,
        "outputs": {"games_csv": cfg.out_games_csv, "summary_json": cfg.out_summary_json},
        "elapsed_sec": elapsed,
    }

    out_summary = Path(cfg.out_summary_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
