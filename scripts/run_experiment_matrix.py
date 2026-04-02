from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List

# Make "python scripts/xxx.py" work from repo root.
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
from src.agents.q_learning_agent import TabularQLearningAmazonsAgent
from src.agents.random_agent import RandomAmazonsAgent
from src.evaluation.evaluate import evaluate_agents


@dataclass
class AgentSpec:
    name: str
    kind: str
    model: str | None = None
    depth: int = 2
    device: str | None = None


def build_agent(spec: AgentSpec):
    k = spec.kind.lower()
    if k == "random":
        return RandomAmazonsAgent()
    if k == "heuristic":
        return HeuristicAmazonsAgent()
    if k == "minimax":
        return MinimaxAmazonsAgent(depth=spec.depth)
    if k == "q":
        if not spec.model:
            raise ValueError(f"{spec.name}: kind=q requires --model")
        return TabularQLearningAmazonsAgent.load(spec.model)
    if k == "dqn":
        if not spec.model:
            raise ValueError(f"{spec.name}: kind=dqn requires --model")
        return DQNAmazonsAgent.load(spec.model, device=spec.device)
    if k == "a2c":
        if not spec.model:
            raise ValueError(f"{spec.name}: kind=a2c requires --model")
        return A2CAmazonsAgent.load(spec.model, device=spec.device)
    if k == "ppo":
        from src.agents.ppo_agent import PPOAmazonsAgent

        if not spec.model:
            raise ValueError(f"{spec.name}: kind=ppo requires --model")
        return PPOAmazonsAgent.load(spec.model, device=spec.device)
    if k == "bc":
        if not spec.model:
            raise ValueError(f"{spec.name}: kind=bc requires --model")
        net = load_checkpoint(spec.model, device=spec.device)
        return BCPolicyPlayer(size=net.size, net=net, device=spec.device)
    if k == "az":
        if not spec.model:
            raise ValueError(f"{spec.name}: kind=az requires --model")
        net = load_checkpoint(spec.model, device=spec.device)
        mcts = AlphaZeroMCTS(
            size=net.size,
            net=net,
            cfg=PUCTConfig(simulations=80, seed=0),
            device=spec.device,
        )
        return AlphaZeroPlayer(size=net.size, net=net, mcts=mcts)
    raise ValueError(f"Unknown kind: {spec.kind}")


def parse_spec(raw: str, device: str | None) -> AgentSpec:
    """
    Format:
      name:kind
      name:kind:model_path
      name:minimax:depth
    """
    parts = raw.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid --agent spec: {raw}")

    name = parts[0]
    kind = parts[1]

    if kind.lower() == "minimax":
        depth = int(parts[2]) if len(parts) >= 3 else 2
        return AgentSpec(name=name, kind=kind, depth=depth, device=device)

    model = parts[2] if len(parts) >= 3 and parts[2] else None
    return AgentSpec(name=name, kind=kind, model=model, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pairwise evaluation matrix for Amazons agents")
    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        help="Agent spec: name:kind[:model_or_depth]. kind in random|heuristic|minimax|q|dqn|a2c|ppo|bc|az",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per matchup")
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--device", type=str, default=None, help="For dqn/a2c load: cuda or cpu")
    parser.add_argument("--out-csv", type=str, default="results/eval_matrix.csv")
    args = parser.parse_args()

    specs: List[AgentSpec] = [parse_spec(s, device=args.device) for s in args.agent]
    agent_objs = {s.name: build_agent(s) for s in specs}

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(len(specs)):
        for j in range(len(specs)):
            if i == j:
                continue
            s0 = specs[i]
            s1 = specs[j]
            result = evaluate_agents(
                agent_objs[s0.name],
                agent_objs[s1.name],
                episodes=args.episodes,
                size=args.size,
                max_turns=args.max_turns,
            )
            row = {
                "agent0": s0.name,
                "agent1": s1.name,
                "win_rate_agent0": result["win_rate_agent0"],
                "win_rate_agent1": result["win_rate_agent1"],
                "draw_rate": result["draw_rate"],
                "episodes": args.episodes,
            }
            rows.append(row)
            print(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["agent0", "agent1", "win_rate_agent0", "win_rate_agent1", "draw_rate", "episodes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved matrix CSV: {out_path}")


if __name__ == "__main__":
    main()

