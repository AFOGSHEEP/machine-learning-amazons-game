from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

from src.agents.q_learning_agent import TabularQLearningAmazonsAgent
from src.envs.amazons_env import MiniAmazonsEnv, AmazonsConfig


@dataclass
class TrainConfig:
    episodes: int = 5000
    seed: int = 42
    model_dir: str = "results/models"
    log_csv: str = "results/train_log.csv"


def train_selfplay(config: TrainConfig | None = None):
    cfg = config or TrainConfig()

    env = MiniAmazonsEnv(AmazonsConfig(size=6, max_turns=200))
    a0 = TabularQLearningAmazonsAgent()
    a1 = TabularQLearningAmazonsAgent()

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for ep in range(1, cfg.episodes + 1):
        state = env.reset()
        done = False

        total_r0 = 0.0
        total_r1 = 0.0
        info = {"winner": -1}

        while not done:
            player = env.current_player
            legal = env.legal_actions(player)

            if player == 0:
                action = a0.select_action(state, legal, training=True)
            else:
                action = a1.select_action(state, legal, training=True)

            next_state, rewards, done, info = env.step(action)
            next_player = env.current_player
            next_legal = env.legal_actions(next_player) if not done else []

            if player == 0:
                a0.update(state, action, rewards[0], next_state, next_legal, done)
            else:
                a1.update(state, action, rewards[1], next_state, next_legal, done)

            total_r0 += rewards[0]
            total_r1 += rewards[1]
            state = next_state

        a0.decay_epsilon()
        a1.decay_epsilon()

        rows.append(
            {
                "episode": ep,
                "winner": info["winner"],
                "reward_0": round(total_r0, 4),
                "reward_1": round(total_r1, 4),
                "epsilon_0": round(a0.cfg.epsilon, 4),
                "epsilon_1": round(a1.cfg.epsilon, 4),
            }
        )

    m0 = str(Path(cfg.model_dir) / "agent0_q.json")
    m1 = str(Path(cfg.model_dir) / "agent1_q.json")
    a0.save(m0)
    a1.save(m1)

    with open(cfg.log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "winner", "reward_0", "reward_1", "epsilon_0", "epsilon_1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"model0": m0, "model1": m1, "log": cfg.log_csv}
