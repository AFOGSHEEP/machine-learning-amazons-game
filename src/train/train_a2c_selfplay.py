from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import Dict

import numpy as np
import torch

from src.agents.a2c_agent import A2CAmazonsAgent, EpisodeStep
from src.envs.amazons_env import AmazonsConfig, MiniAmazonsEnv
from src.train.progress import TerminalProgressBar


@dataclass
class A2CTrainConfig:
    episodes: int = 2000
    seed: int = 42
    size: int = 6
    max_turns: int = 200
    model_dir: str = "results/models"
    log_csv: str = "results/train_a2c_log.csv"

    gamma: float = 0.98
    lr: float = 3e-4
    entropy_beta: float = 0.01
    value_coef: float = 0.5
    policy_epochs: int = 4
    device: str = "cuda"


def train_a2c_selfplay(config: A2CTrainConfig | None = None) -> Dict[str, str]:
    cfg = config or A2CTrainConfig()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = MiniAmazonsEnv(AmazonsConfig(size=cfg.size, max_turns=cfg.max_turns))
    a0 = A2CAmazonsAgent(
        size=cfg.size,
        gamma=cfg.gamma,
        lr=cfg.lr,
        entropy_beta=cfg.entropy_beta,
        value_coef=cfg.value_coef,
        device=cfg.device,
    )
    a1 = A2CAmazonsAgent(
        size=cfg.size,
        gamma=cfg.gamma,
        lr=cfg.lr,
        entropy_beta=cfg.entropy_beta,
        value_coef=cfg.value_coef,
        device=cfg.device,
    )

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pbar = TerminalProgressBar(total=cfg.episodes, title="A2C")

    for ep in range(1, cfg.episodes + 1):
        state = env.reset()
        done = False
        info = {"winner": -1}

        traj0: list[EpisodeStep] = []
        traj1: list[EpisodeStep] = []
        total_r0 = 0.0
        total_r1 = 0.0

        while not done:
            p = env.current_player
            legal = env.legal_actions(p)

            if p == 0:
                action = a0.select_action(state, legal, training=True)
            else:
                action = a1.select_action(state, legal, training=True)

            next_state, rewards, done, info = env.step(action)

            if p == 0:
                traj0.append(EpisodeStep(state=state, action=action, reward=rewards[0], done=done))
                total_r0 += rewards[0]
            else:
                traj1.append(EpisodeStep(state=state, action=action, reward=rewards[1], done=done))
                total_r1 += rewards[1]

            state = next_state

        # Train each agent on its own trajectory.
        stats0 = a0.train_on_episode(traj0, epochs=cfg.policy_epochs)
        stats1 = a1.train_on_episode(traj1, epochs=cfg.policy_epochs)

        rows.append(
            {
                "episode": ep,
                "winner": info["winner"],
                "reward_0": round(total_r0, 4),
                "reward_1": round(total_r1, 4),
                "loss_0": round(stats0["loss"], 6),
                "loss_1": round(stats1["loss"], 6),
                "entropy_0": round(stats0["entropy"], 6),
                "entropy_1": round(stats1["entropy"], 6),
            }
        )
        pbar.update(ep, extra=f"loss0={stats0['loss']:.4f} winner={info['winner']}")

    pbar.close()

    m0 = str(Path(cfg.model_dir) / "agent0_a2c.pt")
    m1 = str(Path(cfg.model_dir) / "agent1_a2c.pt")
    a0.save(m0)
    a1.save(m1)

    with open(cfg.log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "winner", "reward_0", "reward_1", "loss_0", "loss_1", "entropy_0", "entropy_1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"model0": m0, "model1": m1, "log": cfg.log_csv}

