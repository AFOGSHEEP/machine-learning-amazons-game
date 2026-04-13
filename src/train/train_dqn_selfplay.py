from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import Dict

import numpy as np
import torch

from src.agents.dqn_agent import DQNAmazonsAgent
from src.envs.amazons_env import AmazonsConfig, MiniAmazonsEnv
from src.train.progress import TerminalProgressBar


@dataclass
class DQNTrainConfig:
    episodes: int = 2000
    seed: int = 42
    size: int = 6
    max_turns: int = 200
    model_dir: str = "results/models"
    log_csv: str = "results/train_dqn_log.csv"

    gamma: float = 0.98
    lr: float = 1e-3
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    replay_capacity: int = 50_000
    batch_size: int = 64
    start_learning: int = 2_000
    target_update_interval: int = 250
    device: str = "cuda"
    # Rainbow-style options (PER + n-step); defaults match classic DQN
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 100_000
    n_step: int = 1
    prune_top_k: int = 0
    prune_keep_ratio: float = 1.0
    reward_mobility_weight: float = 0.0
    reward_center_weight: float = 0.0
    metadata_json: str = "results/runs/train_dqn_meta.json"


def train_dqn_selfplay(config: DQNTrainConfig | None = None) -> Dict[str, str]:
    cfg = config or DQNTrainConfig()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    from src.train.innovation_utils import prune_actions_by_score, save_run_metadata

    env = MiniAmazonsEnv(
        AmazonsConfig(
            size=cfg.size,
            max_turns=cfg.max_turns,
            reward_mobility_weight=cfg.reward_mobility_weight,
            reward_center_weight=cfg.reward_center_weight,
        )
    )
    a0 = DQNAmazonsAgent(
        size=cfg.size,
        gamma=cfg.gamma,
        lr=cfg.lr,
        epsilon=cfg.epsilon,
        epsilon_min=cfg.epsilon_min,
        epsilon_decay=cfg.epsilon_decay,
        replay_capacity=cfg.replay_capacity,
        batch_size=cfg.batch_size,
        start_learning=cfg.start_learning,
        target_update_interval=cfg.target_update_interval,
        device=cfg.device,
        use_per=cfg.use_per,
        per_alpha=cfg.per_alpha,
        per_beta_start=cfg.per_beta_start,
        per_beta_end=cfg.per_beta_end,
        per_beta_anneal_steps=cfg.per_beta_anneal_steps,
        n_step=cfg.n_step,
    )
    a1 = DQNAmazonsAgent(
        size=cfg.size,
        gamma=cfg.gamma,
        lr=cfg.lr,
        epsilon=cfg.epsilon,
        epsilon_min=cfg.epsilon_min,
        epsilon_decay=cfg.epsilon_decay,
        replay_capacity=cfg.replay_capacity,
        batch_size=cfg.batch_size,
        start_learning=cfg.start_learning,
        target_update_interval=cfg.target_update_interval,
        device=cfg.device,
        use_per=cfg.use_per,
        per_alpha=cfg.per_alpha,
        per_beta_start=cfg.per_beta_start,
        per_beta_end=cfg.per_beta_end,
        per_beta_anneal_steps=cfg.per_beta_anneal_steps,
        n_step=cfg.n_step,
    )

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    wins0 = 0
    pbar = TerminalProgressBar(total=cfg.episodes, title="DQN")

    for ep in range(1, cfg.episodes + 1):
        state = env.reset()
        done = False
        total_r0 = 0.0
        total_r1 = 0.0
        info = {"winner": -1}

        while not done:
            p = env.current_player
            legal = env.legal_actions(p)
            legal = prune_actions_by_score(
                env,
                p,
                legal,
                top_k=cfg.prune_top_k,
                keep_ratio=cfg.prune_keep_ratio,
            )

            if p == 0:
                action = a0.select_action(state, legal, training=True)
            else:
                action = a1.select_action(state, legal, training=True)

            next_state, rewards, done, info = env.step(action)

            if p == 0:
                a0.remember(state, action, rewards[0], next_state, done)
                a0.learn()
                total_r0 += rewards[0]
            else:
                a1.remember(state, action, rewards[1], next_state, done)
                a1.learn()
                total_r1 += rewards[1]

            state = next_state

        a0.decay_epsilon()
        a1.decay_epsilon()

        if info["winner"] == 0:
            wins0 += 1
        rows.append(
            {
                "episode": ep,
                "winner": info["winner"],
                "reward_0": round(total_r0, 4),
                "reward_1": round(total_r1, 4),
                "epsilon_0": round(a0.epsilon, 6),
                "epsilon_1": round(a1.epsilon, 6),
                "turns": env.turns,
                "win_rate_0_running": round(wins0 / ep, 6),
            }
        )
        pbar.update(ep, extra=f"eps0={a0.epsilon:.3f} winner={info['winner']}")

    pbar.close()

    m0 = str(Path(cfg.model_dir) / "agent0_dqn.pt")
    m1 = str(Path(cfg.model_dir) / "agent1_dqn.pt")
    a0.save(m0)
    a1.save(m1)

    with open(cfg.log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "winner",
                "reward_0",
                "reward_1",
                "epsilon_0",
                "epsilon_1",
                "turns",
                "win_rate_0_running",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    run_id = save_run_metadata(
        cfg.metadata_json,
        run_type="train_dqn_selfplay",
        config=cfg,
        extra={"model0": m0, "model1": m1, "log_csv": cfg.log_csv},
    )

    return {"model0": m0, "model1": m1, "log": cfg.log_csv, "run_id": run_id, "meta": cfg.metadata_json}

