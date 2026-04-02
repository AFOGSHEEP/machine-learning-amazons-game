"""
Single-learner DQN vs a stochastic opponent pool (PFSP-style lite).

Pool: random / heuristic / shallow minimax / frozen snapshot of learner / greedy self.

References: Heinrich & Silver (Fictitious Self-Play); AlphaStar-style population play (simplified).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from src.agents.dqn_agent import DQNAmazonsAgent
from src.agents.heuristic_agent import HeuristicAmazonsAgent
from src.agents.minimax_agent import MinimaxAmazonsAgent
from src.agents.random_agent import RandomAmazonsAgent
from src.envs.amazons_env import AmazonsConfig, MiniAmazonsEnv
from src.train.progress import TerminalProgressBar

OpponentKind = Literal["random", "heuristic", "minimax", "frozen", "self"]


@dataclass
class OpponentPoolTrainConfig:
    episodes: int = 3000
    seed: int = 42
    size: int = 6
    max_turns: int = 200
    model_dir: str = "results/models/dqn_opponent_pool"
    log_csv: str = "results/train_dqn_opponent_pool.csv"
    device: str = "cuda"

    gamma: float = 0.98
    lr: float = 1e-3
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    replay_capacity: int = 100_000
    batch_size: int = 64
    start_learning: int = 3_000
    target_update_interval: int = 250
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 150_000
    n_step: int = 3

    w_random: float = 0.15
    w_heuristic: float = 0.15
    w_minimax: float = 0.15
    w_frozen: float = 0.35
    w_self: float = 0.2
    frozen_refresh_episodes: int = 120


def _sample_opponent_kind(cfg: OpponentPoolTrainConfig) -> OpponentKind:
    r = random.random()
    s = cfg.w_random + cfg.w_heuristic + cfg.w_minimax + cfg.w_frozen + cfg.w_self
    if s <= 0:
        return "random"
    r *= s
    c = 0.0
    for w, k in [
        (cfg.w_random, "random"),
        (cfg.w_heuristic, "heuristic"),
        (cfg.w_minimax, "minimax"),
        (cfg.w_frozen, "frozen"),
        (cfg.w_self, "self"),
    ]:
        c += w
        if r <= c:
            return k  # type: ignore[return-value]
    return "random"


def _opponent_action(
    kind: OpponentKind,
    learner: DQNAmazonsAgent,
    frozen: Optional[DQNAmazonsAgent],
    random_a: RandomAmazonsAgent,
    heu_a: HeuristicAmazonsAgent,
    mm_a: MinimaxAmazonsAgent,
    state: Tuple[int, ...],
    legal: List,
) -> object:
    if kind == "random":
        return random_a.select_action(state, legal, training=False)
    if kind == "heuristic":
        return heu_a.select_action(state, legal, training=False)
    if kind == "minimax":
        return mm_a.select_action(state, legal, training=False)
    if kind == "frozen":
        if frozen is None:
            return random_a.select_action(state, legal, training=False)
        return frozen.select_action(state, legal, training=False)
    return learner.select_action(state, legal, training=False)


def train_dqn_opponent_pool(config: OpponentPoolTrainConfig | None = None) -> Dict[str, str]:
    cfg = config or OpponentPoolTrainConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = MiniAmazonsEnv(AmazonsConfig(size=cfg.size, max_turns=cfg.max_turns))
    learner = DQNAmazonsAgent(
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

    rnd = RandomAmazonsAgent()
    heu = HeuristicAmazonsAgent()
    mm = MinimaxAmazonsAgent(depth=1)
    frozen: Optional[DQNAmazonsAgent] = None

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pbar = TerminalProgressBar(total=cfg.episodes, title="DQN-pool")

    for ep in range(1, cfg.episodes + 1):
        if ep % cfg.frozen_refresh_episodes == 0:
            frozen = learner.make_greedy_eval_clone()

        opp_kind = _sample_opponent_kind(cfg)
        learner_is_p0 = random.random() < 0.5

        state = env.reset()
        done = False
        total_r_learn = 0.0
        info = {"winner": -1}

        while not done:
            p = env.current_player
            legal = env.legal_actions(p)
            if not legal:
                break

            acting_learner = (p == 0 and learner_is_p0) or (p == 1 and not learner_is_p0)

            if acting_learner:
                action = learner.select_action(state, legal, training=True)
            else:
                action = _opponent_action(opp_kind, learner, frozen, rnd, heu, mm, state, legal)

            next_state, rewards, done, info = env.step(action)
            if acting_learner:
                learn_r = float(rewards[p])
                learner.remember(state, action, learn_r, next_state, done)
                learner.learn()
                total_r_learn += learn_r

            state = next_state

        learner.decay_epsilon()
        w = int(info.get("winner", -1))
        learn_won = (w == 0 and learner_is_p0) or (w == 1 and not learner_is_p0)

        rows.append(
            {
                "episode": ep,
                "opponent": opp_kind,
                "learner_won": int(learn_won),
                "winner": w,
                "reward_learner": round(total_r_learn, 4),
                "epsilon": round(learner.epsilon, 6),
            }
        )
        pbar.update(ep, extra=f"{opp_kind} eps={learner.epsilon:.3f} lw={int(learn_won)}")

    pbar.close()

    out_path = str(Path(cfg.model_dir) / "dqn_opponent_pool.pt")
    learner.save(out_path)

    with open(cfg.log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "opponent", "learner_won", "winner", "reward_learner", "epsilon"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"model": out_path, "log": cfg.log_csv}
