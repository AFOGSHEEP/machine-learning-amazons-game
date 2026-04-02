from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.alphazero_net import AlphaZeroPVNet, get_device, save_checkpoint
from src.agents.mcts_agent import MCTSAmazonsAgent, MCTSConfig
from src.envs.amazons_env import AmazonsConfig, MiniAmazonsEnv, Action
from src.train.progress import TerminalProgressBar


@dataclass
class BCTrainConfig:
    episodes_generate: int = 200
    seed: int = 42
    size: int = 6
    max_turns: int = 120
    device: str = "cuda"

    mcts_sims: int = 120
    rollout_depth: int = 40

    lr: float = 3e-4
    batch_size: int = 256
    epochs: int = 5

    model_dir: str = "results/models"
    log_csv: str = "results/train_bc_log.csv"


def train_bc_from_mcts(config: BCTrainConfig | None = None) -> Dict[str, str]:
    cfg = config or BCTrainConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = get_device(cfg.device)
    env = MiniAmazonsEnv(AmazonsConfig(size=cfg.size, max_turns=cfg.max_turns))

    teacher = MCTSAmazonsAgent(
        size=cfg.size,
        config=MCTSConfig(simulations=cfg.mcts_sims, rollout_depth=cfg.rollout_depth, seed=cfg.seed),
    )

    # Student uses AlphaZeroPVNet policy head for logits(s,a); ignore value in BC.
    student = AlphaZeroPVNet(size=cfg.size).to(device)
    optim = torch.optim.Adam(student.parameters(), lr=cfg.lr)

    # Dataset: (obs, legal_actions, chosen_action_index)
    dataset: list[tuple[Tuple[int, ...], list[Action], int]] = []
    gen_bar = TerminalProgressBar(total=cfg.episodes_generate, title="BC-Generate")

    for ep in range(1, cfg.episodes_generate + 1):
        obs = env.reset()
        done = False
        while not done:
            p = env.current_player
            legal = env.legal_actions(p)
            if not legal:
                break
            a = teacher.select_action(obs, legal, training=False)
            idx = legal.index(a)
            dataset.append((obs, legal, idx))
            obs, _, done, _ = env.step(a)
        gen_bar.update(ep, extra=f"dataset={len(dataset)}")
    gen_bar.close()

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    train_bar = TerminalProgressBar(total=cfg.epochs, title="BC-Train")
    for ep in range(1, cfg.epochs + 1):
        if len(dataset) < cfg.batch_size:
            break
        random.shuffle(dataset)
        losses = []
        for i in range(0, len(dataset), cfg.batch_size):
            batch = dataset[i : i + cfg.batch_size]
            if len(batch) < cfg.batch_size:
                continue

            loss = torch.tensor(0.0, device=device)
            for obs, legal, idx in batch:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                legal_t = torch.tensor(legal, dtype=torch.float32, device=device)
                logits = student.logits_for_actions(obs_t.expand(legal_t.shape[0], -1), legal_t)
                target = torch.tensor(idx, dtype=torch.long, device=device)
                loss = loss + F.cross_entropy(logits.view(1, -1), target.view(1))

            loss = loss / cfg.batch_size
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.item()))

        rows.append({"epoch": ep, "loss": round(float(np.mean(losses)) if losses else 0.0, 6), "dataset_size": len(dataset)})
        train_bar.update(ep, extra=f"loss={rows[-1]['loss']}")
    train_bar.close()

    out_model = str(Path(cfg.model_dir) / "bc_policy_from_mcts.pt")
    save_checkpoint(student, out_model)

    with open(cfg.log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss", "dataset_size"])
        w.writeheader()
        w.writerows(rows)

    return {"model": out_model, "log": cfg.log_csv, "dataset_size": str(len(dataset))}

