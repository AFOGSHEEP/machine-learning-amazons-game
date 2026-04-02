from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.alphazero_mcts import AlphaZeroMCTS, PUCTConfig
from src.agents.alphazero_net import AlphaZeroPVNet, get_device, save_checkpoint
from src.envs.amazons_env import AmazonsConfig, MiniAmazonsEnv, Action
from src.train.progress import TerminalProgressBar


@dataclass
class AZTrainConfig:
    episodes: int = 400
    seed: int = 42
    size: int = 6
    max_turns: int = 120
    device: str = "cuda"

    mcts_sims: int = 120
    c_puct: float = 1.5

    lr: float = 3e-4
    batch_size: int = 256
    train_steps_per_episode: int = 2
    save_every: int = 50

    model_dir: str = "results/models"
    log_csv: str = "results/train_az_log.csv"


def train_alphazero_selfplay(config: AZTrainConfig | None = None) -> Dict[str, str]:
    cfg = config or AZTrainConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = get_device(cfg.device)
    env = MiniAmazonsEnv(AmazonsConfig(size=cfg.size, max_turns=cfg.max_turns))

    net = AlphaZeroPVNet(size=cfg.size).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    puct_cfg = PUCTConfig(simulations=cfg.mcts_sims, c_puct=cfg.c_puct, seed=cfg.seed)
    mcts = AlphaZeroMCTS(size=cfg.size, net=net, cfg=puct_cfg, device=cfg.device)

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_csv).parent.mkdir(parents=True, exist_ok=True)

    # Replay-like dataset: (obs, legal_actions, pi, z)
    dataset: list[tuple[Tuple[int, ...], list[Action], list[float], float]] = []

    rows = []
    pbar = TerminalProgressBar(total=cfg.episodes, title="AlphaZero")
    for ep in range(1, cfg.episodes + 1):
        obs = env.reset()
        done = False
        info = {"winner": -1}
        traj: list[tuple[Tuple[int, ...], list[Action], list[float], int]] = []  # store player to move

        while not done:
            p = env.current_player
            legal = env.legal_actions(p)
            if not legal:
                # terminal: current player loses
                info["winner"] = 1 - p
                done = True
                break

            # Run MCTS to get improved policy pi over legal actions
            _, pi_dict, _ = mcts.run(obs)
            pi = [float(pi_dict[a]) for a in legal]

            # Sample action from pi (self-play exploration)
            a = random.choices(legal, weights=pi, k=1)[0]
            traj.append((obs, legal, pi, p))

            obs, _, done, info = env.step(a)

        winner = int(info["winner"])
        # Convert to training targets z from each step player's perspective (+1 win, -1 loss)
        for s_obs, s_legal, s_pi, s_player in traj:
            if winner == -1:
                z = 0.0
            else:
                z = 1.0 if winner == s_player else -1.0
            dataset.append((s_obs, s_legal, s_pi, z))

        # Train a few steps
        losses = []
        for _ in range(cfg.train_steps_per_episode):
            if len(dataset) < max(32, cfg.batch_size):
                continue
            batch = random.sample(dataset, k=cfg.batch_size)

            # We compute policy loss over each sample's legal actions (variable length).
            value_loss = torch.tensor(0.0, device=device)
            policy_loss = torch.tensor(0.0, device=device)

            for s_obs, s_legal, s_pi, z in batch:
                obs_t = torch.tensor(s_obs, dtype=torch.float32, device=device).unsqueeze(0)
                legal_t = torch.tensor(s_legal, dtype=torch.float32, device=device)
                logits = net.logits_for_actions(obs_t.expand(legal_t.shape[0], -1), legal_t)
                log_probs = F.log_softmax(logits, dim=0)
                pi_t = torch.tensor(s_pi, dtype=torch.float32, device=device)
                policy_loss = policy_loss + (-(pi_t * log_probs).sum())

                v = net.value(obs_t).squeeze(0)
                z_t = torch.tensor(float(z), dtype=torch.float32, device=device)
                value_loss = value_loss + F.mse_loss(v, z_t)

            policy_loss = policy_loss / cfg.batch_size
            value_loss = value_loss / cfg.batch_size
            loss = policy_loss + value_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.item()))

        rows.append(
            {
                "episode": ep,
                "winner": winner,
                "dataset_size": len(dataset),
                "loss": round(float(np.mean(losses)) if losses else 0.0, 6),
            }
        )
        pbar.update(ep, extra=f"dataset={len(dataset)} loss={rows[-1]['loss']}")

        if cfg.save_every > 0 and ep % cfg.save_every == 0:
            save_checkpoint(net, str(Path(cfg.model_dir) / f"alphazero_pvnet_ep{ep}.pt"))

    out_model = str(Path(cfg.model_dir) / "alphazero_pvnet.pt")
    save_checkpoint(net, out_model)
    pbar.close()

    with open(cfg.log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "winner", "dataset_size", "loss"])
        w.writeheader()
        w.writerows(rows)

    return {"model": out_model, "log": cfg.log_csv}

