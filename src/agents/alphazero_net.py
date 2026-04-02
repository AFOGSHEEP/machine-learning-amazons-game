from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.amazons_env import Action


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        d = torch.device(device)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available in current Python environment")
        return d
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_state_onehot(obs_batch: torch.Tensor, size: int) -> torch.Tensor:
    board_flat = obs_batch[:, : size * size].long()
    current_player = obs_batch[:, -1].float().unsqueeze(1)
    one_hot = F.one_hot(board_flat, num_classes=4).float().view(obs_batch.shape[0], -1)
    return torch.cat([one_hot, current_player], dim=1)


def encode_action(action_batch: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return action_batch.float()
    return (action_batch.float() / float(size - 1)).clamp(0.0, 1.0)


class AlphaZeroPVNet(nn.Module):
    """
    Action-conditional policy: logits(s,a) computed for each legal action.
    Value head: V(s) in [-1,1] via tanh.
    """

    def __init__(self, size: int = 6, hidden: int = 256) -> None:
        super().__init__()
        self.size = size
        state_dim = size * size * 4 + 1

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden + 6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def value(self, obs_batch: torch.Tensor) -> torch.Tensor:
        x = encode_state_onehot(obs_batch, self.size)
        h = self.trunk(x)
        return self.value_head(h).squeeze(-1)

    def logits_for_actions(self, obs_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        x = encode_state_onehot(obs_batch, self.size)
        h = self.trunk(x)
        a = encode_action(action_batch, self.size)
        ha = torch.cat([h, a], dim=1)
        return self.policy_head(ha).squeeze(-1)


@dataclass
class AlphaZeroCheckpoint:
    size: int
    state_dict: dict


def save_checkpoint(net: AlphaZeroPVNet, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"size": net.size, "state_dict": net.state_dict()}
    torch.save(payload, str(p))


def load_checkpoint(path: str, device: str | None = None) -> AlphaZeroPVNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net = AlphaZeroPVNet(size=int(payload["size"]))
    net.load_state_dict(payload["state_dict"])
    net.to(get_device(device))
    net.eval()
    return net

