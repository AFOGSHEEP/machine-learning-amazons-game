from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.amazons_env import Action
from src.envs.amazons_rules import legal_actions_from_board, obs_to_board_and_positions


def _get_device(device: str | None = None) -> torch.device:
    if device is not None:
        d = torch.device(device)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available in current Python environment")
        return d
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encode_state_onehot(obs_batch: torch.Tensor, size: int) -> torch.Tensor:
    board_flat = obs_batch[:, : size * size].long()
    current_player = obs_batch[:, -1].float().unsqueeze(1)
    one_hot = F.one_hot(board_flat, num_classes=4).float()
    one_hot = one_hot.view(obs_batch.shape[0], -1)
    return torch.cat([one_hot, current_player], dim=1)


def _encode_action(action_batch: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return action_batch.float()
    return (action_batch.float() / float(size - 1)).clamp(0.0, 1.0)


class PolicyValueNet(nn.Module):
    def __init__(self, size: int, hidden: int = 256) -> None:
        super().__init__()
        self.size = size
        state_dim = size * size * 4 + 1
        action_dim = 6

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # logit per (s,a)
        )

    def value(self, obs_batch: torch.Tensor) -> torch.Tensor:
        x_state = _encode_state_onehot(obs_batch, self.size)
        return self.value_net(x_state).squeeze(-1)  # [B]

    def logits_for_actions(self, obs_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        """
        obs_batch: [M, dim]
        action_batch: [M,6]
        returns logits: [M]
        """
        x_state = _encode_state_onehot(obs_batch, self.size)
        x_action = _encode_action(action_batch, self.size)
        x = torch.cat([x_state, x_action], dim=1)
        return self.policy_net(x).squeeze(-1)  # [M]


@dataclass
class EpisodeStep:
    state: Tuple[int, ...]
    action: Action
    reward: float
    done: bool


class A2CAmazonsAgent:
    """
    Simple A2C-style actor-critic:
    - For a given state, enumerate legal actions and compute logits for each (s,a).
    - Value network estimates V(s).
    - Train on trajectories collected during self-play.
    """

    def __init__(
        self,
        size: int = 6,
        gamma: float = 0.98,
        lr: float = 3e-4,
        entropy_beta: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: str | None = None,
    ) -> None:
        self.size = size
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.device = _get_device(device)
        self.net = PolicyValueNet(size=size).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _legal_actions_from_obs(self, obs: Tuple[int, ...]) -> List[Action]:
        board, positions, current_player = obs_to_board_and_positions(obs, size=self.size)
        return legal_actions_from_board(board, positions, current_player, self.size)

    def select_action(
        self,
        state: Tuple[int, ...],
        legal_actions: List[Action],
        training: bool = False,
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available")

        obs = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,dim]
        actions = torch.tensor(legal_actions, dtype=torch.float32, device=self.device)  # [M,6]

        with torch.no_grad():
            logits = self.net.logits_for_actions(obs.expand(actions.shape[0], -1), actions)  # [M]
            if training:
                probs = F.softmax(logits, dim=0)
                idx = torch.multinomial(probs, num_samples=1).item()
            else:
                idx = int(torch.argmax(logits).item())
        return legal_actions[idx]

    def collect_episode(self, env, agent, player_id: int):
        """
        Helper for self-play training.
        """
        raise NotImplementedError("Use train loop to collect trajectory.")

    def _log_prob_and_entropy(self, state: Tuple[int, ...], action: Action, legal_actions: List[Action]) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,dim]
        actions = torch.tensor(legal_actions, dtype=torch.float32, device=self.device)  # [M,6]
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,6]

        logits = self.net.logits_for_actions(obs.expand(actions.shape[0], -1), actions)  # [M]
        log_probs = F.log_softmax(logits, dim=0)

        # Find index of the chosen action in legal_actions.
        # (legal_actions generated deterministically by rules, so exact tuple match should work.)
        try:
            idx = legal_actions.index(action)
        except ValueError:
            # Fallback: compute by comparing tensors -> robust but slower
            idx = None
            for i, a in enumerate(legal_actions):
                if a == action:
                    idx = i
                    break
            if idx is None:
                raise RuntimeError("Chosen action not found among legal actions")

        log_prob = log_probs[idx]
        probs = F.softmax(logits, dim=0)
        entropy = -(probs * log_probs).sum()
        return log_prob, entropy

    def train_on_episode(self, episode: List[EpisodeStep], epochs: int = 4) -> dict:
        """
        episode contains steps collected when this agent was the current player.
        """
        if not episode:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Compute discounted returns for this agent's own timeline.
        returns = []
        G = 0.0
        for step in reversed(episode):
            if step.done:
                G = 0.0
            G = float(step.reward) + self.gamma * G
            returns.append(G)
        returns.reverse()

        states = [s.state for s in episode]
        actions = [s.action for s in episode]
        rewards = [s.reward for s in episode]

        # Pre-allocate tensors for value computation (state-only).
        obs_batch = torch.tensor(states, dtype=torch.float32, device=self.device)  # [T, dim]
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)  # [T]

        loss_val = 0.0
        ent_sum = 0.0
        pol_sum = 0.0
        val_sum = 0.0

        for _ in range(epochs):
            values = self.net.value(obs_batch)  # [T]
            advantages = returns_t - values  # [T]

            policy_loss = torch.tensor(0.0, device=self.device)
            value_loss = F.mse_loss(values, returns_t)
            entropy_sum = torch.tensor(0.0, device=self.device)

            # Policy loss needs (s, a, legal_actions(s))
            for i in range(len(episode)):
                st = states[i]
                act = actions[i]
                # Reconstruct legal actions from obs (same as env rules).
                legal_actions = self._legal_actions_from_obs(st)
                log_prob, entropy = self._log_prob_and_entropy(st, act, legal_actions)
                policy_loss = policy_loss + (-log_prob * advantages[i].detach())
                entropy_sum = entropy_sum + entropy

            policy_loss = policy_loss / len(episode)
            value_loss = value_loss
            entropy_loss = -entropy_sum / len(episode)

            loss = policy_loss + self.value_coef * value_loss + self.entropy_beta * entropy_loss

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optim.step()

            loss_val = float(loss.item())
            ent_sum = float((entropy_sum / len(episode)).item())
            pol_sum = float(policy_loss.item())
            val_sum = float(value_loss.item())

        return {
            "loss": loss_val,
            "policy_loss": pol_sum,
            "value_loss": val_sum,
            "entropy": ent_sum,
        }

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "size": self.size,
            "state_dict": self.net.state_dict(),
            "cfg": {
                "gamma": self.gamma,
                "entropy_beta": self.entropy_beta,
                "value_coef": self.value_coef,
                "lr": self.optim.param_groups[0]["lr"],
            },
        }
        torch.save(payload, str(p))

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "A2CAmazonsAgent":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["cfg"]
        agent = cls(
            size=payload["size"],
            gamma=cfg["gamma"],
            lr=cfg["lr"],
            entropy_beta=cfg["entropy_beta"],
            value_coef=cfg["value_coef"],
            device=device,
        )
        agent.net.load_state_dict(payload["state_dict"])
        return agent

