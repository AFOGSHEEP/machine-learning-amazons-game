from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    one_hot = F.one_hot(board_flat, num_classes=4).float().view(obs_batch.shape[0], -1)
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

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(state_dim + 6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def value(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.value_net(_encode_state_onehot(obs_batch, self.size)).squeeze(-1)

    def logits_for_actions(self, obs_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                _encode_state_onehot(obs_batch, self.size),
                _encode_action(action_batch, self.size),
            ],
            dim=1,
        )
        return self.policy_head(x).squeeze(-1)


@dataclass
class PPOStep:
    state: Tuple[int, ...]
    action: Action
    reward: float
    done: bool
    old_log_prob: float


class PPOAmazonsAgent:
    def __init__(
        self,
        size: int = 6,
        gamma: float = 0.98,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_beta: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: str | None = None,
    ) -> None:
        self.size = size
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_beta = entropy_beta
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = _get_device(device)

        self.net = PolicyValueNet(size=size).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _legal_actions_from_obs(self, obs: Tuple[int, ...]) -> List[Action]:
        board, positions, current_player = obs_to_board_and_positions(obs, size=self.size)
        return legal_actions_from_board(board, positions, current_player, self.size)

    def select_action_with_logprob(
        self,
        state: Tuple[int, ...],
        legal_actions: List[Action],
        training: bool = True,
    ) -> tuple[Action, float]:
        if not legal_actions:
            raise ValueError("No legal actions available")
        obs = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = torch.tensor(legal_actions, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.net.logits_for_actions(obs.expand(actions.shape[0], -1), actions)
            log_probs = F.log_softmax(logits, dim=0)
            if training:
                probs = torch.exp(log_probs)
                idx = int(torch.multinomial(probs, 1).item())
            else:
                idx = int(torch.argmax(logits).item())
            lp = float(log_probs[idx].item())
        return legal_actions[idx], lp

    def select_action(
        self,
        state: Tuple[int, ...],
        legal_actions: List[Action],
        training: bool = False,
    ) -> Action:
        action, _ = self.select_action_with_logprob(state, legal_actions, training=training)
        return action

    def train_on_episode(self, episode: List[PPOStep], epochs: int = 4) -> dict:
        if not episode:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        returns = []
        g = 0.0
        for step in reversed(episode):
            if step.done:
                g = 0.0
            g = float(step.reward) + self.gamma * g
            returns.append(g)
        returns.reverse()

        states = [s.state for s in episode]
        actions = [s.action for s in episode]
        old_log_probs = torch.tensor([s.old_log_prob for s in episode], dtype=torch.float32, device=self.device)
        obs_batch = torch.tensor(states, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        out_loss = out_pl = out_vl = out_ent = 0.0
        for _ in range(epochs):
            values = self.net.value(obs_batch)
            advantages = (returns_t - values).detach()

            policy_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            entropy_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            for i, st in enumerate(states):
                legal = self._legal_actions_from_obs(st)
                obs = torch.tensor(st, dtype=torch.float32, device=self.device).unsqueeze(0)
                acts = torch.tensor(legal, dtype=torch.float32, device=self.device)
                logits = self.net.logits_for_actions(obs.expand(acts.shape[0], -1), acts)
                log_probs = F.log_softmax(logits, dim=0)
                probs = torch.exp(log_probs)

                idx = legal.index(actions[i])
                new_log_prob = log_probs[idx]
                ratio = torch.exp(new_log_prob - old_log_probs[i])
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[i]
                policy_loss = policy_loss + (-torch.min(surr1, surr2))
                entropy_sum = entropy_sum + (-(probs * log_probs).sum())

            policy_loss = policy_loss / len(episode)
            value_loss = F.mse_loss(values, returns_t)
            entropy = entropy_sum / len(episode)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_beta * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optim.step()

            out_loss = float(loss.item())
            out_pl = float(policy_loss.item())
            out_vl = float(value_loss.item())
            out_ent = float(entropy.item())

        return {"loss": out_loss, "policy_loss": out_pl, "value_loss": out_vl, "entropy": out_ent}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "size": self.size,
            "state_dict": self.net.state_dict(),
            "cfg": {
                "gamma": self.gamma,
                "clip_eps": self.clip_eps,
                "entropy_beta": self.entropy_beta,
                "value_coef": self.value_coef,
                "lr": self.optim.param_groups[0]["lr"],
            },
        }
        torch.save(payload, str(p))

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "PPOAmazonsAgent":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["cfg"]
        agent = cls(
            size=payload["size"],
            gamma=cfg["gamma"],
            lr=cfg["lr"],
            clip_eps=cfg["clip_eps"],
            entropy_beta=cfg["entropy_beta"],
            value_coef=cfg["value_coef"],
            device=device,
        )
        agent.net.load_state_dict(payload["state_dict"])
        return agent

