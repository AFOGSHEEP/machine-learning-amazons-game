from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Deque, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.replay_buffers import PrioritizedReplayBuffer, n_step_push
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
    """
    obs_batch: [B, size*size + 1], values in board are in {0,1,2,3}.
    returns: [B, size*size*4 + 1]
    """
    board_flat = obs_batch[:, : size * size].long()  # [B, cells]
    current_player = obs_batch[:, -1].float().unsqueeze(1)  # [B, 1]
    one_hot = F.one_hot(board_flat, num_classes=4).float()  # [B, cells, 4]
    one_hot = one_hot.view(obs_batch.shape[0], -1)
    return torch.cat([one_hot, current_player], dim=1)


def _encode_action(action_batch: torch.Tensor, size: int) -> torch.Tensor:
    """
    action_batch: [B, 6] with coords in [0, size-1]
    returns: [B, 6] normalized to [0, 1]
    """
    if size <= 1:
        return action_batch.float()
    return (action_batch.float() / float(size - 1)).clamp(0.0, 1.0)


class QNet(nn.Module):
    def __init__(self, size: int, hidden: int = 256) -> None:
        super().__init__()
        self.size = size
        state_dim = size * size * 4 + 1
        action_dim = 6
        in_dim = state_dim + action_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        x_state = _encode_state_onehot(obs_batch, self.size)
        x_action = _encode_action(action_batch, self.size)
        x = torch.cat([x_state, x_action], dim=1)
        return self.mlp(x).squeeze(-1)  # [B]


@dataclass
class Transition:
    state: Tuple[int, ...]
    action: Action
    reward: float
    next_state: Tuple[int, ...]
    done: bool
    horizon: int = 1  # effective n for gamma**horizon bootstrap


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.data: List[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, t: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.data, batch_size)


class DQNAmazonsAgent:
    """
    Parameterized-action DQN:
    - Network evaluates Q(s, a) for a specific action tuple.
    - During selection, we enumerate all legal actions and take argmax.

    Optional upgrades (Rainbow-style building blocks):
    - Prioritized experience replay (Schaul et al., 2015)
    - n-step returns (Sutton & Barto)
    """

    def __init__(
        self,
        size: int = 6,
        gamma: float = 0.98,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 50_000,
        batch_size: int = 64,
        start_learning: int = 2_000,
        target_update_interval: int = 250,
        device: str | None = None,
        double_dqn: bool = True,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_anneal_steps: int = 100_000,
        n_step: int = 1,
    ) -> None:
        self.size = size
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_learning = start_learning
        self.target_update_interval = target_update_interval
        self.double_dqn = bool(double_dqn)
        self.use_per = bool(use_per)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_anneal_steps = int(per_beta_anneal_steps)
        self.n_step = max(1, int(n_step))

        self.device = _get_device(device)

        self.q_net = QNet(size=size).to(self.device)
        self.target_net = QNet(size=size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.replay: Union[ReplayBuffer, PrioritizedReplayBuffer[Transition]]
        if self.use_per:
            self.replay = PrioritizedReplayBuffer[Transition](capacity=replay_capacity, alpha=per_alpha)
        else:
            self.replay = ReplayBuffer(capacity=replay_capacity)

        self._nstep_buf: Deque[Tuple[Tuple[int, ...], Action, float, Tuple[int, ...], bool]] = deque()
        self.learn_steps = 0

    def _per_beta(self) -> float:
        if not self.use_per:
            return 1.0
        t = min(1.0, self.learn_steps / max(1, self.per_beta_anneal_steps))
        return self.per_beta_start + t * (self.per_beta_end - self.per_beta_start)

    def _replay_push(self, t: Transition) -> None:
        if self.use_per:
            self.replay.add(t)
        else:
            self.replay.push(t)

    def select_action(
        self,
        state: Tuple[int, ...],
        legal_actions: List[Action],
        training: bool = False,
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available")

        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        obs = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = torch.tensor(legal_actions, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_values = self.q_net(obs.expand(actions.shape[0], -1), actions)
            idx = int(torch.argmax(q_values).item())
        return legal_actions[idx]

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def remember(self, state, action, reward, next_state, done) -> None:
        if self.n_step <= 1:
            self._replay_push(
                Transition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    done=bool(done),
                    horizon=1,
                )
            )
            return

        def _emit(s, a, R, s2, d, horizon: int) -> None:
            self._replay_push(
                Transition(
                    state=s,
                    action=a,
                    reward=float(R),
                    next_state=s2,
                    done=bool(d),
                    horizon=int(horizon),
                )
            )

        n_step_push(
            self._nstep_buf,
            (state, action, float(reward), next_state, bool(done)),
            self.n_step,
            self.gamma,
            _emit,
        )

    def flush_nstep(self) -> None:
        """Call between episodes if environment can end without passing done through remember (rare)."""
        self._nstep_buf.clear()

    def _legal_actions_from_obs(self, obs: Tuple[int, ...]) -> List[Action]:
        board, positions, current_player = obs_to_board_and_positions(obs, size=self.size)
        return legal_actions_from_board(board, positions, current_player, self.size)

    def _bootstrap_discount(self, t: Transition) -> float:
        return float(self.gamma ** t.horizon)

    def learn(self) -> None:
        if len(self.replay) < self.start_learning:
            return
        if len(self.replay) < self.batch_size:
            return

        if self.use_per:
            batch, tree_indices, weights_np = self.replay.sample(self.batch_size, self._per_beta())
            weights = torch.tensor(weights_np, dtype=torch.float32, device=self.device)
        else:
            batch = self.replay.sample(self.batch_size)
            tree_indices = None
            weights = torch.ones(self.batch_size, device=self.device)

        state_batch = torch.tensor([t.state for t in batch], dtype=torch.float32, device=self.device)
        action_batch = torch.tensor([t.action for t in batch], dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor([t.next_state for t in batch], dtype=torch.float32, device=self.device)

        q_pred = self.q_net(state_batch, action_batch)

        targets = torch.empty(self.batch_size, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for i, t in enumerate(batch):
                if t.done:
                    targets[i] = t.reward
                    continue
                next_legal = self._legal_actions_from_obs(t.next_state)
                if not next_legal:
                    targets[i] = t.reward
                    continue
                next_actions = torch.tensor(next_legal, dtype=torch.float32, device=self.device)
                next_obs_rep = next_state_batch[i : i + 1].expand(next_actions.shape[0], -1)
                disc = self._bootstrap_discount(t)
                if self.double_dqn:
                    q_online = self.q_net(next_obs_rep, next_actions)
                    best_idx = int(torch.argmax(q_online).item())
                    q_tgt = self.target_net(next_obs_rep, next_actions)
                    best_q = float(q_tgt[best_idx].item())
                else:
                    q_tgt = self.target_net(next_obs_rep, next_actions)
                    best_q = float(torch.max(q_tgt).item())
                targets[i] = t.reward + disc * best_q

        td_err = (q_pred - targets).detach().abs().cpu().numpy()
        if self.use_per and tree_indices is not None:
            self.replay.update_priorities(tree_indices, td_err)

        loss_vec = F.smooth_l1_loss(q_pred, targets, reduction="none")
        loss = (weights * loss_vec).mean()

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optim.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def sync_target_hard(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def make_greedy_eval_clone(self) -> "DQNAmazonsAgent":
        """Frozen policy copy (no training calls expected). Used in opponent-pool self-play."""
        o = DQNAmazonsAgent(
            size=self.size,
            gamma=self.gamma,
            lr=self.optim.param_groups[0]["lr"],
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            replay_capacity=256,
            batch_size=self.batch_size,
            start_learning=10**9,
            target_update_interval=10**9,
            device=str(self.device),
            double_dqn=self.double_dqn,
            use_per=False,
            n_step=1,
        )
        o.q_net.load_state_dict(self.q_net.state_dict())
        o.target_net.load_state_dict(self.target_net.state_dict())
        return o

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "size": self.size,
            "q_state_dict": self.q_net.state_dict(),
            "cfg": {
                "gamma": self.gamma,
                "lr": self.optim.param_groups[0]["lr"],
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "replay_capacity": self.replay.capacity,
                "batch_size": self.batch_size,
                "start_learning": self.start_learning,
                "target_update_interval": self.target_update_interval,
                "double_dqn": self.double_dqn,
                "use_per": self.use_per,
                "per_alpha": getattr(self.replay, "alpha", 0.6),
                "per_beta_start": self.per_beta_start,
                "per_beta_end": self.per_beta_end,
                "per_beta_anneal_steps": self.per_beta_anneal_steps,
                "n_step": self.n_step,
            },
        }
        torch.save(payload, str(p))

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DQNAmazonsAgent":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["cfg"]
        agent = cls(
            size=payload["size"],
            gamma=cfg["gamma"],
            lr=cfg["lr"],
            epsilon=0.0,
            epsilon_min=cfg["epsilon_min"],
            epsilon_decay=cfg["epsilon_decay"],
            replay_capacity=cfg["replay_capacity"],
            batch_size=cfg["batch_size"],
            start_learning=cfg["start_learning"],
            target_update_interval=cfg["target_update_interval"],
            device=device,
            double_dqn=cfg.get("double_dqn", True),
            use_per=cfg.get("use_per", False),
            per_alpha=cfg.get("per_alpha", 0.6),
            per_beta_start=cfg.get("per_beta_start", 0.4),
            per_beta_end=cfg.get("per_beta_end", 1.0),
            per_beta_anneal_steps=cfg.get("per_beta_anneal_steps", 100_000),
            n_step=cfg.get("n_step", 1),
        )
        agent.q_net.load_state_dict(payload["q_state_dict"])
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        return agent
