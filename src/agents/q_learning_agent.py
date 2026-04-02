from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import json
import random

Action = Tuple[int, int, int, int, int, int]


@dataclass
class QConfig:
    alpha: float = 0.1
    gamma: float = 0.98
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995


class TabularQLearningAmazonsAgent:
    def __init__(self, config: QConfig | None = None) -> None:
        self.cfg = config or QConfig()
        self.q = defaultdict(dict)  # state -> {action_str: q_value}

    def select_action(self, state: Tuple[int, ...], legal_actions: List[Action], training: bool = True) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available")

        if training and random.random() < self.cfg.epsilon:
            return random.choice(legal_actions)

        best_a = legal_actions[0]
        best_q = self._q(state, best_a)

        for a in legal_actions[1:]:
            qv = self._q(state, a)
            if qv > best_q:
                best_q = qv
                best_a = a

        return best_a

    def update(
        self,
        state: Tuple[int, ...],
        action: Action,
        reward: float,
        next_state: Tuple[int, ...],
        next_legal_actions: List[Action],
        done: bool,
    ) -> None:
        old = self._q(state, action)
        if done or not next_legal_actions:
            target = reward
        else:
            next_max = max(self._q(next_state, a) for a in next_legal_actions)
            target = reward + self.cfg.gamma * next_max

        new_q = old + self.cfg.alpha * (target - old)
        self.q[state][self._action_key(action)] = new_q

    def decay_epsilon(self) -> None:
        self.cfg.epsilon = max(self.cfg.epsilon_min, self.cfg.epsilon * self.cfg.epsilon_decay)

    def _q(self, state: Tuple[int, ...], action: Action) -> float:
        return self.q[state].get(self._action_key(action), 0.0)

    @staticmethod
    def _action_key(action: Action) -> str:
        return ",".join(str(x) for x in action)

    def save(self, path: str) -> None:
        serial = {
            "config": {
                "alpha": self.cfg.alpha,
                "gamma": self.cfg.gamma,
                "epsilon": self.cfg.epsilon,
                "epsilon_min": self.cfg.epsilon_min,
                "epsilon_decay": self.cfg.epsilon_decay,
            },
            "q": {
                "|".join(str(x) for x in state): table
                for state, table in self.q.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serial, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "TabularQLearningAmazonsAgent":
        with open(path, "r", encoding="utf-8") as f:
            serial = json.load(f)

        agent = cls(QConfig(**serial["config"]))
        for k, table in serial["q"].items():
            state = tuple(int(x) for x in k.split("|"))
            agent.q[state] = {ak: float(v) for ak, v in table.items()}
        return agent
