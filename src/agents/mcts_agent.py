from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.envs.amazons_env import Action
from src.envs.amazons_rules import legal_actions_from_board, obs_to_board_and_positions, step_state


@dataclass
class MCTSConfig:
    simulations: int = 400
    c_ucb: float = 1.4
    rollout_depth: int = 60
    seed: int | None = None


class _Node:
    __slots__ = ("obs", "player", "parent", "children", "N", "W")

    def __init__(self, obs: Tuple[int, ...], player: int, parent: "_Node | None") -> None:
        self.obs = obs
        self.player = player
        self.parent = parent
        self.children: Dict[Action, _Node] = {}
        self.N = 0
        self.W = 0.0  # from root player's perspective


class MCTSAmazonsAgent:
    """
    Classic UCT MCTS with random rollouts.
    Works with current env observation (flat board + current_player).
    """

    def __init__(self, size: int = 6, config: MCTSConfig | None = None) -> None:
        self.size = size
        self.cfg = config or MCTSConfig()
        self.rng = random.Random(self.cfg.seed)

    def select_action(
        self,
        state: Tuple[int, ...],
        legal_actions: List[Action],
        training: bool = False,
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available")

        board, positions, current_player = obs_to_board_and_positions(state, size=self.size)
        root = _Node(state, current_player, parent=None)
        root_player = current_player

        for _ in range(self.cfg.simulations):
            # Selection
            node = root
            b = board
            pos = positions
            p = current_player

            path: list[tuple[_Node, Action | None, list[Action]]] = []
            done = False
            winner = -1

            while True:
                legal = legal_actions_from_board(b, pos, p, self.size)
                path.append((node, None, legal))
                if not legal:
                    done = True
                    winner = 1 - p  # player to move has no legal -> loses
                    break

                # Expand if not fully expanded
                untried = [a for a in legal if a not in node.children]
                if untried:
                    a = self.rng.choice(untried)
                    next_obs, terminal, w = step_state(b, pos, p, a, self.size)
                    child = _Node(next_obs, 1 - p, parent=node)
                    node.children[a] = child
                    node = child
                    b, pos, p = obs_to_board_and_positions(next_obs, size=self.size)
                    done = terminal
                    winner = w
                    # Continue with rollout from this newly expanded node
                    break

                # Otherwise choose best UCB child
                a = self._uct_select(node, legal, root_player)
                next_obs, terminal, w = step_state(b, pos, p, a, self.size)
                node = node.children[a]
                b, pos, p = obs_to_board_and_positions(next_obs, size=self.size)
                done = terminal
                winner = w
                if done:
                    break

            # Rollout if non-terminal
            value = self._rollout_value(b, pos, p, root_player, done, winner)

            # Backprop
            self._backprop(node, value)

        # Pick most visited action
        best_a = None
        best_n = -1
        for a in legal_actions:
            child = root.children.get(a)
            n = child.N if child else 0
            if n > best_n:
                best_n = n
                best_a = a
        return best_a if best_a is not None else self.rng.choice(legal_actions)

    def _uct_select(self, node: _Node, legal: list[Action], root_player: int) -> Action:
        logN = math.log(max(1, node.N))
        best_a = legal[0]
        best_score = float("-inf")
        for a in legal:
            child = node.children[a]
            if child.N == 0:
                return a
            q = child.W / child.N
            u = self.cfg.c_ucb * math.sqrt(logN / child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _rollout_value(
        self,
        board,
        positions,
        current_player: int,
        root_player: int,
        done: bool,
        winner: int,
    ) -> float:
        if done:
            if winner == -1:
                return 0.0
            return 1.0 if winner == root_player else -1.0

        b = board
        pos = positions
        p = current_player

        for _ in range(self.cfg.rollout_depth):
            legal = legal_actions_from_board(b, pos, p, self.size)
            if not legal:
                w = 1 - p
                return 1.0 if w == root_player else -1.0
            a = self.rng.choice(legal)
            next_obs, terminal, w = step_state(b, pos, p, a, self.size)
            if terminal:
                return 1.0 if w == root_player else -1.0
            b, pos, p = obs_to_board_and_positions(next_obs, size=self.size)

        return 0.0

    def _backprop(self, leaf: _Node, value: float) -> None:
        node = leaf
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent

