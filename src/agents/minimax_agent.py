from __future__ import annotations

import math
import random
from typing import List, Tuple

from src.envs.amazons_env import Action
from src.envs.amazons_rules import (
    apply_action,
    legal_actions_from_board,
    obs_to_board_and_positions,
)


class MinimaxAmazonsAgent:
    """
    Alpha-beta minimax with a cheap mobility heuristic.

    depth=1 means: evaluate after our single move (opponent not expanded).
    depth=2 means: our move + opponent reply, etc.
    """

    def __init__(self, depth: int = 2, seed: int | None = None) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.rng = random.Random(seed)

    def select_action(
        self,
        state: Tuple[int, ...],
        legal_actions: List[Action],
        training: bool = False,
    ) -> Action:
        board, positions, player = obs_to_board_and_positions(state)
        size = len(board)

        if not legal_actions:
            raise ValueError("No legal actions available")

        best_actions: List[Action] = []
        best_value = float("-inf")

        # Root player is the side to move at `state`.
        root_player = player

        for action in legal_actions:
            b2, pos2 = apply_action(board, positions, action, player)
            value = self._minimax(
                b2,
                pos2,
                current_player=1 - player,
                root_player=root_player,
                depth=self.depth - 1,
                size=size,
                alpha=float("-inf"),
                beta=float("inf"),
            )
            if value > best_value + 1e-9:
                best_value = value
                best_actions = [action]
            elif abs(value - best_value) <= 1e-9:
                best_actions.append(action)

        return self.rng.choice(best_actions)

    def _heuristic(self, board, positions, root_player: int, size: int) -> float:
        own = legal_actions_from_board(board, positions, root_player, size)
        opp = legal_actions_from_board(board, positions, 1 - root_player, size)
        return float(len(own) - len(opp))

    def _minimax(
        self,
        board,
        positions,
        current_player: int,
        root_player: int,
        depth: int,
        size: int,
        alpha: float,
        beta: float,
    ) -> float:
        legal = legal_actions_from_board(board, positions, current_player, size)

        # Terminal: no legal actions => current_player loses.
        if not legal:
            if current_player == root_player:
                return -1e9
            return 1e9

        if depth <= 0:
            return self._heuristic(board, positions, root_player, size)

        maximizing = current_player == root_player
        if maximizing:
            value = float("-inf")
            for action in legal:
                b2, pos2 = apply_action(board, positions, action, current_player)
                value = max(
                    value,
                    self._minimax(
                        b2,
                        pos2,
                        current_player=1 - current_player,
                        root_player=root_player,
                        depth=depth - 1,
                        size=size,
                        alpha=alpha,
                        beta=beta,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for action in legal:
                b2, pos2 = apply_action(board, positions, action, current_player)
                value = min(
                    value,
                    self._minimax(
                        b2,
                        pos2,
                        current_player=1 - current_player,
                        root_player=root_player,
                        depth=depth - 1,
                        size=size,
                        alpha=alpha,
                        beta=beta,
                    ),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

