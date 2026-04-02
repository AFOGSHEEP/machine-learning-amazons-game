from __future__ import annotations

import random
from typing import List, Tuple

from src.envs.amazons_env import Action
from src.envs.amazons_rules import (
    apply_action,
    legal_actions_from_board,
    obs_to_board_and_positions,
)


class HeuristicAmazonsAgent:
    """
    Simple adversarial heuristic:
    choose the action that minimizes opponent mobility.
    """

    def __init__(self, seed: int | None = None) -> None:
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
        best_score: float = float("-inf")

        for action in legal_actions:
            b2, pos2 = apply_action(board, positions, action, player)
            opp = 1 - player

            opp_legal = legal_actions_from_board(b2, pos2, opp, size)
            # If opponent has no legal action, we win immediately.
            if not opp_legal:
                score = 1e9
            else:
                # Control term: fewer opponent moves is better.
                score = -float(len(opp_legal))

            if score > best_score + 1e-9:
                best_score = score
                best_actions = [action]
            elif abs(score - best_score) <= 1e-9:
                best_actions.append(action)

        return self.rng.choice(best_actions)

