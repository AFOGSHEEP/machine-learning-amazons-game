from __future__ import annotations

from typing import List, Tuple

from src.agents.alphazero_mcts import AlphaZeroMCTS
from src.agents.alphazero_net import AlphaZeroPVNet
from src.envs.amazons_env import Action


class AlphaZeroPlayer:
    """
    Play using AlphaZero-style MCTS guided by a policy/value network.
    """

    def __init__(self, size: int, net: AlphaZeroPVNet, mcts: AlphaZeroMCTS) -> None:
        self.size = size
        self.net = net
        self.mcts = mcts

    def select_action(
        self,
        state: Tuple[int, ...],
        legal_actions: List[Action],
        training: bool = False,
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available")
        counts, _, _ = self.mcts.run(state)
        best_a = None
        best_n = -1
        for a in legal_actions:
            n = counts.get(a, 0)
            if n > best_n:
                best_n = n
                best_a = a
        return best_a if best_a is not None else legal_actions[0]

