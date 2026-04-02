from __future__ import annotations

import random
from typing import List, Tuple

Action = Tuple[int, int, int, int, int, int]


class RandomAmazonsAgent:
    def select_action(
        self,
        state_or_legal_actions: List[Action] | Tuple[int, ...],
        legal_actions: List[Action] | None = None,
        training: bool = False,
    ) -> Action:
        """
        Compatible signature:
        - select_action(legal_actions, training=False)
        - select_action(state, legal_actions, training=False)
        """
        # Called as select_action(legal_actions, ...)
        if legal_actions is None:
            legal = state_or_legal_actions  # type: ignore[assignment]
        else:
            legal = legal_actions

        if not legal:
            raise ValueError("No legal actions available")
        return random.choice(legal)
