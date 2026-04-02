from __future__ import annotations

from typing import List, Tuple

import torch

from src.agents.alphazero_net import AlphaZeroPVNet, get_device
from src.envs.amazons_env import Action


class BCPolicyPlayer:
    """
    Use a (s,a)->logit policy head (trained by behavior cloning) to select greedy action.
    """

    def __init__(self, size: int, net: AlphaZeroPVNet, device: str | None = None) -> None:
        self.size = size
        self.net = net
        self.device = get_device(device)
        self.net.to(self.device)
        self.net.eval()

    def select_action(
        self,
        state: Tuple[int, ...],
        legal_actions: List[Action],
        training: bool = False,
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available")
        obs = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        acts = torch.tensor(legal_actions, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.net.logits_for_actions(obs.expand(acts.shape[0], -1), acts)
            idx = int(torch.argmax(logits).item())
        return legal_actions[idx]

