from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from src.agents.alphazero_net import AlphaZeroPVNet, get_device
from src.envs.amazons_env import Action
from src.envs.amazons_rules import legal_actions_from_board, obs_to_board_and_positions, step_state


@dataclass
class PUCTConfig:
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    seed: int | None = None


class _AZNode:
    __slots__ = ("obs", "player", "parent", "children", "N", "W", "P")

    def __init__(self, obs: Tuple[int, ...], player: int, parent: "_AZNode | None") -> None:
        self.obs = obs
        self.player = player
        self.parent = parent
        self.children: Dict[Action, _AZNode] = {}
        self.N = 0
        self.W = 0.0
        self.P: Dict[Action, float] = {}

    @property
    def Q(self) -> float:
        return 0.0 if self.N == 0 else self.W / self.N


class AlphaZeroMCTS:
    def __init__(self, size: int, net: AlphaZeroPVNet, cfg: PUCTConfig | None = None, device: str | None = None) -> None:
        self.size = size
        self.net = net
        self.cfg = cfg or PUCTConfig()
        self.rng = random.Random(self.cfg.seed)
        self.device = get_device(device)

    def run(self, obs: Tuple[int, ...]) -> tuple[Dict[Action, int], Dict[Action, float], float]:
        """
        Returns:
        - visit counts N(a)
        - policy pi(a) normalized from counts
        - root value estimate V(s)
        """
        board, positions, current_player = obs_to_board_and_positions(obs, size=self.size)
        root = _AZNode(obs, current_player, parent=None)

        legal_root = legal_actions_from_board(board, positions, current_player, self.size)
        if not legal_root:
            return {}, {}, -1.0

        # Root prior + optional dirichlet noise
        priors = self._policy_priors(obs, legal_root)
        if self.cfg.dirichlet_eps > 0:
            noise = self._dirichlet(len(legal_root), self.cfg.dirichlet_alpha)
            for i, a in enumerate(legal_root):
                priors[a] = (1 - self.cfg.dirichlet_eps) * priors[a] + self.cfg.dirichlet_eps * noise[i]
        root.P = priors

        for _ in range(self.cfg.simulations):
            node = root
            b = board
            pos = positions
            p = current_player

            # Selection
            while True:
                legal = legal_actions_from_board(b, pos, p, self.size)
                if not legal:
                    # terminal: player to move loses => value -1 for current player
                    v = -1.0
                    self._backprop(node, v)
                    break

                # Expand
                unexpanded = [a for a in legal if a not in node.children]
                if unexpanded:
                    a = self._select_puct(node, legal)
                    next_obs, done, winner = step_state(b, pos, p, a, self.size)
                    child = _AZNode(next_obs, 1 - p, parent=node)
                    node.children[a] = child

                    if done:
                        v = 1.0 if winner == p else -1.0
                        self._backprop(child, v)
                        break

                    # Evaluate leaf with net
                    b2, pos2, p2 = obs_to_board_and_positions(next_obs, size=self.size)
                    legal2 = legal_actions_from_board(b2, pos2, p2, self.size)
                    if not legal2:
                        # next player has no moves => current player (p) already won, but step_state would have been done
                        v = 1.0
                        self._backprop(child, v)
                        break

                    child.P = self._policy_priors(next_obs, legal2)
                    v_leaf = float(self._value(next_obs))
                    self._backprop(child, v_leaf)
                    break

                # Fully expanded: keep selecting
                a = self._select_puct(node, legal)
                next_obs, done, winner = step_state(b, pos, p, a, self.size)
                node = node.children[a]
                if done:
                    v = 1.0 if winner == p else -1.0
                    self._backprop(node, v)
                    break
                b, pos, p = obs_to_board_and_positions(next_obs, size=self.size)

        counts = {a: root.children[a].N if a in root.children else 0 for a in legal_root}
        total = sum(counts.values())
        pi = {a: (counts[a] / total if total > 0 else 1.0 / len(legal_root)) for a in legal_root}
        v_root = float(self._value(obs))
        return counts, pi, v_root

    def _value(self, obs: Tuple[int, ...]) -> float:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.net.value(x).item()
        return float(v)

    def _policy_priors(self, obs: Tuple[int, ...], legal: List[Action]) -> Dict[Action, float]:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        acts = torch.tensor(legal, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.net.logits_for_actions(x.expand(acts.shape[0], -1), acts)
            probs = torch.softmax(logits, dim=0).cpu().numpy().tolist()
        return {a: float(p) for a, p in zip(legal, probs)}

    def _select_puct(self, node: _AZNode, legal: List[Action]) -> Action:
        best_a = legal[0]
        best_score = float("-inf")
        sqrtN = math.sqrt(max(1, node.N))
        for a in legal:
            p = node.P.get(a, 1e-8)
            child = node.children.get(a)
            n = 0 if child is None else child.N
            q = 0.0 if child is None or child.N == 0 else child.W / child.N
            u = self.cfg.c_puct * p * (sqrtN / (1 + n))
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _backprop(self, leaf: _AZNode, value: float) -> None:
        """
        value is from leaf player's perspective; flip sign each ply when backing up.
        """
        node = leaf
        v = value
        while node is not None:
            node.N += 1
            node.W += v
            v = -v
            node = node.parent

    def _dirichlet(self, k: int, alpha: float) -> List[float]:
        # Simple gamma-based sampler
        xs = [random.gammavariate(alpha, 1.0) for _ in range(k)]
        s = sum(xs)
        return [x / s for x in xs]

