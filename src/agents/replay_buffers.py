"""
Prioritized experience replay (Schaul et al., 2015) with sum-tree sampling.
Used for Rainbow-style DQN upgrades (PER + n-step handled in agent).
"""
from __future__ import annotations

from collections import deque
import random
from typing import Callable, Deque, Generic, List, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


class SumTree:
    """Binary sum tree for O(log n) proportional sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def update(self, data_idx: int, priority: float) -> None:
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, float]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx, float(self.tree[idx])


class PrioritizedReplayBuffer(Generic[T]):
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.tree = SumTree(self.capacity)
        self.data: List[Optional[T]] = [None] * self.capacity
        self.write = 0
        self.n_entries = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.n_entries

    def add(self, item: T) -> None:
        p = (self.max_priority + self.eps) ** self.alpha
        self.tree.update(self.write, p)
        self.data[self.write] = item
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Tuple[List[T], np.ndarray, np.ndarray]:
        n = len(self)
        if n < batch_size:
            raise ValueError("Not enough samples in PER buffer")
        total = self.tree.total()
        if total <= 0:
            raise ValueError("SumTree total is non-positive")

        batch: List[T] = []
        data_indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)

        segment = total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            data_idx, prio = self.tree.get(s)
            # resolve if empty slot (should not happen)
            tries = 0
            while self.data[data_idx] is None and tries < 32:
                s = random.uniform(0.0, total)
                data_idx, prio = self.tree.get(s)
                tries += 1
            if self.data[data_idx] is None:
                data_idx = 0
            batch.append(self.data[data_idx])  # type: ignore[arg-type]
            data_indices[i] = data_idx
            tree_indices[i] = data_idx + self.capacity - 1
            sampling_prob = prio / total
            weights[i] = (n * max(sampling_prob, 1e-12)) ** (-beta)

        m = float(weights.max()) if weights.max() > 0 else 1.0
        weights /= m
        return batch, tree_indices, weights

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        for ti, td in zip(tree_indices, td_errors):
            pr = float(abs(td)) + self.eps
            self.max_priority = max(self.max_priority, pr)
            data_idx = int(ti) - self.capacity + 1
            if 0 <= data_idx < self.capacity:
                p = pr**self.alpha
                self.tree.update(data_idx, p)


def n_step_push(
    buf: Deque[Tuple[object, object, float, object, bool]],
    transition: Tuple[object, object, float, object, bool],
    n: int,
    gamma: float,
    emit: Callable[[object, object, float, object, bool, int], None],
) -> None:
    """
    transition: (state, action, reward, next_state, done) for the acting agent.
    Emits n-step transitions via emit(...) when buffer is ready or on terminal.
    """
    buf.append(transition)
    s, a, r, s2, done = transition
    _ = (s, a, r, s2)
    if done:
        while buf:
            m = min(len(buf), n)
            items = [buf[i] for i in range(m)]
            R = sum((gamma**i) * items[i][2] for i in range(m))
            emit(
                items[0][0],
                items[0][1],
                R,
                items[-1][3],
                any(items[j][4] for j in range(m)),
                m,
            )
            buf.popleft()
        return
    if len(buf) >= n:
        items = [buf[i] for i in range(n)]
        R = sum((gamma**i) * items[i][2] for i in range(n))
        emit(
            items[0][0],
            items[0][1],
            R,
            items[-1][3],
            any(items[j][4] for j in range(n)),
            n,
        )
        buf.popleft()
