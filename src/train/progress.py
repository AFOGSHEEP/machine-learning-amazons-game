from __future__ import annotations

import sys
import time


class TerminalProgressBar:
    """
    Lightweight terminal progress bar without external dependencies.
    """

    def __init__(self, total: int, title: str = "Training", width: int = 30) -> None:
        self.total = max(1, int(total))
        self.title = title
        self.width = max(10, int(width))
        self.start_ts = time.time()
        self.last_len = 0

    def update(self, current: int, extra: str = "") -> None:
        cur = min(max(0, int(current)), self.total)
        ratio = cur / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start_ts
        eta = (elapsed / cur * (self.total - cur)) if cur > 0 else 0.0
        msg = f"{self.title} [{bar}] {cur}/{self.total} ({ratio*100:5.1f}%) ETA {eta:6.1f}s"
        if extra:
            msg += f" | {extra}"
        pad = max(0, self.last_len - len(msg))
        sys.stdout.write("\r" + msg + " " * pad)
        sys.stdout.flush()
        self.last_len = len(msg)

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

