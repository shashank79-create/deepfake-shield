"""
utils/progress.py
Live terminal progress bar with ETA.
"""

import sys
import time


class ProgressBar:
    """Shows real-time progress during PGD attack steps."""

    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self.start = time.time()

    def update(self, step: int, _total: int, loss: float):
        pct     = step / self.total
        filled  = int(self.width * pct)
        bar     = "=" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        eta     = (elapsed / step * (self.total - step)) if step > 0 else 0
        sys.stdout.write(
            f"\r  [{bar}] {pct*100:5.1f}%  step {step:>4}/{self.total}"
            f"  loss {loss:.4f}  ETA {eta:.0f}s"
        )
        sys.stdout.flush()

    def finish(self):
        elapsed = time.time() - self.start
        sys.stdout.write(f"\n  Done in {elapsed:.1f}s\n")
        sys.stdout.flush()
