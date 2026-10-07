"""Graph optimizer extension point."""

from __future__ import annotations


class NoopGraphOptimizer:
    """Baseline graph optimizer doing no transformations yet."""

    def optimize(self) -> None:
        return None
