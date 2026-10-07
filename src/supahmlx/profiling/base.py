"""Reference profiler utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BasicProfiler:
    """Simple phase profiler for benchmark and CLI output."""

    _starts: dict[str, float] = field(default_factory=dict)
    elapsed: dict[str, float] = field(default_factory=dict)

    def start(self, phase: str) -> None:
        self._starts[phase] = time.perf_counter()

    def stop(self, phase: str) -> float:
        started = self._starts.pop(phase, None)
        if started is None:
            return 0.0
        delta = time.perf_counter() - started
        self.elapsed[phase] = self.elapsed.get(phase, 0.0) + delta
        return delta
