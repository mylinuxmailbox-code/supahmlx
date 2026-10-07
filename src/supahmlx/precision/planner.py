"""Precision planning placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionPlan:
    policy: str


def plan_precision(mode_name: str) -> PrecisionPlan:
    """Return a mode-linked precision plan for future backend specialization."""
    return PrecisionPlan(policy=mode_name)
