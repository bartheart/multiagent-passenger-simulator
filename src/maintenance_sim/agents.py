"""
SubsystemAgent: models one aircraft subsystem as a stateful agent.

Each agent maintains a health score (0–1) and degrades each cycle
based on: natural wear, incoming degradation signals from upstream
subsystems (via the knowledge graph), and stochastic noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from maintenance_sim.config import (
    ALERT_THRESHOLDS,
    BASE_DECAY_RATES,
    FAILURE_THRESHOLD,
    NOISE_STD,
    SUBSYSTEMS,
)


@dataclass
class SubsystemAgent:
    name: str
    health: float                     # current health [0, 1]
    base_decay_rate: float            # natural degradation per cycle
    noise_std: float                  # stochastic noise std dev
    rng: np.random.Generator

    def tick(self, incoming_signals: list[tuple[float, float]]) -> float:
        """
        Advance one simulation cycle.

        Args:
            incoming_signals: list of (signal_strength, edge_weight) tuples
                from upstream agents. signal_strength = max(0, 1 - upstream_health).

        Returns:
            emitted_signal: degradation delta emitted to downstream nodes.
        """
        health_before = self.health

        # Natural decay
        decay = self.base_decay_rate

        # Propagated degradation from upstream agents
        cascade_decay = sum(signal * weight for signal, weight in incoming_signals)

        # Stochastic noise (can be positive — natural variation / minor repairs)
        noise = self.rng.normal(0, self.noise_std)

        self.health = float(np.clip(
            self.health - decay - cascade_decay + noise,
            0.0, 1.0
        ))

        return max(0.0, health_before - self.health)

    @property
    def is_failed(self) -> bool:
        return self.health < FAILURE_THRESHOLD

    @property
    def alert_level(self) -> str:
        if self.health >= ALERT_THRESHOLDS["nominal"]:
            return "nominal"
        if self.health >= ALERT_THRESHOLDS["watch"]:
            return "watch"
        if self.health >= ALERT_THRESHOLDS["caution"]:
            return "caution"
        return "critical"

    def clone(self) -> "SubsystemAgent":
        """Return a copy for Monte Carlo branching (shares rng state, so reseed)."""
        return SubsystemAgent(
            name=self.name,
            health=self.health,
            base_decay_rate=self.base_decay_rate,
            noise_std=self.noise_std,
            rng=np.random.default_rng(int(self.rng.integers(0, 2**31))),
        )


def create_agents(
    health_states: dict[str, float],
    seed: int = 42,
) -> dict[str, SubsystemAgent]:
    """
    Instantiate one SubsystemAgent per subsystem with given health states.
    Each agent gets its own seeded RNG for reproducibility.
    """
    agents = {}
    for i, name in enumerate(SUBSYSTEMS):
        agents[name] = SubsystemAgent(
            name=name,
            health=health_states.get(name, 0.85),
            base_decay_rate=BASE_DECAY_RATES[name],
            noise_std=NOISE_STD[name],
            rng=np.random.default_rng(seed + i),
        )
    return agents
