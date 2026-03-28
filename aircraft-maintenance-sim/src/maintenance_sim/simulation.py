"""
SimulationEngine: runs the multi-agent degradation simulation.

step()            — advance all agents one cycle
run(n)            — run n cycles, return health snapshots
forward_simulate  — Monte Carlo forward simulation for failure probabilities
"""

from __future__ import annotations

import copy
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from maintenance_sim.agents import SubsystemAgent, create_agents
from maintenance_sim.config import (
    DEFAULT_FORWARD_HORIZON,
    DEFAULT_MONTE_CARLO_BRANCHES,
    FAILURE_THRESHOLD,
    SUBSYSTEMS,
)
from maintenance_sim.fleet import Aircraft
from maintenance_sim.memory import save_snapshot


@dataclass
class ForwardSimResult:
    subsystem: str
    failure_prob: float
    cascade_chain: list[str]       # most common path leading to failure
    expected_cycle: float | None   # expected cycle of first failure (or None)


class SimulationEngine:
    def __init__(
        self,
        aircraft: Aircraft,
        graph: nx.DiGraph,
        conn: sqlite3.Connection | None = None,
    ):
        self.aircraft = aircraft
        self.G = graph
        self.conn = conn
        self.cycle = 0
        self.agents: dict[str, SubsystemAgent] = create_agents(
            aircraft.health_states, seed=aircraft.seed
        )
        # Topological sort for consistent propagation order
        try:
            self._topo_order = list(nx.topological_sort(self.G))
        except nx.NetworkXUnfeasible:
            self._topo_order = list(self.G.nodes)

    @property
    def health_states(self) -> dict[str, float]:
        return {name: agent.health for name, agent in self.agents.items()}

    def step(self) -> dict[str, float]:
        """
        Advance all agents one cycle via topological propagation.

        For each node N in topo order:
          - gather incoming signals from predecessor nodes P:
              signal = max(0, 1 - agents[P].health) * G[P][N]['weight']
          - call agents[N].tick(signals)
        """
        emitted: dict[str, float] = {}

        for node in self._topo_order:
            if node not in self.agents:
                continue
            incoming = []
            for pred in self.G.predecessors(node):
                if pred in self.agents:
                    # Signal only fires meaningfully below the "watch" threshold (0.70)
                    # A healthy node should not stress downstream systems
                    signal = max(0.0, 0.70 - self.agents[pred].health)
                    weight = self.G[pred][node].get("weight", 0.5)
                    incoming.append((signal, weight))
            emitted[node] = self.agents[node].tick(incoming)

        self.cycle += 1

        if self.conn is not None:
            save_snapshot(
                self.conn,
                tail=self.aircraft.tail_number,
                cycle=self.cycle,
                health_states=self.health_states,
                cumulative_cycles=self.aircraft.total_flight_cycles + self.cycle,
                open_deferred=self.aircraft.open_deferred_items,
            )

        return self.health_states

    def run(self, cycles: int) -> list[dict[str, float]]:
        """Run `cycles` steps. Returns list of health state snapshots."""
        snapshots = []
        for _ in range(cycles):
            snapshots.append(self.step())
        return snapshots

    def forward_simulate(
        self,
        horizon: int = DEFAULT_FORWARD_HORIZON,
        branches: int = DEFAULT_MONTE_CARLO_BRANCHES,
    ) -> dict[str, ForwardSimResult]:
        """
        Monte Carlo forward simulation from current health state.

        Runs `branches` independent stochastic simulations for `horizon` cycles.
        For each subsystem, computes:
          - failure_prob: fraction of branches where health < FAILURE_THRESHOLD
          - cascade_chain: most common predecessor path leading to failure
          - expected_cycle: expected cycle of first failure across failing branches
        """
        initial_health = self.health_states.copy()

        # Per-subsystem tracking across branches
        failure_counts: dict[str, int] = {s: 0 for s in SUBSYSTEMS}
        failure_cycles: dict[str, list[float]] = {s: [] for s in SUBSYSTEMS}
        chain_votes: dict[str, Counter] = {s: Counter() for s in SUBSYSTEMS}

        for b in range(branches):
            branch_result = _run_branch(
                initial_health=initial_health,
                G=self.G,
                horizon=horizon,
                rng=np.random.default_rng(self.aircraft.seed * 1000 + b),
            )
            for subsystem, result in branch_result.items():
                if result["failed"]:
                    failure_counts[subsystem] += 1
                    failure_cycles[subsystem].append(result["first_failure_cycle"])
                    chain_key = tuple(result["cascade_path"])
                    chain_votes[subsystem][chain_key] += 1

        results = {}
        for subsystem in SUBSYSTEMS:
            prob = failure_counts[subsystem] / branches
            # Most common cascade chain (or just [subsystem] if isolated failure)
            if chain_votes[subsystem]:
                most_common_chain = list(chain_votes[subsystem].most_common(1)[0][0])
            else:
                most_common_chain = [subsystem]
            # Expected failure cycle
            fc = failure_cycles[subsystem]
            expected_cycle = float(np.mean(fc)) if fc else None

            results[subsystem] = ForwardSimResult(
                subsystem=subsystem,
                failure_prob=round(prob, 4),
                cascade_chain=most_common_chain,
                expected_cycle=round(expected_cycle, 1) if expected_cycle else None,
            )

        return results


def _run_branch(
    initial_health: dict[str, float],
    G: nx.DiGraph,
    horizon: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """
    Single Monte Carlo branch. Pure function — no DB writes.

    Returns per-subsystem dict: {failed, first_failure_cycle, cascade_path}
    """
    from maintenance_sim.config import BASE_DECAY_RATES, NOISE_STD

    # Local mutable health states
    health = {s: initial_health.get(s, 0.85) for s in G.nodes}

    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo_order = list(G.nodes)

    first_failure: dict[str, int | None] = {s: None for s in health}
    # Track which upstream node first caused each failure (for cascade path)
    failure_trigger: dict[str, str | None] = {s: None for s in health}

    for cycle in range(1, horizon + 1):
        for node in topo_order:
            if node not in health:
                continue
            prev_health = health[node]

            incoming = []
            for pred in G.predecessors(node):
                if pred in health:
                    signal = max(0.0, 0.70 - health[pred])
                    weight = G[pred][node].get("weight", 0.5)
                    incoming.append((signal, weight, pred))

            decay = BASE_DECAY_RATES.get(node, 0.004)
            cascade_decay = sum(s * w for s, w, _ in incoming)
            noise = rng.normal(0, NOISE_STD.get(node, 0.008))

            health[node] = float(np.clip(
                prev_health - decay - cascade_decay + noise,
                0.0, 1.0
            ))

            if health[node] < FAILURE_THRESHOLD and first_failure[node] is None:
                first_failure[node] = cycle
                # Find the strongest upstream contributor
                if incoming:
                    strongest = max(incoming, key=lambda x: x[0] * x[1])
                    failure_trigger[node] = strongest[2]

    results = {}
    for s in health:
        failed = first_failure[s] is not None
        cascade_path = _trace_cascade_path(s, failure_trigger, max_depth=5)
        results[s] = {
            "failed": failed,
            "first_failure_cycle": first_failure[s] if failed else None,
            "cascade_path": cascade_path,
        }

    return results


def _trace_cascade_path(
    subsystem: str,
    failure_trigger: dict[str, str | None],
    max_depth: int = 5,
) -> list[str]:
    """Trace the upstream cascade path that led to subsystem failure."""
    path = [subsystem]
    current = subsystem
    seen = {subsystem}
    for _ in range(max_depth):
        trigger = failure_trigger.get(current)
        if trigger is None or trigger in seen:
            break
        path.insert(0, trigger)
        seen.add(trigger)
        current = trigger
    return path
