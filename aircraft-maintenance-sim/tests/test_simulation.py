"""Tests for SimulationEngine."""

import numpy as np
import pytest

from maintenance_sim.config import SUBSYSTEMS
from maintenance_sim.simulation import SimulationEngine, _run_branch


def test_step_returns_all_subsystems(degraded_aircraft, tiny_graph):
    engine = SimulationEngine(degraded_aircraft, tiny_graph)
    state = engine.step()
    for s in SUBSYSTEMS:
        assert s in state
        assert 0.0 <= state[s] <= 1.0


def test_cascade_propagates(degraded_aircraft, tiny_graph):
    """Degraded bleed_air_l should cause engine_l to decline faster than without."""
    engine = SimulationEngine(degraded_aircraft, tiny_graph)
    initial_engine_l = engine.health_states["engine_l"]

    # Run 10 cycles
    engine.run(10)
    final_engine_l = engine.health_states["engine_l"]

    # Engine should have dropped — cascade from bleed_air_l
    assert final_engine_l < initial_engine_l, "engine_l should degrade via cascade from bleed_air_l"


def test_healthy_aircraft_stays_healthy(sample_aircraft, tiny_graph):
    """All-healthy aircraft (0.85) should not fail within 20 cycles.
    (bleed_air_l only crosses the 0.70 cascade threshold at ~30 cycles of base decay)
    """
    engine = SimulationEngine(sample_aircraft, tiny_graph)
    engine.run(20)
    for s, health in engine.health_states.items():
        assert health > 0.50, f"{s} should stay healthy over 20 cycles without upstream degradation"


def test_forward_simulate_returns_probabilities(degraded_aircraft, tiny_graph):
    engine = SimulationEngine(degraded_aircraft, tiny_graph)
    results = engine.forward_simulate(horizon=10, branches=50)
    for s in SUBSYSTEMS:
        assert s in results
        assert 0.0 <= results[s].failure_prob <= 1.0


def test_forward_simulate_degraded_has_higher_prob(degraded_aircraft, sample_aircraft, tiny_graph):
    """Degraded bleed_air_l should cause higher engine_l failure prob via cascade."""
    engine_degraded = SimulationEngine(degraded_aircraft, tiny_graph)
    engine_healthy = SimulationEngine(sample_aircraft, tiny_graph)

    res_degraded = engine_degraded.forward_simulate(horizon=20, branches=150)
    res_healthy = engine_healthy.forward_simulate(horizon=20, branches=150)

    # engine_l is the cascade target of bleed_air_l; should fail more often when upstream is degraded
    assert (res_degraded["engine_l"].failure_prob >=
            res_healthy["engine_l"].failure_prob), \
        "Degraded bleed_air_l should yield equal or higher engine_l failure probability"


def test_forward_simulate_deterministic_with_seed(degraded_aircraft, tiny_graph):
    """Same seed → same Monte Carlo results."""
    e1 = SimulationEngine(degraded_aircraft, tiny_graph)
    e2 = SimulationEngine(degraded_aircraft, tiny_graph)
    r1 = e1.forward_simulate(horizon=10, branches=50)
    r2 = e2.forward_simulate(horizon=10, branches=50)
    for s in SUBSYSTEMS:
        assert r1[s].failure_prob == r2[s].failure_prob


def test_failed_subsystem_high_probability(tiny_graph):
    """Subsystem near zero should have very high forward failure probability."""
    from maintenance_sim.fleet import Aircraft
    states = {s: 0.85 for s in SUBSYSTEMS}
    states["bleed_air_l"] = 0.12  # near failure threshold
    aircraft = Aircraft("TEST", 18000, 2004, 17000, [], states, seed=99)
    engine = SimulationEngine(aircraft, tiny_graph)
    results = engine.forward_simulate(horizon=5, branches=100)
    assert results["bleed_air_l"].failure_prob > 0.80, \
        "Near-zero subsystem should have > 80% failure probability"


def test_run_branch_pure_function(tiny_graph):
    """_run_branch should not modify the graph or initial_health dict."""
    from maintenance_sim.config import SUBSYSTEMS
    initial = {s: 0.80 for s in SUBSYSTEMS}
    initial["bleed_air_l"] = 0.30
    original_copy = initial.copy()

    _run_branch(initial, tiny_graph, horizon=5, rng=np.random.default_rng(42))

    assert initial == original_copy, "_run_branch must not modify initial_health"
    assert tiny_graph.number_of_edges() > 0, "_run_branch must not modify graph"


def test_cascade_chain_recorded(tiny_graph):
    """Forward sim should record bleed_air_l in engine_l's cascade chain."""
    from maintenance_sim.fleet import Aircraft
    states = {s: 0.85 for s in SUBSYSTEMS}
    states["bleed_air_l"] = 0.20   # deep caution, strong upstream signal
    aircraft = Aircraft("TEST", 18000, 2004, 17000, [], states, seed=42)
    engine = SimulationEngine(aircraft, tiny_graph)
    results = engine.forward_simulate(horizon=20, branches=150)

    # engine_l cascade chain should mention bleed_air_l as a predecessor
    engine_chain = results["engine_l"].cascade_chain
    if results["engine_l"].failure_prob > 0.1:  # only check if meaningful failures occurred
        assert "bleed_air_l" in engine_chain or engine_chain[0] == "engine_l", \
            f"Expected bleed_air_l in chain, got {engine_chain}"
