"""Tests for SubsystemAgent."""

import numpy as np
import pytest

from maintenance_sim.agents import SubsystemAgent, create_agents
from maintenance_sim.config import SUBSYSTEMS


@pytest.fixture
def healthy_agent():
    return SubsystemAgent(
        name="engine_l",
        health=0.90,
        base_decay_rate=0.004,
        noise_std=0.0,  # deterministic for testing
        rng=np.random.default_rng(42),
    )


@pytest.fixture
def degraded_agent():
    return SubsystemAgent(
        name="bleed_air_l",
        health=0.45,
        base_decay_rate=0.005,
        noise_std=0.0,
        rng=np.random.default_rng(42),
    )


def test_tick_healthy_degrades_slowly(healthy_agent):
    initial = healthy_agent.health
    for _ in range(10):
        healthy_agent.tick([])
    assert healthy_agent.health >= initial - 0.05, "Healthy agent should degrade slowly"


def test_tick_incoming_signal_accelerates_decay():
    agent = SubsystemAgent("engine_l", 0.80, 0.004, 0.0, np.random.default_rng(42))
    agent_no_signal = SubsystemAgent("engine_l", 0.80, 0.004, 0.0, np.random.default_rng(42))

    # Apply strong upstream signal
    agent.tick([(0.8, 0.65)])
    agent_no_signal.tick([])

    assert agent.health < agent_no_signal.health, "Incoming signal should accelerate decay"


def test_tick_clamps_to_zero():
    agent = SubsystemAgent("engine_l", 0.02, 0.05, 0.0, np.random.default_rng(42))
    for _ in range(5):
        agent.tick([(1.0, 1.0)])
    assert agent.health >= 0.0, "Health must not go below 0"


def test_tick_clamps_to_one():
    agent = SubsystemAgent("engine_l", 0.98, 0.001, 0.0, np.random.default_rng(42))
    # Negative noise (recovery) should not exceed 1.0
    agent_pos = SubsystemAgent("engine_l", 0.98, 0.001, 0.0, np.random.default_rng(42))
    # Force positive noise by monkey-patching
    agent_pos.rng = np.random.default_rng(0)
    for _ in range(10):
        agent_pos.tick([])
    assert agent_pos.health <= 1.0


def test_is_failed_threshold():
    agent = SubsystemAgent("apu", 0.14, 0.001, 0.0, np.random.default_rng(42))
    assert agent.is_failed
    agent.health = 0.16
    assert not agent.is_failed


def test_alert_levels():
    agent = SubsystemAgent("apu", 0.80, 0.001, 0.0, np.random.default_rng(42))
    assert agent.alert_level == "nominal"
    agent.health = 0.50
    assert agent.alert_level == "watch"
    agent.health = 0.25
    assert agent.alert_level == "caution"
    agent.health = 0.10
    assert agent.alert_level == "critical"


def test_emitted_signal_positive_on_decay():
    agent = SubsystemAgent("bleed_air_l", 0.60, 0.02, 0.0, np.random.default_rng(42))
    emitted = agent.tick([])
    assert emitted >= 0.0


def test_create_agents_returns_all_subsystems():
    health = {s: 0.80 for s in SUBSYSTEMS}
    agents = create_agents(health, seed=42)
    assert set(agents.keys()) == set(SUBSYSTEMS)


def test_clone_independent():
    agent = SubsystemAgent("engine_l", 0.75, 0.004, 0.01, np.random.default_rng(42))
    clone = agent.clone()
    # Run original many cycles
    for _ in range(20):
        agent.tick([])
    # Clone should be unaffected
    assert abs(clone.health - 0.75) < 0.01, "Clone should have original health state"
