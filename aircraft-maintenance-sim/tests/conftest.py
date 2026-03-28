"""Shared pytest fixtures."""

import sqlite3
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from maintenance_sim.config import SUBSYSTEMS
from maintenance_sim.fleet import Aircraft
from maintenance_sim.memory import init_db


@pytest.fixture
def tiny_graph() -> nx.DiGraph:
    """3-node subgraph: bleed_air_l → engine_l → hydraulics."""
    G = nx.DiGraph()
    for s in SUBSYSTEMS:
        G.add_node(s)
    G.add_edge("bleed_air_l", "engine_l",
               weight=0.65, ad_number="2009-03-01", confidence="ad_mandated")
    G.add_edge("engine_l", "hydraulics",
               weight=0.40, ad_number="", confidence="ad_mandated")
    G.add_edge("hydraulics", "flight_controls",
               weight=0.70, ad_number="2021-16972", confidence="ad_mandated")
    return G


@pytest.fixture
def sample_aircraft() -> Aircraft:
    """Single aircraft with known seeded health states."""
    return Aircraft(
        tail_number="N823AN",
        total_flight_cycles=18450,
        manufacture_year=2004,
        last_c_check_cycle=17200,
        open_deferred_items=["36-11"],
        health_states={s: 0.85 for s in SUBSYSTEMS},
        seed=42,
    )


@pytest.fixture
def degraded_aircraft() -> Aircraft:
    """Aircraft with bleed_air_l in caution zone."""
    states = {s: 0.85 for s in SUBSYSTEMS}
    states["bleed_air_l"] = 0.45
    return Aircraft(
        tail_number="N823AN",
        total_flight_cycles=18450,
        manufacture_year=2004,
        last_c_check_cycle=17200,
        open_deferred_items=["36-11"],
        health_states=states,
        seed=42,
    )


@pytest.fixture
def in_memory_db() -> sqlite3.Connection:
    """In-memory SQLite DB with schema initialised."""
    conn = init_db(Path(":memory:"))
    yield conn
    conn.close()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
