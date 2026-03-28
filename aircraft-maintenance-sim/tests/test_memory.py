"""Tests for memory / SQLite persistence module."""

import json
from pathlib import Path

import pytest

from maintenance_sim.memory import (
    get_effective_weights,
    init_db,
    load_biography,
    record_feedback,
    record_prediction,
    save_snapshot,
    update_agent_weights,
)


def test_init_db_creates_tables(in_memory_db):
    tables = {
        row[0] for row in
        in_memory_db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    for expected in ("aircraft_memory", "fault_events", "predictions", "feedback", "agent_weights"):
        assert expected in tables


def test_save_and_load_snapshot(in_memory_db):
    health = {"engine_l": 0.85, "bleed_air_l": 0.45}
    save_snapshot(in_memory_db, "N823AN", cycle=10, health_states=health,
                  cumulative_cycles=18460, open_deferred=["36-11"])
    bio = load_biography(in_memory_db, "N823AN")
    snaps = bio["snapshots"]
    assert len(snaps) == 2  # one row per subsystem
    subsystems_saved = {s["subsystem"] for s in snaps}
    assert "engine_l" in subsystems_saved
    assert "bleed_air_l" in subsystems_saved


def test_record_and_load_prediction(in_memory_db):
    pred_id = record_prediction(
        in_memory_db, "N823AN", cycle=10, horizon=10,
        subsystem="engine_l", prob=0.72,
        chain=["bleed_air_l", "engine_l"],
        recommendation="Inspect bleed duct.",
    )
    assert pred_id is not None and pred_id > 0

    bio = load_biography(in_memory_db, "N823AN")
    preds = bio["predictions"]
    assert len(preds) == 1
    assert preds[0]["subsystem"] == "engine_l"
    assert abs(preds[0]["failure_probability"] - 0.72) < 1e-6


def test_weight_update_confirmed_increases(in_memory_db):
    update_agent_weights(in_memory_db, "N823AN", "engine_l", "bleed_air_l", "confirmed")
    row = in_memory_db.execute(
        "SELECT weight FROM agent_weights WHERE tail_number='N823AN' AND subsystem='engine_l'"
    ).fetchone()
    assert row is not None
    assert row["weight"] > 0.5  # confirmed nudges up from 0.5 default


def test_weight_update_denied_decreases(in_memory_db):
    update_agent_weights(in_memory_db, "N823AN", "engine_l", "bleed_air_l", "denied")
    row = in_memory_db.execute(
        "SELECT weight FROM agent_weights WHERE tail_number='N823AN' AND subsystem='engine_l'"
    ).fetchone()
    assert row["weight"] < 0.5  # denied nudges down from 0.5 default


def test_weight_bounded_after_many_confirmations(in_memory_db):
    for _ in range(200):
        update_agent_weights(in_memory_db, "N823AN", "engine_l", "bleed_air_l", "confirmed")
    row = in_memory_db.execute(
        "SELECT weight FROM agent_weights WHERE tail_number='N823AN' AND subsystem='engine_l'"
    ).fetchone()
    assert row["weight"] <= 0.95


def test_weight_bounded_after_many_denials(in_memory_db):
    for _ in range(200):
        update_agent_weights(in_memory_db, "N823AN", "engine_l", "bleed_air_l", "denied")
    row = in_memory_db.execute(
        "SELECT weight FROM agent_weights WHERE tail_number='N823AN' AND subsystem='engine_l'"
    ).fetchone()
    assert row["weight"] >= 0.05


def test_effective_weights_overrides_graph(in_memory_db, tiny_graph):
    # Record a confirmed outcome to push bleed_air_l→engine_l weight up
    for _ in range(10):
        update_agent_weights(in_memory_db, "N823AN", "engine_l", "bleed_air_l", "confirmed")

    G_eff = get_effective_weights(in_memory_db, "N823AN", tiny_graph)
    original_w = tiny_graph["bleed_air_l"]["engine_l"]["weight"]
    effective_w = G_eff["bleed_air_l"]["engine_l"]["weight"]
    assert effective_w > original_w, "Confirmed feedback should increase edge weight"


def test_record_feedback_updates_weights(in_memory_db):
    pred_id = record_prediction(
        in_memory_db, "N823AN", cycle=10, horizon=10,
        subsystem="engine_l", prob=0.72,
        chain=["bleed_air_l", "engine_l"],
    )
    record_feedback(in_memory_db, pred_id, "N823AN", "engine_l", "confirmed")
    row = in_memory_db.execute(
        "SELECT weight FROM agent_weights WHERE tail_number='N823AN'"
    ).fetchone()
    assert row is not None, "Feedback should have created an agent_weights row"
