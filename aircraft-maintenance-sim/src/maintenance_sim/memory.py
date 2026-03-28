"""
SQLite persistence layer: aircraft biography, predictions, feedback, and
per-tail learned edge weights.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from maintenance_sim.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS aircraft_memory (
    tail_number     TEXT NOT NULL,
    snapshot_cycle  INTEGER NOT NULL,
    subsystem       TEXT NOT NULL,
    health_state    REAL NOT NULL,
    cumulative_cycles INTEGER NOT NULL,
    open_deferred   TEXT,
    last_updated    TEXT NOT NULL,
    PRIMARY KEY (tail_number, snapshot_cycle, subsystem)
);

CREATE TABLE IF NOT EXISTS fault_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tail_number     TEXT NOT NULL,
    subsystem       TEXT NOT NULL,
    detected_cycle  INTEGER NOT NULL,
    health_at_detection REAL NOT NULL,
    cascade_source  TEXT,
    event_type      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    tail_number     TEXT NOT NULL,
    predicted_at_cycle INTEGER NOT NULL,
    horizon_cycles  INTEGER NOT NULL,
    subsystem       TEXT NOT NULL,
    failure_probability REAL NOT NULL,
    cascade_chain   TEXT NOT NULL,
    llm_recommendation TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id   INTEGER REFERENCES predictions(prediction_id),
    tail_number     TEXT NOT NULL,
    subsystem       TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    actual_cycle    INTEGER,
    technician_notes TEXT,
    recorded_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_weights (
    tail_number     TEXT NOT NULL,
    subsystem       TEXT NOT NULL,
    edge_source     TEXT NOT NULL,
    weight          REAL NOT NULL,
    confirmation_count INTEGER DEFAULT 0,
    denial_count    INTEGER DEFAULT 0,
    last_updated    TEXT NOT NULL,
    PRIMARY KEY (tail_number, subsystem, edge_source)
);
"""

_LEARNING_RATE = 0.05
_WEIGHT_MIN = 0.05
_WEIGHT_MAX = 0.95


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create or open the SQLite database and apply schema."""
    if str(db_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_snapshot(conn: sqlite3.Connection, tail: str, cycle: int,
                   health_states: dict[str, float], cumulative_cycles: int,
                   open_deferred: list[str]) -> None:
    """Upsert current health states into aircraft_memory."""
    now = _now()
    conn.executemany(
        """INSERT OR REPLACE INTO aircraft_memory
           (tail_number, snapshot_cycle, subsystem, health_state,
            cumulative_cycles, open_deferred, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (tail, cycle, subsystem, health, cumulative_cycles,
             json.dumps(open_deferred), now)
            for subsystem, health in health_states.items()
        ],
    )
    conn.commit()


def load_biography(conn: sqlite3.Connection, tail: str) -> dict:
    """Return a summary of the aircraft's full history."""
    snapshots = conn.execute(
        "SELECT * FROM aircraft_memory WHERE tail_number = ? ORDER BY snapshot_cycle",
        (tail,)
    ).fetchall()

    predictions = conn.execute(
        "SELECT * FROM predictions WHERE tail_number = ? ORDER BY created_at DESC LIMIT 20",
        (tail,)
    ).fetchall()

    feedback = conn.execute(
        """SELECT f.*, p.subsystem as pred_subsystem
           FROM feedback f
           JOIN predictions p ON f.prediction_id = p.prediction_id
           WHERE f.tail_number = ?
           ORDER BY f.recorded_at DESC LIMIT 10""",
        (tail,)
    ).fetchall()

    return {
        "tail": tail,
        "snapshots": [dict(r) for r in snapshots],
        "predictions": [dict(r) for r in predictions],
        "feedback": [dict(r) for r in feedback],
    }


def record_prediction(
    conn: sqlite3.Connection,
    tail: str,
    cycle: int,
    horizon: int,
    subsystem: str,
    prob: float,
    chain: list[str],
    recommendation: str = "",
) -> int:
    """Insert a prediction row and return its prediction_id."""
    cur = conn.execute(
        """INSERT INTO predictions
           (tail_number, predicted_at_cycle, horizon_cycles, subsystem,
            failure_probability, cascade_chain, llm_recommendation, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (tail, cycle, horizon, subsystem, prob,
         json.dumps(chain), recommendation, _now()),
    )
    conn.commit()
    return cur.lastrowid


def record_feedback(
    conn: sqlite3.Connection,
    prediction_id: int,
    tail: str,
    subsystem: str,
    outcome: str,
    actual_cycle: int | None = None,
    notes: str = "",
) -> None:
    """Record technician outcome and update learned weights."""
    conn.execute(
        """INSERT INTO feedback
           (prediction_id, tail_number, subsystem, outcome,
            actual_cycle, technician_notes, recorded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (prediction_id, tail, subsystem, outcome, actual_cycle, notes, _now()),
    )
    conn.commit()

    # Fetch the cascade chain to know which edge to update
    row = conn.execute(
        "SELECT cascade_chain FROM predictions WHERE prediction_id = ?",
        (prediction_id,)
    ).fetchone()
    if row:
        chain = json.loads(row["cascade_chain"])
        # Update weights for all edges in the chain
        for i in range(len(chain) - 1):
            update_agent_weights(conn, tail, chain[i + 1], chain[i], outcome)


def update_agent_weights(
    conn: sqlite3.Connection,
    tail: str,
    subsystem: str,
    edge_source: str,
    outcome: str,
) -> None:
    """
    Multiplicative Bayesian weight update per tail.

    confirmed: w = w + lr * (1 - w)  → nudge toward 1.0
    denied:    w = w - lr * w         → nudge toward 0.0
    Clamped to [WEIGHT_MIN, WEIGHT_MAX].
    """
    # Get current weight (falls back to None if no row yet)
    row = conn.execute(
        """SELECT weight, confirmation_count, denial_count
           FROM agent_weights
           WHERE tail_number = ? AND subsystem = ? AND edge_source = ?""",
        (tail, subsystem, edge_source),
    ).fetchone()

    if row is None:
        # Initialise from knowledge graph default (0.5 if unknown)
        current_weight = 0.5
        confirm_count = 0
        deny_count = 0
    else:
        current_weight = row["weight"]
        confirm_count = row["confirmation_count"]
        deny_count = row["denial_count"]

    if outcome == "confirmed":
        new_weight = current_weight + _LEARNING_RATE * (1 - current_weight)
        confirm_count += 1
    elif outcome == "denied":
        new_weight = current_weight - _LEARNING_RATE * current_weight
        deny_count += 1
    else:
        new_weight = current_weight  # "partial" — no update

    new_weight = float(max(_WEIGHT_MIN, min(_WEIGHT_MAX, new_weight)))

    conn.execute(
        """INSERT OR REPLACE INTO agent_weights
           (tail_number, subsystem, edge_source, weight,
            confirmation_count, denial_count, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tail, subsystem, edge_source, new_weight,
         confirm_count, deny_count, _now()),
    )
    conn.commit()


def get_effective_weights(
    conn: sqlite3.Connection, tail: str, G: nx.DiGraph
) -> nx.DiGraph:
    """
    Return a copy of G with edge weights overridden by any tail-specific
    learned weights from the agent_weights table.
    """
    G_copy = G.copy()
    rows = conn.execute(
        "SELECT subsystem, edge_source, weight FROM agent_weights WHERE tail_number = ?",
        (tail,)
    ).fetchall()
    for row in rows:
        src, tgt, w = row["edge_source"], row["subsystem"], row["weight"]
        if G_copy.has_edge(src, tgt):
            G_copy[src][tgt]["weight"] = w
    return G_copy


def get_recent_predictions(
    conn: sqlite3.Connection, tail: str, limit: int = 10
) -> list[dict]:
    """Return most recent predictions for a tail, with feedback if available."""
    rows = conn.execute(
        """SELECT p.*, f.outcome, f.technician_notes
           FROM predictions p
           LEFT JOIN feedback f ON p.prediction_id = f.prediction_id
           WHERE p.tail_number = ?
           ORDER BY p.created_at DESC
           LIMIT ?""",
        (tail, limit),
    ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["cascade_chain"] = json.loads(d["cascade_chain"])
        results.append(d)
    return results


def get_all_predictions(conn: sqlite3.Connection) -> list[dict]:
    """Return all predictions across all tails for the feedback UI."""
    rows = conn.execute(
        """SELECT p.prediction_id, p.tail_number, p.subsystem,
                  p.failure_probability, p.predicted_at_cycle,
                  p.cascade_chain, p.created_at, f.outcome
           FROM predictions p
           LEFT JOIN feedback f ON p.prediction_id = f.prediction_id
           ORDER BY p.created_at DESC"""
    ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["cascade_chain"] = json.loads(d["cascade_chain"])
        results.append(d)
    return results
