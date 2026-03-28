"""Tests for validation module."""

import pandas as pd
import pytest

from maintenance_sim.knowledge_graph import build_graph
from maintenance_sim.validation import (
    ValidationMetrics,
    build_actual_cascades,
    format_validation_table,
    run_validation,
    threshold_baseline,
)


@pytest.fixture
def mini_test_df():
    """Minimal synthetic test DataFrame."""
    return pd.DataFrame({
        "subsystem": ["bleed_air_l", "engine_l", "hydraulics", "flight_controls",
                      "apu", "bleed_air_r"],
        "difficulty_date": pd.to_datetime([
            "2022-03-01", "2022-03-15", "2022-04-01", "2022-04-10",
            "2022-05-01", "2022-05-20"
        ]),
        "difficulty_year": [2022] * 6,
    })


@pytest.fixture
def mini_train_df():
    """Minimal synthetic train DataFrame."""
    rows = []
    subsystems = ["bleed_air_l"] * 50 + ["engine_l"] * 40 + ["hydraulics"] * 30
    for i, s in enumerate(subsystems):
        rows.append({
            "subsystem": s,
            "difficulty_year": 2020,
            "difficulty_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
        })
    return pd.DataFrame(rows)


def test_build_actual_cascades_returns_list(mini_test_df):
    G = build_graph()
    cascades = build_actual_cascades(mini_test_df, G)
    assert isinstance(cascades, list)
    assert len(cascades) > 0


def test_cascade_pairs_have_graph_edges(mini_test_df):
    G = build_graph()
    cascades = build_actual_cascades(mini_test_df, G)
    for c in cascades:
        assert G.has_edge(c["first_subsystem"], c["second_subsystem"]), \
            f"Cascade pair ({c['first_subsystem']}, {c['second_subsystem']}) has no graph edge"


def test_threshold_baseline_returns_float(mini_train_df):
    G = build_graph()
    cascades = build_actual_cascades(pd.DataFrame(), G)
    result = threshold_baseline(cascades, mini_train_df)
    assert 0.0 <= result <= 1.0


def test_temporal_split_no_leakage(mini_test_df, mini_train_df):
    train_years = set(mini_train_df["difficulty_year"].unique())
    test_years = set(mini_test_df["difficulty_year"].unique())
    assert not train_years.intersection(test_years), "Train/test years must not overlap"


def test_run_validation_returns_metrics():
    metrics = run_validation(
        test_df=pd.DataFrame(),
        train_df=pd.DataFrame(),
        horizon=5,
        branches=30,
    )
    assert isinstance(metrics, ValidationMetrics)
    assert 0.0 <= metrics.cascade_recall <= 1.0
    assert 0.0 <= metrics.threshold_recall <= 1.0
    assert 0.0 <= metrics.precision <= 1.0


def test_format_validation_table():
    metrics = ValidationMetrics(
        cascade_recall=0.74,
        threshold_recall=0.41,
        precision=0.61,
        cascade_delta=0.33,
        avg_detection_lead_days=8.3,
        n_cascades_found=30,
        n_total_cascades=40,
    )
    table = format_validation_table(metrics)
    assert len(table) == 4
    metrics_names = [row["Metric"] for row in table]
    assert "Cascade Recall" in metrics_names
    assert "Cascade Delta (↑)" in metrics_names
    # Cascade model should show higher recall
    recall_row = next(r for r in table if r["Metric"] == "Cascade Recall")
    assert "74%" in recall_row["Cascade Model"]
