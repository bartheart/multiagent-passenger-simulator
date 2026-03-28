"""Tests for knowledge_graph module."""

import json
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from maintenance_sim.config import SUBSYSTEMS
from maintenance_sim.knowledge_graph import (
    build_graph,
    get_cascade_description,
    load_graph,
    save_graph,
)


def test_build_graph_has_all_subsystems():
    G = build_graph()
    for s in SUBSYSTEMS:
        assert s in G.nodes, f"Missing node: {s}"


def test_build_graph_has_edges():
    G = build_graph()
    assert G.number_of_edges() > 0


def test_known_cascade_edge_exists():
    G = build_graph()
    assert G.has_edge("bleed_air_l", "engine_l"), "Key cascade edge missing"
    assert G.has_edge("hydraulics", "flight_controls"), "Hydraulics cascade edge missing"


def test_edge_weights_bounded():
    G = build_graph()
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0)
        assert 0.0 <= w <= 1.0, f"Edge ({u},{v}) weight {w} out of range"


def test_save_and_load_roundtrip(tiny_graph):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    save_graph(tiny_graph, path)
    loaded = load_graph(path)

    assert set(loaded.nodes) >= set(tiny_graph.nodes)
    for u, v in tiny_graph.edges:
        assert loaded.has_edge(u, v), f"Edge ({u},{v}) lost in roundtrip"
        orig_w = tiny_graph[u][v]["weight"]
        loaded_w = loaded[u][v]["weight"]
        assert abs(orig_w - loaded_w) < 1e-6

    path.unlink()


def test_cascade_description_reachable(tiny_graph):
    chains = get_cascade_description(tiny_graph, "bleed_air_l")
    downstream = [chain[-1] for chain in chains]
    assert "engine_l" in downstream
    assert "hydraulics" in downstream  # 2-hop


def test_cascade_description_excludes_source(tiny_graph):
    chains = get_cascade_description(tiny_graph, "bleed_air_l")
    for chain in chains:
        assert "bleed_air_l" not in chain[1:], "Source should not appear in chain targets"


def test_no_self_loops_in_default_graph():
    G = build_graph()
    for node in G.nodes:
        assert not G.has_edge(node, node), f"Self-loop on {node}"
