"""
Build and manage the aircraft system dependency knowledge graph.

The graph is a directed weighted NetworkX DiGraph where:
  - Nodes are subsystem names (from config.SUBSYSTEMS)
  - Edges represent degradation propagation: A → B means
    degradation in A increases failure risk in B
  - Edge weight (0–1) quantifies propagation strength
"""

import csv
import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from maintenance_sim.config import PROCESSED_DIR, SEEDS_DIR, SUBSYSTEMS

# Fallback edges if no files are available (ensures offline demo always works)
_DOMAIN_FALLBACK_EDGES = [
    ("bleed_air_l", "engine_l",      0.65, "2009-03-01", "75/72"),
    ("bleed_air_r", "engine_r",      0.65, "2009-03-01", "75/72"),
    ("apu",         "bleed_air_l",   0.45, "2014-09-08", "49/36"),
    ("apu",         "bleed_air_r",   0.45, "2014-09-08", "49/36"),
    ("engine_l",    "hydraulics",    0.40, "",            "71/29"),
    ("engine_r",    "hydraulics",    0.40, "",            "71/29"),
    ("hydraulics",  "flight_controls", 0.70, "2021-16972", "29/27"),
    ("hydraulics",  "landing_gear",  0.55, "",            "29/32"),
    ("bleed_air_l", "avionics",      0.30, "",            "36/24"),
    ("bleed_air_r", "avionics",      0.30, "",            "36/24"),
    ("engine_l",    "bleed_air_l",   0.35, "",            "72/36"),
    ("engine_r",    "bleed_air_r",   0.35, "",            "72/36"),
]


def _empty_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for s in SUBSYSTEMS:
        G.add_node(s)
    return G


def _load_ad_edges(path: Path) -> list[tuple[str, str, dict]]:
    edges = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src, tgt = row["source"].strip(), row["target"].strip()
            if src in SUBSYSTEMS and tgt in SUBSYSTEMS:
                edges.append((src, tgt, {
                    "weight": float(row["weight"]),
                    "ad_number": row.get("ad_number", ""),
                    "ata_chapter": row.get("ata_chapter", ""),
                    "confidence": "ad_mandated",
                }))
    return edges


def build_graph(
    ad_edges_path: Path | None = None,
    graph_json_path: Path | None = None,
) -> nx.DiGraph:
    """
    Build the 737-800 subsystem dependency graph.

    Priority order:
      1. Cached knowledge_graph.json (produced by prepare_data.py)
      2. ad_edges.csv seed file
      3. Hard-coded domain fallback (always works offline)
    """
    # Try cached graph first
    cached = graph_json_path or (PROCESSED_DIR / "knowledge_graph.json")
    if cached.exists():
        return load_graph(cached)

    G = _empty_graph()
    seen: set[tuple[str, str]] = set()

    # Load AD edges
    ad_path = ad_edges_path or (SEEDS_DIR / "ad_edges.csv")
    if ad_path.exists():
        for src, tgt, attrs in _load_ad_edges(ad_path):
            G.add_edge(src, tgt, **attrs)
            seen.add((src, tgt))
    else:
        # Use hardcoded fallback
        for src, tgt, w, ad_num, ata in _DOMAIN_FALLBACK_EDGES:
            G.add_edge(src, tgt,
                       weight=w, ad_number=ad_num,
                       ata_chapter=ata, confidence="ad_mandated")
            seen.add((src, tgt))

    return G


def load_graph(path: Path) -> nx.DiGraph:
    """Deserialize graph from JSON (node_link_data format)."""
    with open(path) as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data, directed=True, multigraph=False)
    # Ensure all subsystems are present as nodes
    for s in SUBSYSTEMS:
        if s not in G:
            G.add_node(s)
    return G


def save_graph(G: nx.DiGraph, path: Path) -> None:
    """Serialize graph to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_graph.node_link_data(G), f, indent=2)


def get_cascade_description(G: nx.DiGraph, source: str, depth: int = 3) -> list[str]:
    """
    Return a list of subsystems reachable from `source` within `depth` hops,
    ordered by cumulative path weight (strongest cascades first).
    """
    paths = []
    for target in G.nodes:
        if target == source:
            continue
        try:
            path = nx.shortest_path(G, source, target)
            if 1 < len(path) <= depth + 1:
                weight = 1.0
                for i in range(len(path) - 1):
                    weight *= G[path[i]][path[i + 1]].get("weight", 0.5)
                paths.append((target, weight, path))
        except nx.NetworkXNoPath:
            continue

    paths.sort(key=lambda x: -x[1])
    return [p[2] for p in paths]


def get_edge_context(G: nx.DiGraph, source: str, target: str) -> dict:
    """Return edge metadata for use in LLM prompts."""
    if G.has_edge(source, target):
        return dict(G[source][target])
    return {}
