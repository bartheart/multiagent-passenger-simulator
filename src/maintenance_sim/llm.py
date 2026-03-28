"""
LLM integration: generates structured maintenance recommendations via Claude.

Uses tool_use with forced tool selection to guarantee schema-compliant output.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024


class MaintenanceRecommendation(BaseModel):
    tail_number: str
    urgency: str            # "immediate" | "within_3_cycles" | "within_10_cycles" | "monitor"
    primary_subsystem: str
    cascade_chain: list[str]
    recommended_action: str   # one clear sentence for the technician
    reasoning: str            # 2–3 sentence explanation
    ata_references: list[str] # e.g. ["ATA 36-11", "ATA 71-00"]
    estimated_aog_risk: str   # "high" | "medium" | "low"


_TOOL_SCHEMA = {
    "name": "submit_recommendation",
    "description": (
        "Submit a structured maintenance recommendation based on simulation results. "
        "Fill every field accurately based on the provided data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tail_number": {"type": "string"},
            "urgency": {
                "type": "string",
                "enum": ["immediate", "within_3_cycles", "within_10_cycles", "monitor"],
            },
            "primary_subsystem": {"type": "string"},
            "cascade_chain": {"type": "array", "items": {"type": "string"}},
            "recommended_action": {"type": "string"},
            "reasoning": {"type": "string"},
            "ata_references": {"type": "array", "items": {"type": "string"}},
            "estimated_aog_risk": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": [
            "tail_number", "urgency", "primary_subsystem", "cascade_chain",
            "recommended_action", "reasoning", "ata_references", "estimated_aog_risk",
        ],
    },
}

_SYSTEM_PROMPT = (
    "You are an expert Boeing 737-800 maintenance analyst with 20 years of experience "
    "interpreting predictive maintenance signals and cascade failure patterns. "
    "You work within FAR 121 maintenance requirements and airline operational constraints. "
    "Given simulation output showing subsystem health degradation and cascade propagation "
    "probabilities, produce actionable maintenance recommendations that a line technician "
    "can act on immediately. Be specific about ATA chapters and inspection steps. "
    "Urgency must reflect the failure probability: "
    "P > 0.70 → immediate, P 0.40–0.70 → within_3_cycles, "
    "P 0.20–0.40 → within_10_cycles, P < 0.20 → monitor."
)


def _build_user_message(
    tail: str,
    biography: dict,
    forward_results: dict,
    graph_context: dict,
) -> str:
    """Format simulation results as a structured prompt."""
    # Top 3 subsystems by failure probability
    sorted_results = sorted(
        forward_results.items(),
        key=lambda x: x[1].failure_prob if hasattr(x[1], "failure_prob") else x[1].get("failure_prob", 0),
        reverse=True,
    )[:3]

    result_lines = []
    for subsystem, result in sorted_results:
        if hasattr(result, "failure_prob"):
            prob = result.failure_prob
            chain = result.cascade_chain
            exp_cycle = result.expected_cycle
        else:
            prob = result.get("failure_prob", 0)
            chain = result.get("cascade_chain", [subsystem])
            exp_cycle = result.get("expected_cycle")

        chain_str = " → ".join(chain) if chain else subsystem
        cycle_str = f", expected cycle: {exp_cycle:.1f}" if exp_cycle else ""
        result_lines.append(
            f"  - {subsystem}: failure_prob={prob:.2f}, cascade={chain_str}{cycle_str}"
        )

    # Recent feedback history
    feedback = biography.get("feedback", [])[:5]
    feedback_lines = []
    for fb in feedback:
        outcome = fb.get("outcome", "unknown")
        subsystem = fb.get("subsystem") or fb.get("pred_subsystem", "unknown")
        feedback_lines.append(f"  [{outcome}] {subsystem}")

    # Open deferred items
    snaps = biography.get("snapshots", [])
    deferred = []
    if snaps:
        latest = max(snaps, key=lambda s: s.get("snapshot_cycle", 0))
        raw_deferred = latest.get("open_deferred", "[]")
        try:
            deferred = json.loads(raw_deferred) if isinstance(raw_deferred, str) else raw_deferred
        except (json.JSONDecodeError, TypeError):
            deferred = []

    # Graph edges context
    edge_lines = []
    for (src, tgt), attrs in graph_context.items():
        w = attrs.get("weight", "?")
        ad = attrs.get("ad_number", "")
        ata = attrs.get("ata_chapter", "")
        ad_str = f", AD {ad}" if ad else ""
        ata_str = f", ATA {ata}" if ata else ""
        edge_lines.append(f"  {src} → {tgt} (weight={w}{ad_str}{ata_str})")

    parts = [
        f"AIRCRAFT: {tail}",
        f"OPEN DEFERRED ITEMS: {', '.join(deferred) if deferred else 'none'}",
        "",
        f"FORWARD SIMULATION (next 10 cycles, Monte Carlo):",
    ]
    parts.extend(result_lines)

    if edge_lines:
        parts.append("\nRELEVANT CASCADE EDGES:")
        parts.extend(edge_lines)

    if feedback_lines:
        parts.append("\nPRIOR PREDICTION OUTCOMES (last 5):")
        parts.extend(feedback_lines)

    parts.append(
        "\nPlease submit a structured maintenance recommendation using the "
        "submit_recommendation tool."
    )

    return "\n".join(parts)


def generate_recommendation(
    tail: str,
    biography: dict,
    forward_results: dict,
    graph_context: dict | None = None,
) -> MaintenanceRecommendation:
    """
    Call claude-sonnet-4-6 and return a structured MaintenanceRecommendation.
    graph_context: dict of {(src, tgt): edge_attrs} for relevant edges.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    if graph_context is None:
        graph_context = {}

    user_message = _build_user_message(tail, biography, forward_results, graph_context)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        tools=[_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_recommendation"},
    )

    # Extract tool input — tool_choice forces exactly one tool_use block
    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        # Fallback if API returns unexpected format
        return _fallback_recommendation(tail, forward_results)

    return MaintenanceRecommendation(**tool_block.input)


def _fallback_recommendation(tail: str, forward_results: dict) -> MaintenanceRecommendation:
    """Return a safe fallback if the API call fails."""
    top = max(
        forward_results.items(),
        key=lambda x: x[1].failure_prob if hasattr(x[1], "failure_prob") else 0,
        default=(list(forward_results.keys())[0] if forward_results else "unknown", None),
    )
    subsystem = top[0]
    return MaintenanceRecommendation(
        tail_number=tail,
        urgency="monitor",
        primary_subsystem=subsystem,
        cascade_chain=[subsystem],
        recommended_action=f"Review {subsystem} health data at next scheduled check.",
        reasoning="Recommendation generated from simulation data without LLM analysis.",
        ata_references=[],
        estimated_aog_risk="low",
    )


def build_graph_context(G, cascade_chain: list[str]) -> dict:
    """Extract relevant edge attributes from graph for the prompt."""
    context = {}
    for i in range(len(cascade_chain) - 1):
        src, tgt = cascade_chain[i], cascade_chain[i + 1]
        if G.has_edge(src, tgt):
            context[(src, tgt)] = dict(G[src][tgt])
    return context
