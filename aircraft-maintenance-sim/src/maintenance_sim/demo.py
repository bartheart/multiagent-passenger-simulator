"""
Streamlit demo: AA Predictive Maintenance Prototype

Panels:
  1. Fleet Dashboard  — 20 tails, color-coded health table, run simulation
  2. Cascade Detail   — per-tail drill-down, graph viz, Monte Carlo results, Claude card
  3. Validation       — cascade model vs. threshold baseline on held-out data
  4. Feedback Entry   — submit confirmed/denied outcomes → weight update
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from maintenance_sim.config import DB_PATH, SUBSYSTEMS
from maintenance_sim.fleet import create_fleet
from maintenance_sim.knowledge_graph import build_graph
from maintenance_sim.memory import (
    get_all_predictions,
    get_effective_weights,
    init_db,
    record_feedback,
    record_prediction,
)
from maintenance_sim.simulation import SimulationEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AA Predictive Maintenance",
    page_icon="✈",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "db_conn" not in st.session_state:
    st.session_state.db_conn = init_db(DB_PATH)
if "fleet" not in st.session_state:
    st.session_state.fleet = create_fleet()
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "sim_results" not in st.session_state:
    st.session_state.sim_results = {}   # tail → dict of ForwardSimResult
if "recommendations" not in st.session_state:
    st.session_state.recommendations = {}  # tail → MaintenanceRecommendation

conn = st.session_state.db_conn
fleet = st.session_state.fleet
G = st.session_state.graph


# ── Helpers ───────────────────────────────────────────────────────────────────

ALERT_COLORS = {
    "nominal": "#2ecc71",
    "watch": "#f39c12",
    "caution": "#e67e22",
    "critical": "#e74c3c",
}

def _alert_level(health: float) -> str:
    if health >= 0.70: return "nominal"
    if health >= 0.45: return "watch"
    if health >= 0.20: return "caution"
    return "critical"

def _health_color(health: float) -> str:
    return ALERT_COLORS[_alert_level(health)]

def _health_emoji(health: float) -> str:
    level = _alert_level(health)
    return {"nominal": "🟢", "watch": "🟡", "caution": "🟠", "critical": "🔴"}[level]

def _build_fleet_table(fleet) -> pd.DataFrame:
    rows = []
    for ac in fleet:
        row = {
            "Tail": ac.tail_number,
            "Cycles": ac.total_flight_cycles,
            "Since C-Check": ac.cycles_since_c_check,
        }
        for s in SUBSYSTEMS:
            h = ac.health_states.get(s, 1.0)
            row[s] = f"{_health_emoji(h)} {h:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)

def _run_forward_sim(aircraft, branches: int = 200) -> dict:
    G_eff = get_effective_weights(conn, aircraft.tail_number, G)
    engine = SimulationEngine(aircraft, G_eff)
    return engine.forward_simulate(horizon=10, branches=branches)

def _draw_graph(aircraft, forward_results) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    pos = nx.spring_layout(G, seed=42, k=2.5)

    node_colors = []
    for node in G.nodes:
        health = aircraft.health_states.get(node, 1.0)
        node_colors.append(_health_color(health))

    # Highlight cascade chain edges
    top_result = max(forward_results.values(), key=lambda r: r.failure_prob)
    cascade_edges = set()
    chain = top_result.cascade_chain
    for i in range(len(chain) - 1):
        cascade_edges.add((chain[i], chain[i + 1]))

    edge_colors = []
    edge_widths = []
    for u, v in G.edges:
        if (u, v) in cascade_edges:
            edge_colors.append("#e74c3c")
            edge_widths.append(3.0)
        else:
            edge_colors.append("#555555")
            edge_widths.append(1.0)

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors, node_size=1200,
        edge_color=edge_colors, width=edge_widths,
        font_color="white", font_size=8,
        arrows=True, arrowsize=15,
    )
    ax.set_title(
        f"{aircraft.tail_number} — System Dependency Graph",
        color="white", fontsize=12, pad=10
    )
    plt.tight_layout()
    return fig

def _render_recommendation_card(rec):
    urgency_colors = {
        "immediate": "#e74c3c",
        "within_3_cycles": "#e67e22",
        "within_10_cycles": "#f39c12",
        "monitor": "#2ecc71",
    }
    color = urgency_colors.get(rec.urgency, "#888")
    st.markdown(f"""
<div style="border-left: 4px solid {color}; padding: 12px 16px;
            background: #1a1a2e; border-radius: 4px; margin: 8px 0;">
  <div style="color:{color}; font-weight:bold; font-size:14px; margin-bottom:6px;">
    ⚠ {rec.urgency.replace('_', ' ').upper()} — {rec.primary_subsystem.replace('_', ' ').upper()}
    &nbsp;|&nbsp; AOG Risk: {rec.estimated_aog_risk.upper()}
  </div>
  <div style="color:#fff; font-size:14px; margin-bottom:8px;">
    <strong>Action:</strong> {rec.recommended_action}
  </div>
  <div style="color:#aaa; font-size:13px; margin-bottom:6px;">
    {rec.reasoning}
  </div>
  <div style="color:#888; font-size:12px;">
    ATA refs: {', '.join(rec.ata_references) if rec.ata_references else '—'}
    &nbsp;|&nbsp; Cascade: {' → '.join(rec.cascade_chain)}
  </div>
</div>
""", unsafe_allow_html=True)


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("✈ AA Predictive Maintenance — Multi-Agent Cascade Simulation")

tab1, tab2, tab3, tab4 = st.tabs([
    "🛩 Fleet Dashboard",
    "🔍 Cascade Detail",
    "📊 Validation",
    "✅ Feedback",
])

# ─── Panel 1: Fleet Dashboard ─────────────────────────────────────────────────
with tab1:
    st.subheader("Fleet Health Overview — 20 Boeing 737-800 Aircraft")

    col_run, col_branches = st.columns([2, 1])
    with col_branches:
        branches = st.slider("Monte Carlo branches", 50, 500, 200, 50)
    with col_run:
        run_sim = st.button("▶ Run Forward Simulation (10 cycles)", type="primary")

    if run_sim:
        prog = st.progress(0, text="Running simulation...")
        for i, ac in enumerate(fleet):
            results = _run_forward_sim(ac, branches=branches)
            st.session_state.sim_results[ac.tail_number] = results
            # Save top predictions to DB
            top = max(results.values(), key=lambda r: r.failure_prob)
            if top.failure_prob > 0.20:
                record_prediction(
                    conn, ac.tail_number, cycle=0, horizon=10,
                    subsystem=top.subsystem,
                    prob=top.failure_prob,
                    chain=top.cascade_chain,
                )
            prog.progress((i + 1) / len(fleet),
                          text=f"Simulating {ac.tail_number}... ({i+1}/{len(fleet)})")
        prog.empty()
        st.success("Simulation complete.")

    # Fleet table
    df = _build_fleet_table(fleet)

    # Augment with risk score if simulation has run
    if st.session_state.sim_results:
        risk_scores = []
        for ac in fleet:
            res = st.session_state.sim_results.get(ac.tail_number, {})
            if res:
                max_prob = max(r.failure_prob for r in res.values())
            else:
                max_prob = 0.0
            risk_scores.append(f"{_health_emoji(1 - max_prob)} {max_prob:.0%}")
        df.insert(3, "Sim Risk", risk_scores)

    st.dataframe(df, use_container_width=True, height=600)

    # Alert summary
    if st.session_state.sim_results:
        st.subheader("Risk Alerts")
        for ac in fleet:
            res = st.session_state.sim_results.get(ac.tail_number, {})
            if not res:
                continue
            top = max(res.values(), key=lambda r: r.failure_prob)
            if top.failure_prob >= 0.40:
                chain_str = " → ".join(top.cascade_chain)
                color = "#e74c3c" if top.failure_prob >= 0.70 else "#e67e22"
                st.markdown(
                    f"**{ac.tail_number}**: {top.subsystem.replace('_',' ').upper()} "
                    f"— P={top.failure_prob:.0%} — cascade: `{chain_str}`",
                    help=f"Alert level: {'CRITICAL' if top.failure_prob >= 0.70 else 'CAUTION'}"
                )


# ─── Panel 2: Cascade Detail ──────────────────────────────────────────────────
with tab2:
    st.subheader("Cascade Detail — Per-Tail Drill-Down")

    tail_options = [ac.tail_number for ac in fleet]
    selected_tail = st.selectbox("Select tail", tail_options, index=0)
    aircraft = next(ac for ac in fleet if ac.tail_number == selected_tail)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Current Health States**")
        for s in SUBSYSTEMS:
            h = aircraft.health_states.get(s, 1.0)
            level = _alert_level(h)
            color = _health_color(h)
            st.markdown(
                f"{_health_emoji(h)} **{s.replace('_', ' ').title()}**: "
                f"<span style='color:{color}'>{h:.3f} ({level})</span>",
                unsafe_allow_html=True,
            )

        if aircraft.open_deferred_items:
            st.markdown(f"**Open Deferred**: {', '.join(aircraft.open_deferred_items)}")

    with col_right:
        # Forward simulation for this tail
        if st.button(f"▶ Simulate {selected_tail}", key="single_sim"):
            with st.spinner("Running Monte Carlo simulation..."):
                results = _run_forward_sim(aircraft, branches=300)
                st.session_state.sim_results[selected_tail] = results

        results = st.session_state.sim_results.get(selected_tail)
        if results:
            st.markdown("**Failure Probabilities (10-cycle horizon)**")
            probs = {s: r.failure_prob for s, r in results.items()}
            sorted_probs = sorted(probs.items(), key=lambda x: -x[1])

            chart_data = pd.DataFrame(
                [(s.replace("_", " "), p) for s, p in sorted_probs],
                columns=["Subsystem", "Failure Prob"]
            )
            st.bar_chart(chart_data.set_index("Subsystem"), color="#e74c3c")

    # Graph visualisation
    if st.session_state.sim_results.get(selected_tail):
        st.markdown("---")
        st.markdown("**System Dependency Graph** (red edges = active cascade path)")
        fig = _draw_graph(aircraft, st.session_state.sim_results[selected_tail])
        st.pyplot(fig)
        plt.close(fig)

    # Claude recommendation
    st.markdown("---")
    st.markdown("**AI Maintenance Recommendation**")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.warning("Set ANTHROPIC_API_KEY in .env to enable AI recommendations.")
    elif st.button(f"Generate Recommendation for {selected_tail}", key="gen_rec"):
        if not st.session_state.sim_results.get(selected_tail):
            st.error("Run simulation first.")
        else:
            with st.spinner("Consulting Claude..."):
                from maintenance_sim.llm import (
                    MaintenanceRecommendation,
                    build_graph_context,
                    generate_recommendation,
                )
                from maintenance_sim.memory import load_biography

                results = st.session_state.sim_results[selected_tail]
                top = max(results.values(), key=lambda r: r.failure_prob)
                graph_ctx = build_graph_context(G, top.cascade_chain)
                biography = load_biography(conn, selected_tail)

                try:
                    rec = generate_recommendation(
                        selected_tail, biography, results, graph_ctx
                    )
                    st.session_state.recommendations[selected_tail] = rec
                    # Save recommendation text to DB
                    record_prediction(
                        conn, selected_tail, cycle=0, horizon=10,
                        subsystem=rec.primary_subsystem,
                        prob=top.failure_prob,
                        chain=rec.cascade_chain,
                        recommendation=rec.recommended_action,
                    )
                except Exception as e:
                    st.error(f"LLM error: {e}")

    rec = st.session_state.recommendations.get(selected_tail)
    if rec:
        _render_recommendation_card(rec)


# ─── Panel 3: Validation ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Historical Validation — Cascade Model vs. Threshold Baseline")
    st.markdown(
        "Tests the cascade model against **held-out 2022–2023 FAA SDR data** "
        "it has never seen. Shows how many real multi-system failure sequences "
        "the model would have caught ahead of the first filed SDR report."
    )

    val_branches = st.slider("Branches per cascade", 30, 200, 80, 10,
                             key="val_branches")

    if st.button("▶ Run Validation", type="primary"):
        with st.spinner("Evaluating against held-out test data..."):
            from maintenance_sim.validation import (
                format_validation_table,
                run_validation,
            )
            metrics = run_validation(horizon=10, branches=val_branches)
            st.session_state["val_metrics"] = metrics
            st.session_state["val_table"] = format_validation_table(metrics)

    if "val_metrics" in st.session_state:
        m = st.session_state["val_metrics"]
        table = st.session_state["val_table"]

        # Headline metric
        delta = m.cascade_delta
        st.metric(
            "Cascade Detection Improvement",
            f"+{delta:.0%}",
            help="Cascade model recall minus threshold baseline recall"
        )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Cascade Recall", f"{m.cascade_recall:.0%}",
                     f"{m.cascade_recall - m.threshold_recall:+.0%} vs baseline")
        col_b.metric("Precision", f"{m.precision:.0%}")
        col_c.metric("Avg Detection Lead", f"{m.avg_detection_lead_days:.1f} days")

        st.markdown("**Comparison Table**")
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        st.info(
            f"Model caught **{m.n_cascades_found}/{m.n_total_cascades}** observed cascade "
            f"sequences in the test set vs. ~{m.threshold_recall:.0%} for a threshold-only approach."
        )


# ─── Panel 4: Feedback Entry ─────────────────────────────────────────────────
with tab4:
    st.subheader("Feedback Loop — Submit Technician Outcomes")
    st.markdown(
        "When a technician confirms or denies a predicted fault after inspection, "
        "submit the outcome here. The model's cascade weights for this specific tail "
        "update automatically — each confirmation/denial teaches the system about "
        "this aircraft's unique failure patterns."
    )

    all_preds = get_all_predictions(conn)

    if not all_preds:
        st.info("No predictions recorded yet. Run the simulation first.")
    else:
        # Build display options
        pred_options = {
            f"[{p['tail_number']}] {p['subsystem']} P={p['failure_probability']:.0%} "
            f"({p['created_at'][:10]})": p
            for p in all_preds
            if p.get("outcome") is None  # only unresolved predictions
        }

        if not pred_options:
            st.success("All predictions have received feedback. Run simulation to generate new ones.")
        else:
            selected_label = st.selectbox("Select prediction", list(pred_options.keys()))
            pred = pred_options[selected_label]

            st.markdown(f"**Tail:** {pred['tail_number']}")
            st.markdown(f"**Predicted fault:** {pred['subsystem'].replace('_', ' ').title()}")
            st.markdown(f"**Failure probability:** {pred['failure_probability']:.0%}")
            st.markdown(f"**Cascade chain:** {' → '.join(pred['cascade_chain'])}")

            outcome = st.radio(
                "Inspection outcome",
                ["confirmed", "denied", "partial"],
                horizontal=True,
            )
            actual_cycle = st.number_input(
                "Actual cycle at fault (if confirmed)", min_value=0, value=0
            )
            notes = st.text_area("Technician notes (optional)")

            if st.button("Submit Feedback", type="primary"):
                record_feedback(
                    conn,
                    prediction_id=pred["prediction_id"],
                    tail=pred["tail_number"],
                    subsystem=pred["subsystem"],
                    outcome=outcome,
                    actual_cycle=actual_cycle if actual_cycle > 0 else None,
                    notes=notes,
                )
                st.success(
                    f"Feedback recorded: **{outcome}** for {pred['tail_number']} "
                    f"{pred['subsystem']}. Edge weights updated."
                )
                st.rerun()
