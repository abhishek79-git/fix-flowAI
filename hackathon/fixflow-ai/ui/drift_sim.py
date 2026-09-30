"""FixFlow AI - Drift Simulator Page.

Simulates environmental changes and shows dynamic priority re-computation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random


def _generate_demo_issues(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Generate demo issues for simulation."""
    random.seed(seed)
    np.random.seed(seed)

    categories = [
        'Road Damage', 'Electrical Hazard', 'Water Leakage',
        'Waste Management', 'Streetlight Failure', 'Drainage Blockage',
        'Building Damage', 'Public Safety'
    ]
    zones = ['Academic Zone', 'Hostel Zone', 'Residential Zone', 'Public Zone', 'Administrative Zone']

    records = []
    for i in range(n):
        cat = random.choice(categories)
        severity = round(random.uniform(0.3, 0.9), 2)
        urgency = round(random.uniform(0.3, 0.9), 2)
        safety = round(random.uniform(0.2, 0.95), 2)
        priority = round(severity * 25 + urgency * 25 + safety * 20 + random.uniform(5, 30), 1)
        priority = min(100, priority)

        records.append({
            'id': f'ISS-{i+1:03d}',
            'category': cat,
            'location': random.choice(zones),
            'severity': severity,
            'urgency': urgency,
            'safety_risk': safety,
            'priority': priority,
            'affected_people': random.randint(5, 200),
            'reports': random.randint(1, 15),
        })
    return pd.DataFrame(records)


def _apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Apply a drift scenario to issues and recompute priorities."""
    result = df.copy()

    if scenario == "Heavy Rain":
        # Water and drainage issues become critical
        water_mask = result['category'].isin(['Water Leakage', 'Drainage Blockage'])
        result.loc[water_mask, 'severity'] = np.clip(result.loc[water_mask, 'severity'] + 0.3, 0, 1)
        result.loc[water_mask, 'urgency'] = np.clip(result.loc[water_mask, 'urgency'] + 0.25, 0, 1)
        result.loc[water_mask, 'safety_risk'] = np.clip(result.loc[water_mask, 'safety_risk'] + 0.2, 0, 1)

        # Electrical hazards also worsen in rain
        elec_mask = result['category'] == 'Electrical Hazard'
        result.loc[elec_mask, 'severity'] = np.clip(result.loc[elec_mask, 'severity'] + 0.2, 0, 1)
        result.loc[elec_mask, 'safety_risk'] = np.clip(result.loc[elec_mask, 'safety_risk'] + 0.3, 0, 1)

        # Road damage worsens
        road_mask = result['category'] == 'Road Damage'
        result.loc[road_mask, 'severity'] = np.clip(result.loc[road_mask, 'severity'] + 0.15, 0, 1)

    elif scenario == "Complaint Surge":
        result['reports'] = result['reports'] * 5
        result['affected_people'] = result['affected_people'] * 3

    elif scenario == "Missing Data":
        np.random.seed(42)
        mask = np.random.random(len(result)) < 0.3
        result.loc[mask, 'severity'] = result.loc[mask, 'severity'] * 0.7

    elif scenario == "Text Noise":
        pass  # Text noise doesn't directly change priorities

    elif scenario == "Numeric Shift":
        result['affected_people'] = result['affected_people'] * 2
        result['severity'] = np.clip(result['severity'] + 0.1, 0, 1)

    elif scenario == "CRISIS":
        # Everything at once
        water_mask = result['category'].isin(['Water Leakage', 'Drainage Blockage'])
        result.loc[water_mask, 'severity'] = np.clip(result.loc[water_mask, 'severity'] + 0.35, 0, 1)
        result.loc[water_mask, 'urgency'] = np.clip(result.loc[water_mask, 'urgency'] + 0.3, 0, 1)

        elec_mask = result['category'] == 'Electrical Hazard'
        result.loc[elec_mask, 'severity'] = np.clip(result.loc[elec_mask, 'severity'] + 0.25, 0, 1)
        result.loc[elec_mask, 'safety_risk'] = np.clip(result.loc[elec_mask, 'safety_risk'] + 0.35, 0, 1)

        road_mask = result['category'] == 'Road Damage'
        result.loc[road_mask, 'severity'] = np.clip(result.loc[road_mask, 'severity'] + 0.2, 0, 1)

        result['reports'] = result['reports'] * 8
        result['affected_people'] = result['affected_people'] * 4

    # Recompute priorities
    result['priority'] = np.clip(
        result['severity'] * 25 +
        result['urgency'] * 25 +
        result['safety_risk'] * 20 +
        np.minimum(result['affected_people'] / 100, 1.0) * 15 +
        np.minimum(result['reports'] / 10, 1.0) * 15,
        0, 100
    ).round(1)

    return result


def show() -> None:
    """Render Drift Simulator page."""
    st.title("🌧 Drift Simulator")
    st.markdown("*Simulate environmental changes and observe how FixFlow AI dynamically re-prioritizes issues.*")

    # Generate baseline issues
    if 'sim_issues' not in st.session_state:
        st.session_state['sim_issues'] = _generate_demo_issues()

    base_df = st.session_state['sim_issues']

    st.markdown("---")

    # Scenario buttons
    st.subheader("Select Scenario")
    col1, col2, col3 = st.columns(3)
    with col1:
        heavy_rain = st.button("🌧 Heavy Rain", use_container_width=True)
        complaint_surge = st.button("📈 Complaint Surge", use_container_width=True)
    with col2:
        missing_data = st.button("❓ Missing Data", use_container_width=True)
        numeric_shift = st.button("📊 Numeric Shift", use_container_width=True)
    with col3:
        text_noise = st.button("📝 Text Noise", use_container_width=True)
        reset = st.button("🔄 Reset to Normal", use_container_width=True)

    st.markdown("---")

    # CRISIS button
    crisis = st.button(
        "⚡ SIMULATE CRISIS (Heavy Rain + Surge + Multiple Failures)",
        type="primary", use_container_width=True
    )

    # Determine active scenario
    scenario = None
    if heavy_rain: scenario = "Heavy Rain"
    elif complaint_surge: scenario = "Complaint Surge"
    elif missing_data: scenario = "Missing Data"
    elif text_noise: scenario = "Text Noise"
    elif numeric_shift: scenario = "Numeric Shift"
    elif crisis: scenario = "CRISIS"
    elif reset:
        st.session_state['sim_issues'] = _generate_demo_issues()
        st.rerun()

    if scenario:
        st.warning(f"⚠️ Distribution shift detected: **{scenario}**")

        with st.spinner("Re-optimizing priorities..."):
            shifted_df = _apply_scenario(base_df, scenario)

        # Before / After comparison
        st.subheader("Priority Changes")

        comparison = pd.DataFrame({
            'Issue': base_df['id'],
            'Category': base_df['category'],
            'Before': base_df['priority'],
            'After': shifted_df['priority'],
            'Change': (shifted_df['priority'] - base_df['priority']).round(1),
        }).sort_values('Change', ascending=False)

        # Show top changers
        top_changes = comparison.head(10)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Before', x=top_changes['Category'] + ' (' + top_changes['Issue'] + ')',
            y=top_changes['Before'], marker_color='#4488ff'
        ))
        fig.add_trace(go.Bar(
            name='After', x=top_changes['Category'] + ' (' + top_changes['Issue'] + ')',
            y=top_changes['After'], marker_color='#ff4444'
        ))
        fig.update_layout(
            barmode='group', title='Top 10 Priority Changes',
            xaxis_title='Issue', yaxis_title='Priority Score',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # New action plan
        st.subheader("🔄 Re-optimized Action Plan")
        new_plan = shifted_df.sort_values('priority', ascending=False).head(8)

        for rank, (_, row) in enumerate(new_plan.iterrows(), 1):
            sev_emoji = "🔴" if row['severity'] > 0.7 else ("🟡" if row['severity'] > 0.4 else "🟢")
            col_a, col_b, col_c = st.columns([1, 4, 2])
            with col_a:
                st.markdown(f"### {rank}")
            with col_b:
                st.markdown(f"**{sev_emoji} {row['category']}** — {row['location']}")
                st.caption(f"Severity: {row['severity']:.0%} | Urgency: {row['urgency']:.0%} | Safety: {row['safety_risk']:.0%}")
            with col_c:
                st.metric("Priority", f"{row['priority']:.0f}")

        # Stats
        st.markdown("---")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            avg_change = comparison['Change'].mean()
            st.metric("Avg Priority Change", f"+{avg_change:.1f}" if avg_change > 0 else f"{avg_change:.1f}")
        with s2:
            max_change = comparison['Change'].max()
            st.metric("Max Priority Increase", f"+{max_change:.1f}")
        with s3:
            critical_count = len(shifted_df[shifted_df['priority'] > 80])
            st.metric("Critical Issues", critical_count)
        with s4:
            st.metric("Total Issues", len(shifted_df))

    else:
        # Show current state
        st.subheader("Current Issue Priorities")
        st.dataframe(
            base_df[['id', 'category', 'location', 'severity', 'urgency', 'safety_risk', 'priority']]
            .sort_values('priority', ascending=False),
            use_container_width=True, hide_index=True
        )


show()
