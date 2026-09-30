"""FixFlow AI - Command Center Dashboard.

Main overview page showing system status, metrics, and issue distribution.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random


def _generate_demo_metrics() -> dict:
    """Generate demo model health metrics."""
    return {
        'accuracy': 93.2,
        'ood_f1': 87.4,
        'ece': 0.041,
        'fairness_gap': 0.027,
        'latency_ms': 14,
        'param_count': '148K',
    }


def _generate_demo_issues(n: int = 124) -> pd.DataFrame:
    """Generate demo issue data for dashboard."""
    random.seed(42)
    np.random.seed(42)

    categories = [
        'Road Damage', 'Electrical Hazard', 'Water Leakage',
        'Waste Management', 'Streetlight Failure', 'Drainage Blockage',
        'Building Damage', 'Public Safety'
    ]
    zones = ['Academic Zone', 'Hostel Zone', 'Residential Zone', 'Public Zone', 'Administrative Zone']
    severities = ['Critical', 'High', 'Medium', 'Low']
    sev_weights = [0.08, 0.22, 0.41, 0.29]

    records = []
    for i in range(n):
        sev = random.choices(severities, weights=sev_weights, k=1)[0]
        records.append({
            'id': f'ISS-{i+1:04d}',
            'category': random.choice(categories),
            'location': random.choice(zones),
            'severity': sev,
            'priority': random.randint(20, 100),
            'reports': random.randint(1, 20),
            'affected': random.randint(5, 300),
            'status': random.choices(['Open', 'In Progress', 'Resolved'], weights=[0.6, 0.25, 0.15], k=1)[0]
        })
    return pd.DataFrame(records)


def show() -> None:
    """Render the Command Center dashboard."""
    st.markdown("""
    <div style='text-align:center; padding:10px 0;'>
        <h1>🔧 FixFlow AI</h1>
        <p style='font-size:18px; color:#888;'>From Complaint to Action — Command Center</p>
    </div>
    """, unsafe_allow_html=True)

    issues_df = _generate_demo_issues()
    metrics = _generate_demo_metrics()

    # Top metrics row
    st.markdown("### 📊 System Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Active Issues", len(issues_df[issues_df['status'] != 'Resolved']), "+12")
    with c2:
        critical = len(issues_df[issues_df['severity'] == 'Critical'])
        st.metric("🔴 Critical", critical, "-2", delta_color="inverse")
    with c3:
        high = len(issues_df[issues_df['severity'] == 'High'])
        st.metric("🟠 High", high, "+5")
    with c4:
        medium = len(issues_df[issues_df['severity'] == 'Medium'])
        st.metric("🟡 Medium", medium)
    with c5:
        resolved = len(issues_df[issues_df['status'] == 'Resolved'])
        st.metric("✅ Resolved", resolved, "+8")

    st.markdown("---")

    # Model Health Panel
    st.markdown("### 🤖 Model Health")
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    with h1:
        st.metric("Accuracy", f"{metrics['accuracy']}%")
    with h2:
        st.metric("OOD F1", f"{metrics['ood_f1']}%")
    with h3:
        st.metric("Calibration", f"{metrics['ece']}")
    with h4:
        st.metric("Fairness Gap", f"{metrics['fairness_gap']}")
    with h5:
        st.metric("Latency", f"{metrics['latency_ms']} ms")
    with h6:
        st.metric("Parameters", metrics['param_count'])

    # System status indicator
    st.markdown(
        '<div style="background:#1a4d1a;border-radius:8px;padding:10px;text-align:center;">'
        '<span style="color:#44ff44;font-size:20px;">●</span> '
        '<span style="color:#44ff44;font-weight:bold;">SYSTEM HEALTHY</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Charts
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("### Category Distribution")
        cat_counts = issues_df['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig = px.pie(cat_counts, values='Count', names='Category', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        st.markdown("### Severity Distribution")
        sev_order = ['Critical', 'High', 'Medium', 'Low']
        sev_colors = {'Critical': '#ff4444', 'High': '#ff8800', 'Medium': '#ffcc00', 'Low': '#44ff44'}
        sev_counts = issues_df['severity'].value_counts().reindex(sev_order).fillna(0).reset_index()
        sev_counts.columns = ['Severity', 'Count']
        fig2 = px.bar(sev_counts, x='Severity', y='Count',
                      color='Severity', color_discrete_map=sev_colors)
        fig2.update_layout(height=350, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Priority distribution by zone
    chart3, chart4 = st.columns(2)
    with chart3:
        st.markdown("### Priority by Zone")
        zone_priority = issues_df.groupby('location')['priority'].mean().sort_values(ascending=True).reset_index()
        zone_priority.columns = ['Zone', 'Avg Priority']
        fig3 = px.bar(zone_priority, y='Zone', x='Avg Priority', orientation='h',
                      color='Avg Priority', color_continuous_scale='RdYlGn_r')
        fig3.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with chart4:
        st.markdown("### Recent Activity")
        # Simulated time series
        np.random.seed(42)
        days = pd.date_range(end=pd.Timestamp.now(), periods=14, freq='D')
        activity = pd.DataFrame({
            'Date': days,
            'New Issues': np.random.poisson(12, 14),
            'Resolved': np.random.poisson(10, 14),
        })
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=activity['Date'], y=activity['New Issues'],
                                   name='New', line=dict(color='#ff4444')))
        fig4.add_trace(go.Scatter(x=activity['Date'], y=activity['Resolved'],
                                   name='Resolved', line=dict(color='#44ff44')))
        fig4.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig4, use_container_width=True)

    # Top priority issues
    st.markdown("### 🔥 Top Priority Issues")
    top_issues = issues_df.sort_values('priority', ascending=False).head(5)
    for _, row in top_issues.iterrows():
        sev_color = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}.get(row['severity'], '⚪')
        with st.container():
            tc1, tc2, tc3, tc4 = st.columns([1, 3, 1, 1])
            with tc1:
                st.markdown(f"**{sev_color} {row['severity']}**")
            with tc2:
                st.markdown(f"**{row['category']}** — {row['location']}")
            with tc3:
                st.markdown(f"Priority: **{row['priority']}**")
            with tc4:
                st.markdown(f"Reports: **{row['reports']}**")


show()
