import streamlit as st
from typing import Dict, Any

def metric_card(title: str, value: str, delta: str = None, color: str = 'green') -> None:
    """Displays a styled metric card."""
    st.markdown(f'''
    <div class="metric-card">
        <h4>{title}</h4>
        <h2>{value}</h2>
        {f"<p style='color: {color};'>{delta}</p>" if delta else ""}
    </div>
    ''', unsafe_allow_html=True)

def severity_badge(level: str) -> None:
    """Displays a colored badge for severity."""
    colors = {"Critical": "critical", "High": "high", "Medium": "medium", "Low": "low"}
    cls = colors.get(level, "low")
    st.markdown(f"<span class='{cls}' style='font-weight:bold;'>{level}</span>", unsafe_allow_html=True)

def priority_bar(score: float) -> None:
    """Displays a horizontal priority bar."""
    st.progress(score / 100)

def status_indicator(status: str, label: str) -> None:
    """Displays a status indicator dot."""
    color = "green" if status == "OK" else ("yellow" if status == "WARN" else "red")
    st.markdown(f"<span style='color: {color};'>●</span> {label}", unsafe_allow_html=True)

def issue_card(issue_data: Dict[str, Any]) -> None:
    """Displays a compact card for an issue."""
    st.markdown(f"""
    <div style='border:1px solid #444; border-radius:5px; padding:10px; margin-bottom:10px;'>
        <h4>{issue_data.get('title', 'Unknown Issue')}</h4>
        <p>Category: {issue_data.get('category', 'N/A')}</p>
        <p>Priority: {issue_data.get('priority', 0)}</p>
    </div>
    """, unsafe_allow_html=True)

def model_health_panel(metrics: Dict[str, float]) -> None:
    """Displays the model health panel."""
    cols = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(k, f"{v:.2f}")
