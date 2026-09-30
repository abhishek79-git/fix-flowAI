"""FixFlow AI — From Complaint to Action.

Main Streamlit application entry point.
AI-powered complaint prioritization and resource optimization system.
"""

import os
import sys

import streamlit as st
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="FixFlow AI — From Complaint to Action",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    /* Global styling */
    .stApp { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        margin-bottom: 10px;
    }

    .critical { color: #ff4444 !important; font-weight: bold; }
    .high { color: #ff8800 !important; font-weight: bold; }
    .medium { color: #ffcc00 !important; font-weight: bold; }
    .low { color: #44ff44 !important; font-weight: bold; }

    /* Header branding */
    .brand-header {
        text-align: center;
        padding: 5px 0;
        border-bottom: 2px solid #333;
        margin-bottom: 15px;
    }
    .brand-header h3 {
        color: #4488ff;
        margin: 0;
    }
    .brand-header p {
        color: #888;
        font-size: 12px;
        margin: 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117, #1a1a2e);
    }
</style>
""", unsafe_allow_html=True)


def main() -> None:
    """Main application entry point."""
    # Sidebar
    st.sidebar.markdown("""
    <div class="brand-header">
        <h3>🔧 FixFlow AI</h3>
        <p>From Complaint to Action</p>
    </div>
    """, unsafe_allow_html=True)

    demo_mode = st.sidebar.toggle("🎮 Demo Mode", value=True,
                                   help="Use bundled synthetic data for reliable offline demos")
    st.session_state["demo_mode"] = demo_mode

    if demo_mode:
        st.sidebar.success("📦 Demo Mode Active — No internet or model required")
    else:
        st.sidebar.info("🔬 Live Mode — Using trained model")

    # Load config
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            st.session_state["config"] = config
    except FileNotFoundError:
        st.session_state["config"] = {}

    # Navigation
    pages = {
        "🏠 Operations": [
            st.Page("ui/dashboard.py", title="🏠 Command Center", default=True),
            st.Page("ui/report.py", title="📝 Report Problem"),
            st.Page("ui/triage.py", title="🔍 AI Triage"),
            st.Page("ui/issues.py", title="📋 Issue Explorer"),
            st.Page("ui/optimize_ui.py", title="⚡ Resource Optimizer"),
        ],
        "🧪 Simulation & Analysis": [
            st.Page("ui/drift_sim.py", title="🌧 Drift Simulator"),
            st.Page("ui/ai_lab.py", title="🧪 AI Lab"),
            st.Page("ui/benchmark_ui.py", title="📊 Benchmark"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align:center; color:#666; font-size:11px;">
        <p>FixFlow AI v1.0.0</p>
        <p>OptiForge 2026 Hackathon</p>
        <p>ML & AI Track</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
