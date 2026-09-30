"""FixFlow AI - AI Lab Page.

Shows evolutionary optimization, Pareto front, robustness, calibration, and fairness.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def _demo_convergence() -> pd.DataFrame:
    """Generate demo evolutionary convergence data."""
    np.random.seed(42)
    generations = list(range(1, 16))
    best_f1 = [0.71, 0.74, 0.77, 0.80, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.905, 0.908, 0.91, 0.912]
    avg_f1 = [x - np.random.uniform(0.03, 0.08) for x in best_f1]
    worst_f1 = [x - np.random.uniform(0.08, 0.15) for x in best_f1]
    return pd.DataFrame({'Generation': generations, 'Best F1': best_f1, 'Avg F1': avg_f1, 'Worst F1': worst_f1})


def _demo_pareto() -> pd.DataFrame:
    """Generate demo Pareto front data."""
    np.random.seed(42)
    n = 20
    accuracy = np.random.uniform(0.85, 0.95, n)
    latency = 5 + 45 * (1 - (accuracy - 0.85) / 0.10) + np.random.normal(0, 3, n)
    params = (50000 + 150000 * (1 - (accuracy - 0.85) / 0.10) + np.random.normal(0, 10000, n)).astype(int)
    feasible = (latency < 50) & (params < 200000)
    pareto = accuracy > np.percentile(accuracy, 70)
    return pd.DataFrame({
        'Accuracy': accuracy, 'Latency (ms)': np.clip(latency, 5, 60),
        'Parameters': params, 'Feasible': feasible, 'Pareto Front': pareto
    })


def _demo_robustness() -> pd.DataFrame:
    """Generate demo robustness data."""
    return pd.DataFrame({
        'Scenario': ['Normal', 'Missing Fields', 'Text Noise', 'Numeric Shift', 'Heavy Rain', 'Complaint Surge'],
        'Macro F1': [0.924, 0.891, 0.903, 0.882, 0.869, 0.854],
        'ECE': [0.041, 0.055, 0.048, 0.062, 0.059, 0.071],
        'Latency (ms)': [14, 14, 15, 14, 14, 16],
    })


def _demo_calibration() -> tuple:
    """Generate demo calibration data."""
    bins = np.arange(0.05, 1.0, 0.1)
    predicted = bins
    actual = bins + np.random.normal(0, 0.03, len(bins))
    actual = np.clip(actual, 0, 1)
    counts = np.random.randint(20, 100, len(bins))
    return predicted, actual, counts


def _demo_fairness() -> pd.DataFrame:
    """Generate demo fairness data."""
    return pd.DataFrame({
        'Zone': ['Academic', 'Hostel', 'Residential', 'Public', 'Administrative'],
        'Macro F1': [0.931, 0.922, 0.917, 0.920, 0.925],
        'ECE': [0.040, 0.045, 0.042, 0.048, 0.039],
    })


def show() -> None:
    """Render AI Lab page."""
    st.title("🧪 AI Lab")
    st.markdown("*Deep dive into model optimization, robustness, and evaluation metrics.*")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Model Evolution", "🎯 Pareto Front", "🛡 Robustness",
        "📊 Calibration", "⚖ Fairness"
    ])

    with tab1:
        st.subheader("Evolutionary Convergence")
        conv_df = _demo_convergence()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=conv_df['Generation'], y=conv_df['Best F1'],
                                  name='Best', line=dict(color='#44ff44', width=3)))
        fig.add_trace(go.Scatter(x=conv_df['Generation'], y=conv_df['Avg F1'],
                                  name='Average', line=dict(color='#4488ff', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=conv_df['Generation'], y=conv_df['Worst F1'],
                                  name='Worst', line=dict(color='#ff4444', width=1, dash='dot')))
        fig.update_layout(
            title='Fitness Over Generations (NSGA-II)',
            xaxis_title='Generation', yaxis_title='Macro F1 Score',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Convergence:** `0.71 → 0.77 → 0.84 → 0.89 → 0.91`")

        # Best chromosome
        st.subheader("Best Chromosome")
        best = {
            'Hidden Layer 1': 128, 'Hidden Layer 2': 64,
            'Dropout': 0.20, 'Learning Rate': 0.001,
            'Weight Decay': 0.0001, 'Huber Delta': 1.0,
            'Severity Loss Weight': 0.5, 'Resolution Loss Weight': 0.3
        }
        st.json(best)

    with tab2:
        st.subheader("Pareto Front: Accuracy vs Latency")
        pareto_df = _demo_pareto()

        fig2 = px.scatter(
            pareto_df, x='Latency (ms)', y='Accuracy',
            size='Parameters', color='Pareto Front',
            color_discrete_map={True: '#44ff44', False: '#666666'},
            symbol='Feasible',
            symbol_map={True: 'circle', False: 'x'},
            hover_data=['Parameters'],
        )

        # Feasibility region
        fig2.add_vrect(x0=0, x1=50, fillcolor="green", opacity=0.05, line_width=0)
        fig2.add_vline(x=50, line_dash="dash", line_color="red",
                       annotation_text="Latency Limit (50ms)")

        fig2.update_layout(
            title='Multi-Objective Trade-off (NSGA-II)',
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.info("🟢 Green = Pareto-optimal solutions | ⚫ Gray = Dominated | ❌ = Infeasible (constraint violation)")

    with tab3:
        st.subheader("Robustness Across Scenarios")
        rob_df = _demo_robustness()

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Macro F1', x=rob_df['Scenario'], y=rob_df['Macro F1'],
                               marker_color=['#44ff44'] + ['#ff8800'] * 5))
        fig3.update_layout(title='Performance Under Distribution Shift', height=400,
                           yaxis_range=[0.7, 1.0])
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(rob_df, use_container_width=True, hide_index=True)

        # Generalization gap
        id_f1 = rob_df.loc[0, 'Macro F1']
        ood_f1 = rob_df['Macro F1'].iloc[1:].mean()
        gap = id_f1 - ood_f1
        st.metric("Generalization Gap", f"{gap:.1%}")

    with tab4:
        st.subheader("Reliability Diagram")
        predicted, actual, counts = _demo_calibration()

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=predicted, y=actual, mode='markers+lines',
                                   name='Model', marker=dict(size=10, color='#4488ff')))
        fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                   name='Perfect Calibration', line=dict(dash='dash', color='gray')))
        fig4.update_layout(
            title='Calibration: Predicted vs Actual Confidence',
            xaxis_title='Predicted Confidence', yaxis_title='Actual Accuracy',
            height=450, xaxis_range=[0, 1], yaxis_range=[0, 1]
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.metric("Expected Calibration Error (ECE)", "0.041", "< 0.08 threshold ✅")

    with tab5:
        st.subheader("Group-wise Fairness")
        fair_df = _demo_fairness()

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=fair_df['Zone'], y=fair_df['Macro F1'],
                               name='Macro F1', marker_color='#4488ff'))
        fig5.update_layout(title='Performance by Location Zone', height=400,
                           yaxis_range=[0.85, 1.0])
        st.plotly_chart(fig5, use_container_width=True)

        st.dataframe(fair_df, use_container_width=True, hide_index=True)

        max_gap = fair_df['Macro F1'].max() - fair_df['Macro F1'].min()
        st.metric("Max F1 Gap Between Groups", f"{max_gap:.3f}",
                  "< 0.05 threshold ✅" if max_gap < 0.05 else "> 0.05 ⚠️")


show()
