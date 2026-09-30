"""
AI Triage Page for FixFlow AI.
Displays the detailed analysis of the most recent complaint.
"""
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List

def get_demo_prediction() -> Dict[str, Any]:
    """
    Generates a demo prediction for testing purposes.
    
    Returns:
        Dict[str, Any]: A dictionary containing demo prediction data.
    """
    return {
        "text": "Exposed electrical wiring near Block B",
        "category": "Electrical Hazard",
        "confidence": 0.95,
        "severity": 0.9,
        "urgency": 0.85,
        "safety_risk": 0.95,
        "zone": "Hostel Zone",
        "duplicates": [
            {"text": "Live wires hanging outside Block B", "similarity": 0.92},
            {"text": "Sparks coming from pole near B block", "similarity": 0.88}
        ],
        "priority_score": 0.93,
        "priority_factors": {
            "severity_contrib": 0.45,
            "urgency_contrib": 0.35,
            "safety_contrib": 0.13
        },
        "explanations": {
            "severity": "Exposed wiring poses a high risk of electric shock or fire.",
            "urgency": "Immediate action required to prevent accidents in high-traffic hostel area.",
            "safety_risk": "High potential for severe injury or fatality."
        }
    }

def create_gauge_chart(value: float, title: str, color: str) -> go.Figure:
    """
    Creates a Plotly gauge chart for metric visualization.
    
    Args:
        value (float): The metric value (0.0 to 1.0).
        title (str): Title of the gauge chart.
        color (str): Primary color for the gauge bar.
        
    Returns:
        go.Figure: The generated Plotly figure.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def show() -> None:
    """
    Displays the AI Triage page.
    """
    st.title("AI Triage Analysis")
    
    try:
        prediction = st.session_state.get('last_prediction', get_demo_prediction())
        
        st.subheader(f"Complaint: {prediction['text']}")
        st.write(f"**Category:** {prediction['category']} | **Zone:** {prediction['zone']}")
        
        st.write("### Impact Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(create_gauge_chart(prediction['severity'], "Severity", "red"), use_container_width=True)
            st.info(f"**Why:** {prediction['explanations']['severity']}")
        with col2:
            st.plotly_chart(create_gauge_chart(prediction['urgency'], "Urgency", "orange"), use_container_width=True)
            st.info(f"**Why:** {prediction['explanations']['urgency']}")
        with col3:
            st.plotly_chart(create_gauge_chart(prediction['safety_risk'], "Safety Risk", "darkred"), use_container_width=True)
            st.info(f"**Why:** {prediction['explanations']['safety_risk']}")
            
        st.subheader("Priority Calculation Breakdown")
        factors = prediction['priority_factors']
        fig_bar = go.Figure(go.Bar(
            x=[factors['severity_contrib'], factors['urgency_contrib'], factors['safety_contrib']],
            y=['Severity', 'Urgency', 'Safety Risk'],
            orientation='h',
            marker_color=['red', 'orange', 'darkred']
        ))
        fig_bar.update_layout(
            title="Contribution to Final Priority Score", 
            height=300,
            xaxis_title="Contribution Factor",
            yaxis_title="Metric"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.subheader("Duplicate Detection")
        if prediction.get('duplicates'):
            for dup in prediction['duplicates']:
                st.write(f"- {dup['text']} (Similarity: {dup['similarity'] * 100:.1f}%)")
        else:
            st.write("No similar recent complaints found.")
            
    except Exception as e:
        st.error(f"Error loading triage analysis: {str(e)}")

if __name__ == "__main__":
    show()
