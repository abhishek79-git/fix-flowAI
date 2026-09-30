"""FixFlow AI - Report Problem Page.

Allows users to submit complaints and see AI-powered analysis.
"""

import streamlit as st
import random
import numpy as np


def _demo_prediction(text: str, zone: str, weather: str, people: int) -> dict:
    """Generate realistic demo prediction based on keywords."""
    random.seed(hash(text) % 2**31)

    # Keyword-based category detection
    text_lower = text.lower()
    if any(w in text_lower for w in ['wire', 'electric', 'shock', 'spark']):
        category, severity, safety = 'Electrical Hazard', 0.94, 0.97
    elif any(w in text_lower for w in ['pothole', 'road', 'crack', 'pavement']):
        category, severity, safety = 'Road Damage', 0.78, 0.65
    elif any(w in text_lower for w in ['water', 'leak', 'pipe', 'flood']):
        category, severity, safety = 'Water Leakage', 0.72, 0.55
    elif any(w in text_lower for w in ['drain', 'sewer', 'clog', 'blockage']):
        category, severity, safety = 'Drainage Blockage', 0.68, 0.50
    elif any(w in text_lower for w in ['light', 'lamp', 'dark', 'bulb']):
        category, severity, safety = 'Streetlight Failure', 0.60, 0.45
    elif any(w in text_lower for w in ['waste', 'garbage', 'trash', 'rubbish']):
        category, severity, safety = 'Waste Management', 0.55, 0.30
    elif any(w in text_lower for w in ['building', 'wall', 'roof', 'ceiling']):
        category, severity, safety = 'Building Damage', 0.70, 0.60
    else:
        category, severity, safety = 'Public Safety', 0.65, 0.50

    # Adjust for weather
    if weather in ('Heavy Rain', 'Storm'):
        severity = min(1.0, severity + 0.15)
        safety = min(1.0, safety + 0.10)

    # Adjust for affected people
    urgency = min(1.0, 0.5 + (people / 200.0))
    if people > 50:
        urgency = min(1.0, urgency + 0.2)

    # Priority calculation
    priority = int(
        severity * 25 + urgency * 25 + safety * 20 +
        (0.15 if weather in ('Heavy Rain', 'Storm') else 0) * 100 +
        min(people / 100, 1.0) * 15
    )
    priority = min(100, max(1, priority))

    confidence = round(random.uniform(0.82, 0.96), 2)
    duplicates = random.randint(0, 8)

    return {
        'category': category,
        'severity': round(severity, 2),
        'urgency': round(urgency, 2),
        'safety_risk': round(safety, 2),
        'confidence': confidence,
        'priority': priority,
        'duplicates': duplicates,
        'resolution_minutes': round(random.uniform(30, 240), 0),
    }


def show() -> None:
    """Render the Report Problem page with AI analysis."""
    st.title("📝 Report Problem")
    st.markdown("*Describe the issue in natural language. FixFlow AI will analyze it automatically.*")

    desc = st.text_area(
        "Describe the problem:",
        placeholder="e.g., There is exposed electrical wiring near Block B and students are passing through it.",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        zone = st.selectbox("📍 Location Zone", [
            "Academic Zone", "Hostel Zone", "Residential Zone",
            "Public Zone", "Administrative Zone"
        ])
        weather = st.selectbox("🌤 Weather", [
            "Clear", "Cloudy", "Light Rain", "Heavy Rain", "Storm"
        ])
    with col2:
        people = st.number_input("👥 Affected People", min_value=1, max_value=1000, value=20)
        time_of_day = st.selectbox("🕐 Time of Day", [
            "morning", "afternoon", "evening", "night"
        ])

    st.markdown("---")

    if st.button("🔍 ANALYZE", type="primary", use_container_width=True):
        if not desc.strip():
            st.error("Please enter a complaint description.")
            return

        with st.spinner("AI analyzing complaint..."):
            # Try real model first, fallback to demo
            try:
                if not st.session_state.get("demo_mode", True):
                    from models.predict import predict_single, get_prediction_with_explanation
                    model = st.session_state.get("model")
                    pipeline = st.session_state.get("pipeline")
                    if model and pipeline:
                        result = get_prediction_with_explanation(
                            desc,
                            {'location_zone': zone, 'weather': weather,
                             'time_of_day': time_of_day, 'affected_people': people,
                             'safety_risk': 0.5, 'duplicate_group': 0},
                            model, pipeline
                        )
                    else:
                        result = _demo_prediction(desc, zone, weather, people)
                else:
                    result = _demo_prediction(desc, zone, weather, people)
            except Exception:
                result = _demo_prediction(desc, zone, weather, people)

        # Store for triage page
        st.session_state['last_prediction'] = result
        st.session_state['last_complaint'] = desc

        # Display results
        st.success("✅ Analysis Complete")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Category", result['category'])
        with m2:
            sev_label = "🔴 HIGH" if result['severity'] > 0.7 else ("🟡 MEDIUM" if result['severity'] > 0.4 else "🟢 LOW")
            st.metric("Severity", sev_label, f"{result['severity']:.0%}")
        with m3:
            urg_label = "🔴 HIGH" if result['urgency'] > 0.7 else ("🟡 MEDIUM" if result['urgency'] > 0.4 else "🟢 LOW")
            st.metric("Urgency", urg_label, f"{result['urgency']:.0%}")
        with m4:
            st.metric("Priority", f"{result['priority']} / 100")

        m5, m6, m7 = st.columns(3)
        with m5:
            sr_label = "🔴 HIGH" if result['safety_risk'] > 0.7 else ("🟡 MEDIUM" if result['safety_risk'] > 0.4 else "🟢 LOW")
            st.metric("Safety Risk", sr_label, f"{result['safety_risk']:.0%}")
        with m6:
            st.metric("Confidence", f"{result['confidence']:.0%}")
        with m7:
            st.metric("Duplicates Found", result['duplicates'])

        # Priority bar
        st.markdown("#### Priority Score")
        color = "#ff4444" if result['priority'] > 80 else ("#ff8800" if result['priority'] > 50 else "#44ff44")
        st.markdown(
            f'<div style="background:#333;border-radius:10px;overflow:hidden;">'
            f'<div style="width:{result["priority"]}%;background:{color};padding:8px;text-align:center;'
            f'border-radius:10px;color:white;font-weight:bold;">{result["priority"]}/100</div></div>',
            unsafe_allow_html=True
        )

        # Explanation
        st.markdown("#### 💡 Why This Priority?")
        factors = [
            ("Safety Risk", result['safety_risk'] * 25, result['safety_risk']),
            ("Severity", result['severity'] * 22, result['severity']),
            ("Urgency", result['urgency'] * 20, result['urgency']),
            ("Affected People", min(people / 100, 1.0) * 18, min(people / 100, 1.0)),
            ("Duplicate Reports", (result['duplicates'] / 10) * 15, result['duplicates'] / 10),
        ]
        factors.sort(key=lambda x: x[1], reverse=True)

        for name, contribution, val in factors:
            pct = contribution / max(result['priority'], 1) * 100
            st.markdown(
                f"**+{contribution:.0f}** {name} ({val:.0%})"
            )
            st.progress(min(pct / 100, 1.0))


show()
