"""
Issue Explorer Page for FixFlow AI.
Displays generated demo issues with filtering and sorting functionalities.
"""
import streamlit as st
import random
from typing import List, Dict, Any

CATEGORIES = [
    "Road Damage", "Electrical Hazard", "Water Leakage", 
    "Waste Management", "Streetlight Failure", "Drainage Blockage", 
    "Building Damage", "Public Safety"
]
ZONES = [
    "Academic Zone", "Hostel Zone", "Residential Zone", 
    "Public Zone", "Administrative Zone"
]

CATEGORY_ICONS = {
    "Road Damage": "🟠",
    "Electrical Hazard": "🔴",
    "Water Leakage": "🔵",
    "Waste Management": "🟤",
    "Streetlight Failure": "🟡",
    "Drainage Blockage": "⚫",
    "Building Damage": "🧱",
    "Public Safety": "🛡️"
}

def generate_demo_issues(num_issues: int = 30) -> List[Dict[str, Any]]:
    """
    Generates demo issues with proper attributes for the application.
    
    Args:
        num_issues (int): The number of issues to generate. Defaults to 30.
        
    Returns:
        List[Dict[str, Any]]: A list of generated issues.
    """
    random.seed(42)
    issues = []
    for i in range(num_issues):
        category = random.choice(CATEGORIES)
        zone = random.choice(ZONES)
        severity_score = random.uniform(0.1, 1.0)
        
        if severity_score > 0.8:
            severity = "Critical"
            color = "red"
        elif severity_score > 0.6:
            severity = "High"
            color = "orange"
        elif severity_score > 0.3:
            severity = "Medium"
            color = "yellow"
        else:
            severity = "Low"
            color = "green"
            
        priority = severity_score * 0.7 + random.uniform(0, 0.3)
            
        issues.append({
            "id": f"ISSUE-{1000+i}",
            "title": f"Reported {category} in {zone}",
            "category": category,
            "zone": zone,
            "severity_score": severity_score,
            "severity_level": severity,
            "color": color,
            "priority": min(priority, 1.0),
            "reports_count": random.randint(1, 15),
            "affected_people": random.randint(5, 500)
        })
    return issues

def show() -> None:
    """
    Displays the Issue Explorer page.
    """
    st.title("Issue Explorer")
    
    try:
        if 'demo_issues' not in st.session_state:
            st.session_state['demo_issues'] = generate_demo_issues()
            
        issues = st.session_state['demo_issues']
        
        st.sidebar.header("Filters")
        selected_categories = st.sidebar.multiselect("Category", CATEGORIES, default=CATEGORIES)
        selected_severities = st.sidebar.multiselect("Severity", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
        selected_zones = st.sidebar.multiselect("Zone", ZONES, default=ZONES)
        
        sort_by = st.sidebar.selectbox("Sort By", ["Priority", "Severity", "Recency"])
        
        filtered_issues = [
            i for i in issues 
            if i['category'] in selected_categories 
            and i['severity_level'] in selected_severities
            and i['zone'] in selected_zones
        ]
        
        if sort_by == "Priority":
            filtered_issues.sort(key=lambda x: x['priority'], reverse=True)
        elif sort_by == "Severity":
            filtered_issues.sort(key=lambda x: x['severity_score'], reverse=True)
        else:
            # Random sort for recency simulation
            random.seed(42)
            random.shuffle(filtered_issues)
            
        st.write(f"Showing {len(filtered_issues)} issues")
        
        for issue in filtered_issues:
            icon = CATEGORY_ICONS.get(issue['category'], "📌")
            
            with st.container():
                st.markdown(f"""
                <div style="border-left: 5px solid {issue['color']}; padding: 15px; margin-bottom: 15px; background-color: #f0f2f6; border-radius: 8px; color: black; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);">
                    <h4 style="margin-top: 0;">{icon} {issue['title']}</h4>
                    <p style="margin-bottom: 5px;"><strong>Location:</strong> {issue['zone']} | <strong>Category:</strong> {issue['category']} | <strong>ID:</strong> {issue['id']}</p>
                    <p style="margin-bottom: 0;">
                        <strong>Priority Score:</strong> {issue['priority']:.2f} | 
                        <strong>Reports:</strong> {issue['reports_count']} | 
                        <strong>Affected People:</strong> {issue['affected_people']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error loading issue explorer: {str(e)}")

if __name__ == "__main__":
    show()
