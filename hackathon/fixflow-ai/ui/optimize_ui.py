"""
Resource Optimizer Page for FixFlow AI.
Simulates assigning limited resources to prioritize issues.
"""
import streamlit as st
import random
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """
    Stores the result of a resource optimization run.
    """
    assigned_issues: List[Dict[str, Any]]
    rejected_issues: List[Dict[str, Any]]
    workers_used: int
    vehicles_used: int
    time_used: int
    total_priority_resolved: float

def generate_optimizer_issues() -> List[Dict[str, Any]]:
    """
    Generates demo issues with resource requirements.
    
    Returns:
        List[Dict[str, Any]]: A list of issues requiring resources.
    """
    random.seed(42)
    categories = ["Road Damage", "Electrical Hazard", "Water Leakage"]
    issues = []
    for i in range(15):
        category = random.choice(categories)
        priority = random.uniform(0.3, 1.0)
        issues.append({
            "id": f"OPT-{2000+i}",
            "title": f"{category} Repair",
            "priority": priority,
            "workers_needed": random.randint(1, 4),
            "vehicles_needed": random.randint(0, 2),
            "time_needed": random.randint(1, 4),
            "reason": ""
        })
    return sorted(issues, key=lambda x: x['priority'], reverse=True)

def optimize_resources(
    issues: List[Dict[str, Any]], 
    max_workers: int, 
    max_vehicles: int, 
    max_time: int
) -> OptimizationResult:
    """
    Simple greedy optimizer to allocate resources to issues based on priority.
    
    Args:
        issues (List[Dict[str, Any]]): List of issues to process.
        max_workers (int): Maximum available workers.
        max_vehicles (int): Maximum available vehicles.
        max_time (int): Maximum available time in hours.
        
    Returns:
        OptimizationResult: The outcome of the resource allocation.
    """
    assigned = []
    rejected = []
    
    w_used, v_used, t_used = 0, 0, 0
    priority_resolved = 0.0
    
    for issue in issues:
        if (w_used + issue['workers_needed'] <= max_workers and
            v_used + issue['vehicles_needed'] <= max_vehicles and
            t_used + issue['time_needed'] <= max_time):
            
            assigned.append(issue)
            w_used += issue['workers_needed']
            v_used += issue['vehicles_needed']
            t_used += issue['time_needed']
            priority_resolved += issue['priority']
        else:
            reason = []
            if w_used + issue['workers_needed'] > max_workers: reason.append("Workers")
            if v_used + issue['vehicles_needed'] > max_vehicles: reason.append("Vehicles")
            if t_used + issue['time_needed'] > max_time: reason.append("Time")
            issue['reason'] = "Insufficient " + ", ".join(reason)
            rejected.append(issue)
            
    return OptimizationResult(
        assigned_issues=assigned, 
        rejected_issues=rejected, 
        workers_used=w_used, 
        vehicles_used=v_used, 
        time_used=t_used, 
        total_priority_resolved=priority_resolved
    )

def show() -> None:
    """
    Displays the Resource Optimizer page.
    """
    st.title("Resource Optimizer")
    
    try:
        st.sidebar.header("Available Resources")
        workers = st.sidebar.slider("Workers", 1, 10, 5)
        vehicles = st.sidebar.slider("Vehicles", 1, 5, 2)
        time_hours = st.sidebar.slider("Time (Hours)", 1, 8, 4)
        
        if st.button("OPTIMIZE PLAN", type="primary"):
            issues = generate_optimizer_issues()
            result = optimize_resources(issues, workers, vehicles, time_hours)
            
            st.header("Optimization Results")
            st.metric("Total Priority Resolved", f"{result.total_priority_resolved:.2f}")
            
            st.subheader("Resource Utilization")
            st.progress(
                min(result.workers_used / workers if workers > 0 else 0, 1.0), 
                text=f"Workers: {result.workers_used}/{workers}"
            )
            st.progress(
                min(result.vehicles_used / vehicles if vehicles > 0 else 0, 1.0), 
                text=f"Vehicles: {result.vehicles_used}/{vehicles}"
            )
            st.progress(
                min(result.time_used / time_hours if time_hours > 0 else 0, 1.0), 
                text=f"Time: {result.time_used}/{time_hours} hrs"
            )
            
            st.subheader("Action Plan (Assigned Issues)")
            if result.assigned_issues:
                for idx, issue in enumerate(result.assigned_issues, 1):
                    st.success(f"**{idx}. {issue['title']}** (ID: {issue['id']}) - Priority: {issue['priority']:.2f} | "
                               f"Requires: {issue['workers_needed']}W, {issue['vehicles_needed']}V, {issue['time_needed']}H")
            else:
                st.info("No issues could be assigned with the given resources.")
                
            st.subheader("Rejected Issues")
            if result.rejected_issues:
                for issue in result.rejected_issues:
                    st.error(f"**{issue['title']}** (ID: {issue['id']}) - Reason: {issue['reason']}")
            else:
                st.info("All issues were assigned.")
    except Exception as e:
        st.error(f"Error optimizing resources: {str(e)}")

if __name__ == "__main__":
    show()
