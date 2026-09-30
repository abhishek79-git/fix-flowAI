"""FixFlow AI Resource Optimizer.

Constrained resource optimization using greedy selection + local swap improvement.
Solves: max Σ Pᵢxᵢ subject to worker, vehicle, and time constraints.
"""

from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class OptimizationResult:
    """Result of resource optimization."""
    selected_issues: list[dict[str, Any]]
    rejected_issues: list[dict[str, Any]]
    total_priority_resolved: float
    resource_utilization: dict[str, float]
    explanation: list[str]


def greedy_optimize(
    issues: list[dict[str, Any]],
    workers: int,
    vehicles: int,
    time_hours: float
) -> OptimizationResult:
    """Select issues greedily by priority while respecting resource constraints.
    
    Args:
        issues: List of issue dicts with priority, workers_needed, vehicles_needed, time_needed.
        workers: Available workers.
        vehicles: Available vehicles.
        time_hours: Available time in hours.
        
    Returns:
        OptimizationResult with selected and rejected issues.
    """
    sorted_issues = sorted(issues, key=lambda x: x.get('priority', 0), reverse=True)
    
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    w_used = 0.0
    v_used = 0.0
    t_used = 0.0
    total_priority = 0.0
    explanations: list[str] = []
    
    for issue in sorted_issues:
        w_req = issue.get('workers_needed', 1)
        v_req = 1 if issue.get('vehicle_required', False) else 0
        t_req = issue.get('time_needed', issue.get('estimated_resolution_minutes', 60) / 60.0)
        
        if (w_used + w_req <= workers and 
            v_used + v_req <= vehicles and 
            t_used + t_req <= time_hours):
            selected.append(issue)
            w_used += w_req
            v_used += v_req
            t_used += t_req
            total_priority += issue.get('priority', 0.0)
            explanations.append(
                f"Selected: {issue.get('category', 'Unknown')} at {issue.get('location_zone', '?')} "
                f"(priority={issue.get('priority', 0):.1f}, workers={w_req}, time={t_req:.1f}h)"
            )
        else:
            rejected.append(issue)
            reason_parts = []
            if w_used + w_req > workers:
                reason_parts.append(f"workers ({w_used+w_req:.0f}/{workers})")
            if v_used + v_req > vehicles:
                reason_parts.append(f"vehicles ({v_used+v_req:.0f}/{vehicles})")
            if t_used + t_req > time_hours:
                reason_parts.append(f"time ({t_used+t_req:.1f}/{time_hours}h)")
            explanations.append(
                f"Rejected: {issue.get('category', 'Unknown')} — exceeds {', '.join(reason_parts)}"
            )
    
    return OptimizationResult(
        selected_issues=selected,
        rejected_issues=rejected,
        total_priority_resolved=total_priority,
        resource_utilization={
            'workers_used': w_used,
            'workers_available': float(workers),
            'vehicles_used': v_used,
            'vehicles_available': float(vehicles),
            'time_used': t_used,
            'time_available': time_hours,
            'workers_pct': (w_used / workers * 100) if workers > 0 else 0,
            'vehicles_pct': (v_used / vehicles * 100) if vehicles > 0 else 0,
            'time_pct': (t_used / time_hours * 100) if time_hours > 0 else 0,
        },
        explanation=explanations
    )


def local_improvement(
    result: OptimizationResult,
    all_issues: list[dict[str, Any]]
) -> OptimizationResult:
    """Try swapping rejected high-priority issues with selected low-priority ones.
    
    For each rejected issue (sorted by priority desc), try swapping it with
    each selected issue (sorted by priority asc). If the swap improves total
    priority while remaining feasible, accept it.
    
    Args:
        result: Initial optimization result from greedy.
        all_issues: All issues for reference.
        
    Returns:
        Improved OptimizationResult.
    """
    selected = list(result.selected_issues)
    rejected = list(result.rejected_issues)
    total_priority = result.total_priority_resolved
    util = dict(result.resource_utilization)
    improved = False
    
    workers_avail = util.get('workers_available', 5)
    vehicles_avail = util.get('vehicles_available', 2)
    time_avail = util.get('time_available', 4.0)
    
    # Sort rejected by priority desc, selected by priority asc for swap candidates
    rejected_sorted = sorted(rejected, key=lambda x: x.get('priority', 0), reverse=True)
    
    for rej_issue in rejected_sorted:
        rej_priority = rej_issue.get('priority', 0)
        rej_w = rej_issue.get('workers_needed', 1)
        rej_v = 1 if rej_issue.get('vehicle_required', False) else 0
        rej_t = rej_issue.get('time_needed', rej_issue.get('estimated_resolution_minutes', 60) / 60.0)
        
        selected_sorted = sorted(selected, key=lambda x: x.get('priority', 0))
        
        for sel_issue in selected_sorted:
            sel_priority = sel_issue.get('priority', 0)
            if rej_priority <= sel_priority:
                continue  # No improvement possible
            
            sel_w = sel_issue.get('workers_needed', 1)
            sel_v = 1 if sel_issue.get('vehicle_required', False) else 0
            sel_t = sel_issue.get('time_needed', sel_issue.get('estimated_resolution_minutes', 60) / 60.0)
            
            # Check if swap is feasible
            new_w = util['workers_used'] - sel_w + rej_w
            new_v = util['vehicles_used'] - sel_v + rej_v
            new_t = util['time_used'] - sel_t + rej_t
            
            if new_w <= workers_avail and new_v <= vehicles_avail and new_t <= time_avail:
                # Accept swap
                selected.remove(sel_issue)
                selected.append(rej_issue)
                rejected.remove(rej_issue)
                rejected.append(sel_issue)
                total_priority = total_priority - sel_priority + rej_priority
                util['workers_used'] = new_w
                util['vehicles_used'] = new_v
                util['time_used'] = new_t
                improved = True
                break
    
    if improved:
        util['workers_pct'] = (util['workers_used'] / workers_avail * 100) if workers_avail > 0 else 0
        util['vehicles_pct'] = (util['vehicles_used'] / vehicles_avail * 100) if vehicles_avail > 0 else 0
        util['time_pct'] = (util['time_used'] / time_avail * 100) if time_avail > 0 else 0
    
    explanations = list(result.explanation)
    if improved:
        explanations.append("Local improvement: swapped lower-priority selected issues for higher-priority rejected ones.")
    else:
        explanations.append("Local improvement: no beneficial swaps found.")
    
    return OptimizationResult(
        selected_issues=selected,
        rejected_issues=rejected,
        total_priority_resolved=total_priority,
        resource_utilization=util,
        explanation=explanations
    )


def optimize_resources(
    issues_df: pd.DataFrame,
    workers: int = 5,
    vehicles: int = 2,
    time_hours: float = 4.0
) -> OptimizationResult:
    """Full resource optimization pipeline.
    
    Args:
        issues_df: DataFrame with issue data including priority scores.
        workers: Number of available workers.
        vehicles: Number of available vehicles.
        time_hours: Available time budget in hours.
        
    Returns:
        OptimizationResult with selected feasible action plan.
    """
    if issues_df.empty:
        return OptimizationResult(
            selected_issues=[], rejected_issues=[],
            total_priority_resolved=0.0,
            resource_utilization={'workers_used': 0, 'vehicles_used': 0, 'time_used': 0,
                                  'workers_available': float(workers), 'vehicles_available': float(vehicles),
                                  'time_available': time_hours,
                                  'workers_pct': 0, 'vehicles_pct': 0, 'time_pct': 0},
            explanation=['No issues to optimize.']
        )
    
    issues = issues_df.to_dict('records')
    greedy_result = greedy_optimize(issues, workers, vehicles, time_hours)
    final_result = local_improvement(greedy_result, issues)
    return final_result
