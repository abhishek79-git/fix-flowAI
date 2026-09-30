import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ConstraintViolation:
    constraint_name: str
    expected: Any
    actual: Any
    message: str

def check_model_constraints(param_count: int, latency_ms: float, ece: float, group_gap: float, config: Dict[str, Any]) -> Dict[str, bool]:
    """Check if a model satisfies constraints."""
    max_params = config.get('max_param_count', float('inf'))
    max_latency = config.get('max_latency_ms', float('inf'))
    max_ece = config.get('max_ece', float('inf'))
    max_group_gap = config.get('max_group_gap', float('inf'))
    
    return {
        'param_count': param_count <= max_params,
        'latency_ms': latency_ms <= max_latency,
        'ece': ece <= max_ece,
        'group_gap': group_gap <= max_group_gap
    }

def check_resource_constraints(workers_needed: int, vehicles_needed: int, time_needed: float, available: Dict[str, Any]) -> bool:
    """Check if resource constraints are met."""
    return (
        workers_needed <= available.get('workers', 0) and
        vehicles_needed <= available.get('vehicles', 0) and
        time_needed <= available.get('time_hours', 0.0)
    )
