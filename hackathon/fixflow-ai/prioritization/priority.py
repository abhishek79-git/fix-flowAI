import pandas as pd
from typing import Dict, List, Any, Optional

def calculate_priority(
    severity: float, 
    urgency: float, 
    impact: float, 
    safety_risk: float, 
    duplicate_count: int, 
    resolution_cost: float, 
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate dynamic priority score 0-100."""
    if weights is None:
        weights = {
            'severity': 0.25,
            'urgency': 0.20,
            'impact': 0.20,
            'safety_risk': 0.25,
            'duplicate_count': 0.05,
            'resolution_cost': -0.05
        }
        
    score = (
        severity * weights.get('severity', 0.25) +
        urgency * weights.get('urgency', 0.20) +
        impact * weights.get('impact', 0.20) +
        safety_risk * weights.get('safety_risk', 0.25) +
        min(duplicate_count / 10.0, 1.0) * weights.get('duplicate_count', 0.05)
    )
    
    score += (1.0 - min(resolution_cost, 1.0)) * weights.get('resolution_cost', -0.05)
    
    return max(0.0, min(100.0, score * 100))

def calculate_priorities_batch(issues_df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Calculate priorities for a batch of issues."""
    priorities = []
    for _, row in issues_df.iterrows():
        p = calculate_priority(
            row.get('severity', 0.5),
            row.get('urgency', 0.5),
            row.get('impact', 0.5),
            row.get('safety_risk', 0.0),
            row.get('duplicate_count', 0),
            row.get('resolution_cost', 0.5),
            weights
        )
        priorities.append(p)
    return pd.Series(priorities, index=issues_df.index)

def adjust_priorities_for_weather(issues_df: pd.DataFrame, weather_condition: str) -> pd.DataFrame:
    """Adjust priorities based on weather conditions."""
    df = issues_df.copy()
    if 'priority' not in df.columns:
        df['priority'] = calculate_priorities_batch(df)
        
    if weather_condition.lower() in ['storm', 'heavy_rain', 'snow']:
        if 'category' in df.columns:
            mask = df['category'] == 'roof'
            df.loc[mask, 'priority'] = (df.loc[mask, 'priority'] * 1.2).clip(upper=100.0)
    return df

def get_priority_explanation(issue: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Get explanation of priority score calculation."""
    if weights is None:
        weights = {
            'severity': 0.25,
            'urgency': 0.20,
            'impact': 0.20,
            'safety_risk': 0.25,
            'duplicate_count': 0.05,
            'resolution_cost': -0.05
        }
        
    factors = []
    total_abs_weight = sum(abs(w) for w in weights.values())
    
    for key, value in issue.items():
        if key in weights:
            val = value
            if key == 'duplicate_count':
                val = min(value / 10.0, 1.0)
            elif key == 'resolution_cost':
                val = 1.0 - min(value, 1.0)
                
            contribution = val * weights[key]
            percentage = abs(weights[key]) / total_abs_weight * 100
            
            factors.append({
                'factor': key,
                'contribution': contribution * 100,
                'percentage': percentage
            })
            
    return factors
