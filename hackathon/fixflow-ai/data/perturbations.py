import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from dataclasses import dataclass
import random
import string
import yaml

@dataclass
class DriftReport:
    """Report containing drift metrics and dataset changes."""
    scenario_name: str
    description: str
    num_records: int
    feature_changes: Dict[str, Any]
    psi_scores: Dict[str, float]
    performance_impact: Optional[Dict[str, float]]

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Computes the Population Stability Index (PSI) for a continuous variable.
    
    Args:
        reference: Reference distribution array.
        current: Current distribution array.
        bins: Number of bins for histogram.
        
    Returns:
        PSI float value.
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    
    hist_ref, bin_edges = np.histogram(reference, bins=bins, density=False)
    hist_cur, _ = np.histogram(current, bins=bin_edges, density=False)
    
    p_ref = hist_ref / np.sum(hist_ref)
    p_cur = hist_cur / np.sum(hist_cur)
    
    p_ref = np.where(p_ref == 0, 1e-5, p_ref)
    p_cur = np.where(p_cur == 0, 1e-5, p_cur)
    
    psi_values = (p_cur - p_ref) * np.log(p_cur / p_ref)
    return float(np.sum(psi_values))

def missing_fields(df: pd.DataFrame, rate: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """Randomly removes fields like location_zone, affected_people, weather.
    
    Args:
        df: Input DataFrame.
        rate: Rate of missingness.
        seed: Random seed.
        
    Returns:
        DataFrame with missing fields.
    """
    np.random.seed(seed)
    res_df = df.copy()
    
    cols_to_drop = ['location_zone', 'affected_people', 'weather']
    for col in cols_to_drop:
        mask = np.random.rand(len(res_df)) < rate
        res_df.loc[mask, col] = np.nan
        
    return res_df

def text_noise(df: pd.DataFrame, rate: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Adds typos, extra punctuation, case changes to text.
    
    Args:
        df: Input DataFrame.
        rate: Rate of noise injection.
        seed: Random seed.
        
    Returns:
        DataFrame with noisy text.
    """
    random.seed(seed)
    res_df = df.copy()
    
    def corrupt(text: Any) -> Any:
        if not isinstance(text, str) or random.random() > rate:
            return text
        
        corr_type = random.choice(['upper', 'lower', 'typo', 'punct'])
        if corr_type == 'upper':
            return text.upper()
        elif corr_type == 'lower':
            return text.lower()
        elif corr_type == 'punct':
            return text + random.choice(string.punctuation) * 3
        else:
            if len(text) > 3:
                idx = random.randint(0, len(text)-2)
                return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
            return text
            
    res_df['complaint_text'] = res_df['complaint_text'].apply(corrupt)
    return res_df

def numeric_shift(df: pd.DataFrame, factor: float = 2.0, seed: int = 42) -> pd.DataFrame:
    """Shifts affected_people and resolution time distributions.
    
    Args:
        df: Input DataFrame.
        factor: Shift multiplier factor.
        seed: Random seed.
        
    Returns:
        DataFrame with shifted numeric fields.
    """
    np.random.seed(seed)
    res_df = df.copy()
    
    res_df['affected_people'] = res_df['affected_people'] * factor
    res_df['estimated_resolution_minutes'] = res_df['estimated_resolution_minutes'] * factor
    
    return res_df

def heavy_rain_scenario(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Changes weather to Heavy Rain/Storm, increases water/drainage severity.
    
    Args:
        df: Input DataFrame.
        seed: Random seed.
        
    Returns:
        DataFrame simulating a heavy rain scenario.
    """
    random.seed(seed)
    np.random.seed(seed)
    res_df = df.copy()
    
    res_df['weather'] = np.random.choice(['Heavy Rain', 'Storm'], size=len(res_df))
    
    water_mask = res_df['category'].isin(['Water Leakage', 'Drainage Blockage'])
    res_df.loc[water_mask, 'severity'] = np.clip(res_df.loc[water_mask, 'severity'] + 0.3, 0.0, 1.0)
    
    return res_df

def complaint_surge(df: pd.DataFrame, factor: int = 10, seed: int = 42) -> pd.DataFrame:
    """Multiplies complaint count, increases duplicate density.
    
    Args:
        df: Input DataFrame.
        factor: Surge multiplier.
        seed: Random seed.
        
    Returns:
        DataFrame simulating a complaint surge.
    """
    np.random.seed(seed)
    res_df = pd.concat([df] * factor, ignore_index=True)
    res_df['ticket_id'] = [f"SURGE-TKT-{i:06d}" for i in range(len(res_df))]
    
    res_df['duplicate_group'] = np.random.choice(res_df['duplicate_group'].unique(), size=len(res_df))
    
    return res_df

def create_stress_dataset(df: pd.DataFrame, scenario_name: str, seed: int = 42) -> pd.DataFrame:
    """Creates a stress dataset based on the requested scenario.
    
    Args:
        df: Input DataFrame.
        scenario_name: Name of the scenario (missing, noise, shift, rain, surge).
        seed: Random seed.
        
    Returns:
        Stressed DataFrame.
    """
    if scenario_name == 'missing':
        return missing_fields(df, seed=seed)
    elif scenario_name == 'noise':
        return text_noise(df, seed=seed)
    elif scenario_name == 'shift':
        return numeric_shift(df, seed=seed)
    elif scenario_name == 'rain':
        return heavy_rain_scenario(df, seed=seed)
    elif scenario_name == 'surge':
        return complaint_surge(df, seed=seed)
    else:
        return df.copy()
