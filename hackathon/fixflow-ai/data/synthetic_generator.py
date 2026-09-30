import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict

def generate_templates() -> dict:
    """Generates 15 complaint templates per category."""
    templates = {}
    categories = [
        'Road Damage', 'Electrical Hazard', 'Water Leakage', 'Waste Management', 
        'Streetlight Failure', 'Drainage Blockage', 'Building Damage', 'Public Safety'
    ]
    
    for cat in categories:
        cat_lower = cat.lower()
        templates[cat] = [
            f"There is a {cat_lower} near the {{location}}.",
            f"Please fix the {cat_lower} at {{location}}.",
            f"Urgent: {cat_lower} reported in the {{location}}.",
            f"I noticed a severe {cat_lower} when passing by {{location}}.",
            f"The {cat_lower} at {{location}} is getting worse.",
            f"Can someone look into the {cat_lower} at {{location}}?",
            f"Major {cat_lower} spotted in the {{location}}.",
            f"Danger: {cat_lower} near {{location}}.",
            f"We have a {cat_lower} issue at {{location}}.",
            f"Reporting a {cat_lower} that occurred during {{weather}}.",
            f"The recent {{weather}} caused {cat_lower} at {{location}}.",
            f"Huge {cat_lower} at {{location}}, needs immediate attention.",
            f"Minor {cat_lower} observed in {{location}}.",
            f"Continuous {cat_lower} problem at {{location}}.",
            f"Fix required for {cat_lower} in {{location}}."
        ]
    return templates

def generate_dataset(seed: int = 42, num_samples: int = 5000) -> pd.DataFrame:
    """Generates synthetic complaint records.
    
    Args:
        seed: Random seed for reproducibility.
        num_samples: Total number of records to generate.
        
    Returns:
        pd.DataFrame containing the generated dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    templates = generate_templates()
    categories = list(templates.keys())
    zones = ['Academic Zone', 'Hostel Zone', 'Residential Zone', 'Public Zone', 'Administrative Zone']
    weathers = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Storm']
    times = ['morning', 'afternoon', 'evening', 'night']
    
    records = []
    
    for i in range(num_samples):
        cat = random.choice(categories)
        zone = random.choice(zones)
        weather = random.choice(weathers)
        tod = random.choice(times)
        
        # Correlations: Heavy rain -> water leakage/drainage
        if weather in ['Heavy Rain', 'Storm'] and random.random() < 0.6:
            cat = random.choice(['Water Leakage', 'Drainage Blockage'])
        
        severity = random.uniform(0, 1)
        urgency = random.uniform(0, 1)
        safety_risk = random.uniform(0, 1)
        
        # Correlations: Electrical -> high safety risk
        if cat == 'Electrical Hazard':
            safety_risk = random.uniform(0.7, 1.0)
            urgency = random.uniform(0.7, 1.0)
            
        affected = int(random.lognormvariate(2, 1))
        
        # Correlations: More affected -> higher urgency
        if affected > 20:
            urgency = min(1.0, urgency + 0.3)
            
        text = random.choice(templates[cat]).format(location=zone, weather=weather)
        
        workers = random.randint(1, 3)
        vehicle = random.choice([True, False])
        res_mins = random.uniform(30, 300)
        
        records.append({
            'ticket_id': f"TKT-{i:05d}",
            'complaint_text': text,
            'category': cat,
            'severity': severity,
            'urgency': urgency,
            'safety_risk': safety_risk,
            'location_zone': zone,
            'weather': weather,
            'time_of_day': tod,
            'affected_people': affected,
            'duplicate_group': i,
            'estimated_resolution_minutes': res_mins,
            'workers_required': workers,
            'vehicle_required': vehicle
        })
        
    df = pd.DataFrame(records)
    
    # Introduce duplicates
    num_duplicates = int(num_samples * 0.1)
    dup_indices = np.random.choice(df.index, num_duplicates, replace=False)
    for idx in dup_indices:
        df.loc[idx, 'duplicate_group'] = random.choice(df['duplicate_group'].unique())
        
    return df

def split_dataset(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into train, id_validation, and ood_validation.
    
    Args:
        df: Input DataFrame.
        seed: Random seed.
        
    Returns:
        Tuple of (train, id_val, ood_val) DataFrames.
    """
    np.random.seed(seed)
    
    ood_mask = df['weather'].isin(['Heavy Rain', 'Storm']) | (df['affected_people'] > 50)
    ood_pool = df[ood_mask]
    id_pool = df[~ood_mask]
    
    ood_val_size = int(len(df) * 0.15)
    if len(ood_pool) > ood_val_size:
        ood_val = ood_pool.sample(n=ood_val_size, random_state=seed)
        remaining_ood = ood_pool.drop(ood_val.index)
        id_pool = pd.concat([id_pool, remaining_ood])
    else:
        ood_val = ood_pool
    
    id_val_size = int(len(df) * 0.15)
    id_val = id_pool.sample(n=id_val_size, random_state=seed)
    train = id_pool.drop(id_val.index)
    
    return train, id_val, ood_val

def load_or_generate(data_dir: str = 'data/processed', seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Loads dataset from disk or generates it if it doesn't exist.
    
    Args:
        data_dir: Directory to save/load processed data.
        seed: Random seed.
        
    Returns:
        Dictionary with train, id_val, and ood_val DataFrames.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, 'train.csv')
    id_val_path = os.path.join(data_dir, 'id_val.csv')
    ood_val_path = os.path.join(data_dir, 'ood_val.csv')
    
    if os.path.exists(train_path) and os.path.exists(id_val_path) and os.path.exists(ood_val_path):
        return {
            'train': pd.read_csv(train_path),
            'id_val': pd.read_csv(id_val_path),
            'ood_val': pd.read_csv(ood_val_path)
        }
        
    df = generate_dataset(seed=seed)
    train, id_val, ood_val = split_dataset(df, seed=seed)
    
    train.to_csv(train_path, index=False)
    id_val.to_csv(id_val_path, index=False)
    ood_val.to_csv(ood_val_path, index=False)
    
    return {
        'train': train,
        'id_val': id_val,
        'ood_val': ood_val
    }
