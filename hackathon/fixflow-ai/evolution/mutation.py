import random
from evolution.chromosome import Chromosome

def polynomial_mutation(c: Chromosome, rate: float = 0.2, seed: int = 42) -> Chromosome:
    """Bounded mutation respecting parameter ranges."""
    rng = random.Random(seed)
    
    if rng.random() < rate:
        c.hidden_1 = rng.randint(32, 256)
    if rng.random() < rate:
        c.hidden_2 = rng.randint(16, 128)
        
    if rng.random() < rate:
        delta = rng.uniform(-0.1, 0.1)
        c.dropout = max(0.05, min(0.5, c.dropout + delta))
        
    if rng.random() < rate:
        delta = rng.uniform(-0.001, 0.001)
        c.learning_rate = max(0.0001, min(0.01, c.learning_rate + delta))
        
    if rng.random() < rate:
        delta = rng.uniform(-0.0001, 0.0001)
        c.weight_decay = max(0.00001, min(0.001, c.weight_decay + delta))
        
    if rng.random() < rate:
        delta = rng.uniform(-0.2, 0.2)
        c.huber_delta = max(0.5, min(2.0, c.huber_delta + delta))
        
    if rng.random() < rate:
        delta = rng.uniform(-0.1, 0.1)
        c.loss_weight_severity = max(0.1, min(1.0, c.loss_weight_severity + delta))
        
    if rng.random() < rate:
        delta = rng.uniform(-0.1, 0.1)
        c.loss_weight_resolution = max(0.1, min(1.0, c.loss_weight_resolution + delta))
        
    return c
