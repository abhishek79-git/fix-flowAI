import random
from typing import Tuple
from evolution.chromosome import Chromosome

def sbx_crossover(p1: Chromosome, p2: Chromosome, seed: int = 42) -> Tuple[Chromosome, Chromosome]:
    """Simulated binary crossover for continuous, uniform crossover for discrete."""
    rng = random.Random(seed)
    
    c1 = Chromosome.random_chromosome(seed)
    c2 = Chromosome.random_chromosome(seed + 1)
    
    # Discrete crossover
    if rng.random() < 0.5:
        c1.hidden_1, c2.hidden_1 = p1.hidden_1, p2.hidden_1
    else:
        c1.hidden_1, c2.hidden_1 = p2.hidden_1, p1.hidden_1
        
    if rng.random() < 0.5:
        c1.hidden_2, c2.hidden_2 = p1.hidden_2, p2.hidden_2
    else:
        c1.hidden_2, c2.hidden_2 = p2.hidden_2, p1.hidden_2
        
    # SBX for continuous
    alpha = rng.random()
    c1.dropout = alpha * p1.dropout + (1 - alpha) * p2.dropout
    c2.dropout = (1 - alpha) * p1.dropout + alpha * p2.dropout
    
    alpha = rng.random()
    c1.learning_rate = alpha * p1.learning_rate + (1 - alpha) * p2.learning_rate
    c2.learning_rate = (1 - alpha) * p1.learning_rate + alpha * p2.learning_rate
    
    alpha = rng.random()
    c1.weight_decay = alpha * p1.weight_decay + (1 - alpha) * p2.weight_decay
    c2.weight_decay = (1 - alpha) * p1.weight_decay + alpha * p2.weight_decay
    
    alpha = rng.random()
    c1.huber_delta = alpha * p1.huber_delta + (1 - alpha) * p2.huber_delta
    c2.huber_delta = (1 - alpha) * p1.huber_delta + alpha * p2.huber_delta
    
    alpha = rng.random()
    c1.loss_weight_severity = alpha * p1.loss_weight_severity + (1 - alpha) * p2.loss_weight_severity
    c2.loss_weight_severity = (1 - alpha) * p1.loss_weight_severity + alpha * p2.loss_weight_severity
    
    alpha = rng.random()
    c1.loss_weight_resolution = alpha * p1.loss_weight_resolution + (1 - alpha) * p2.loss_weight_resolution
    c2.loss_weight_resolution = (1 - alpha) * p1.loss_weight_resolution + alpha * p2.loss_weight_resolution

    # Bound checking
    c1.dropout = max(0.05, min(0.5, c1.dropout))
    c2.dropout = max(0.05, min(0.5, c2.dropout))
    c1.learning_rate = max(0.0001, min(0.01, c1.learning_rate))
    c2.learning_rate = max(0.0001, min(0.01, c2.learning_rate))
    c1.weight_decay = max(0.00001, min(0.001, c1.weight_decay))
    c2.weight_decay = max(0.00001, min(0.001, c2.weight_decay))
    c1.huber_delta = max(0.5, min(2.0, c1.huber_delta))
    c2.huber_delta = max(0.5, min(2.0, c2.huber_delta))
    c1.loss_weight_severity = max(0.1, min(1.0, c1.loss_weight_severity))
    c2.loss_weight_severity = max(0.1, min(1.0, c2.loss_weight_severity))
    c1.loss_weight_resolution = max(0.1, min(1.0, c1.loss_weight_resolution))
    c2.loss_weight_resolution = max(0.1, min(1.0, c2.loss_weight_resolution))

    return c1, c2
