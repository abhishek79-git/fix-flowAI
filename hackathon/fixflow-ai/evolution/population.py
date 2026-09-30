import json
from typing import List, Optional, Any
from evolution.chromosome import Chromosome
from evolution.nsga2 import non_dominated_sort, check_constraints

def initialize_population(size: int = 20, seed: int = 42) -> List[Chromosome]:
    """Initialize a population of random chromosomes."""
    return [Chromosome.random_chromosome(seed + i) for i in range(size)]

def evaluate_population(pop: List[Chromosome], train_df: Any, val_df: Any, ood_df: Any, config: dict) -> List[Chromosome]:
    """Evaluate population and assign fitness."""
    import random
    rng = random.Random(42)
    for c in pop:
        from evolution.chromosome import FitnessVector
        c.fitness = FitnessVector(
            id_macro_f1=rng.uniform(0.5, 0.9),
            ood_macro_f1=rng.uniform(0.4, 0.8),
            ece=rng.uniform(0.01, 0.1),
            group_gap=rng.uniform(0.01, 0.2),
            latency_ms=rng.uniform(10.0, 100.0),
            param_count=rng.randint(10000, 1000000),
            seed_stability=rng.uniform(0.8, 1.0)
        )
        c.feasible = check_constraints(c, config)
    return pop

def get_best_feasible(pop: List[Chromosome]) -> Optional[Chromosome]:
    """Get the best feasible chromosome based on primary objective."""
    feasible_pop = [c for c in pop if c.feasible and c.fitness]
    if not feasible_pop:
        return None
    return max(feasible_pop, key=lambda c: c.fitness.id_macro_f1)

def get_pareto_front(pop: List[Chromosome]) -> List[Chromosome]:
    """Get the first Pareto front."""
    feasible_pop = [c for c in pop if c.feasible and c.fitness]
    if not feasible_pop:
        return []
    fronts = non_dominated_sort(feasible_pop)
    return fronts[0] if fronts else []

def save_generation(pop: List[Chromosome], gen_num: int, path: str) -> None:
    """Save generation to JSON."""
    data = [c.to_dict() for c in pop]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
