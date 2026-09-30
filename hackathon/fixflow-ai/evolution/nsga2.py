import random
from typing import List
from evolution.chromosome import Chromosome, FitnessVector

def dominates(a: FitnessVector, b: FitnessVector) -> bool:
    """Check if fitness vector a dominates b."""
    objs_a = [-a.id_macro_f1, -a.ood_macro_f1, a.ece, a.group_gap, a.latency_ms, a.param_count, -a.seed_stability]
    objs_b = [-b.id_macro_f1, -b.ood_macro_f1, b.ece, b.group_gap, b.latency_ms, b.param_count, -b.seed_stability]
    
    better_in_at_least_one = False
    for oa, ob in zip(objs_a, objs_b):
        if oa > ob:
            return False
        if oa < ob:
            better_in_at_least_one = True
    return better_in_at_least_one

def non_dominated_sort(pop: List[Chromosome]) -> List[List[Chromosome]]:
    """Sort population into non-dominated fronts."""
    fronts: List[List[Chromosome]] = [[]]
    domination_counts = {id(c): 0 for c in pop}
    dominated_lists = {id(c): [] for c in pop}
    
    for p in pop:
        for q in pop:
            if not p.fitness or not q.fitness:
                continue
            if dominates(p.fitness, q.fitness):
                dominated_lists[id(p)].append(q)
            elif dominates(q.fitness, p.fitness):
                domination_counts[id(p)] += 1
        
        if domination_counts[id(p)] == 0:
            fronts[0].append(p)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_lists[id(p)]:
                domination_counts[id(q)] -= 1
                if domination_counts[id(q)] == 0:
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break
            
    return fronts

def crowding_distance(front: List[Chromosome]) -> List[float]:
    """Calculate crowding distance for a front."""
    size = len(front)
    distances = [0.0] * size
    if size <= 2:
        return [float('inf')] * size
        
    objs = [
        lambda c: -c.fitness.id_macro_f1 if c.fitness else 0.0,
        lambda c: -c.fitness.ood_macro_f1 if c.fitness else 0.0,
        lambda c: c.fitness.ece if c.fitness else 0.0,
        lambda c: c.fitness.group_gap if c.fitness else 0.0,
        lambda c: c.fitness.latency_ms if c.fitness else 0.0,
        lambda c: float(c.fitness.param_count) if c.fitness else 0.0,
        lambda c: -c.fitness.seed_stability if c.fitness else 0.0
    ]
    
    for obj_idx, obj_func in enumerate(objs):
        sorted_indices = sorted(range(size), key=lambda i: obj_func(front[i]))
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        min_obj = obj_func(front[sorted_indices[0]])
        max_obj = obj_func(front[sorted_indices[-1]])
        if max_obj == min_obj:
            continue
            
        for i in range(1, size - 1):
            distances[sorted_indices[i]] += (
                obj_func(front[sorted_indices[i+1]]) - obj_func(front[sorted_indices[i-1]])
            ) / (max_obj - min_obj)
            
    return distances

def tournament_selection(pop: List[Chromosome], tournament_size: int = 3, seed: int = 42) -> Chromosome:
    """Select a chromosome using tournament selection."""
    rng = random.Random(seed)
    tournament = rng.sample(pop, tournament_size)
    tournament.sort(key=lambda c: -c.fitness.id_macro_f1 if c.fitness else 0.0)
    return tournament[0]

def check_constraints(c: Chromosome, config: dict) -> bool:
    """Check if chromosome meets constraints."""
    if not c.fitness:
        return False
    max_params = config.get('max_param_count', float('inf'))
    max_latency = config.get('max_latency_ms', float('inf'))
    if c.fitness.param_count > max_params or c.fitness.latency_ms > max_latency:
        return False
    return True
