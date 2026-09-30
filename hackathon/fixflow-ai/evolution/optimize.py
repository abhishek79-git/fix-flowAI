import yaml
import os
import copy
from typing import Any, Dict
from evolution.population import initialize_population, evaluate_population, get_best_feasible, get_pareto_front, save_generation
from evolution.nsga2 import non_dominated_sort, tournament_selection
from evolution.crossover import sbx_crossover
from evolution.mutation import polynomial_mutation

def run_evolution(train_df: Any, val_df: Any, ood_df: Any, config_path: str = 'config.yaml', seed: int = 42) -> Dict[str, Any]:
    """Run the main NSGA-II evolutionary optimization loop."""
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
    pop_size = config.get('population_size', 20)
    generations = config.get('generations', 10)
    
    pop = initialize_population(size=pop_size, seed=seed)
    pop = evaluate_population(pop, train_df, val_df, ood_df, config)
    
    best_feasible = get_best_feasible(pop)
    all_generations = [copy.deepcopy(pop)]
    convergence_history = []
    
    for gen in range(1, generations + 1):
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(pop, seed=seed + gen + len(offspring))
            p2 = tournament_selection(pop, seed=seed + gen + len(offspring) + 1)
            
            c1, c2 = sbx_crossover(p1, p2, seed=seed + gen + len(offspring) + 2)
            c1 = polynomial_mutation(c1, seed=seed + gen + len(offspring) + 3)
            c2 = polynomial_mutation(c2, seed=seed + gen + len(offspring) + 4)
            
            c1.generation = gen
            c2.generation = gen
            
            offspring.extend([c1, c2])
            
        # Evaluate offspring
        offspring = evaluate_population(offspring, train_df, val_df, ood_df, config)
        
        # Combine
        combined_pop = pop + offspring
        
        # Select next generation
        fronts = non_dominated_sort(combined_pop)
        next_pop = []
        
        for front in fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend(front)
            else:
                remaining = pop_size - len(next_pop)
                next_pop.extend(front[:remaining])
                break
                
        pop = next_pop
        all_generations.append(copy.deepcopy(pop))
        
        # Save generation
        save_generation(pop, gen, f'generation_{gen}.json')
        
        # Keep track of best feasible
        current_best = get_best_feasible(pop)
        if current_best and (not best_feasible or (current_best.fitness and best_feasible.fitness and current_best.fitness.id_macro_f1 > best_feasible.fitness.id_macro_f1)):
            best_feasible = copy.deepcopy(current_best)
            
        if best_feasible and best_feasible.fitness:
            convergence_history.append(best_feasible.fitness.id_macro_f1)
            
    pareto_front = get_pareto_front(pop)
    
    return {
        'best_chromosome': best_feasible,
        'pareto_front': pareto_front,
        'convergence_history': convergence_history,
        'all_generations': all_generations
    }
