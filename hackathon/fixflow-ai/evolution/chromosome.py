from dataclasses import dataclass
import random
from typing import Any, Dict

@dataclass
class FitnessVector:
    """Fitness vector for multi-objective optimization."""
    id_macro_f1: float
    ood_macro_f1: float
    ece: float           # lower better
    group_gap: float     # lower better
    latency_ms: float    # lower better
    param_count: int     # lower better
    seed_stability: float  # higher better

@dataclass
class Chromosome:
    """Genetic representation of model hyperparameters."""
    hidden_1: int          # 32-256
    hidden_2: int          # 16-128
    dropout: float         # 0.05-0.5
    learning_rate: float   # 0.0001-0.01
    weight_decay: float    # 0.00001-0.001
    huber_delta: float     # 0.5-2.0
    loss_weight_severity: float  # 0.1-1.0
    loss_weight_resolution: float  # 0.1-1.0
    
    fitness: FitnessVector | None = None
    feasible: bool = True
    generation: int = 0

    @classmethod
    def random_chromosome(cls, seed: int = 42) -> 'Chromosome':
        """Generate a random chromosome within defined bounds."""
        rng = random.Random(seed)
        return cls(
            hidden_1=rng.randint(32, 256),
            hidden_2=rng.randint(16, 128),
            dropout=rng.uniform(0.05, 0.5),
            learning_rate=rng.uniform(0.0001, 0.01),
            weight_decay=rng.uniform(0.00001, 0.001),
            huber_delta=rng.uniform(0.5, 2.0),
            loss_weight_severity=rng.uniform(0.1, 1.0),
            loss_weight_resolution=rng.uniform(0.1, 1.0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            'hidden_1': self.hidden_1,
            'hidden_2': self.hidden_2,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'huber_delta': self.huber_delta,
            'loss_weight_severity': self.loss_weight_severity,
            'loss_weight_resolution': self.loss_weight_resolution,
            'feasible': self.feasible,
            'generation': self.generation
        }
        if self.fitness:
            d['fitness'] = {
                'id_macro_f1': self.fitness.id_macro_f1,
                'ood_macro_f1': self.fitness.ood_macro_f1,
                'ece': self.fitness.ece,
                'group_gap': self.fitness.group_gap,
                'latency_ms': self.fitness.latency_ms,
                'param_count': self.fitness.param_count,
                'seed_stability': self.fitness.seed_stability,
            }
        else:
            d['fitness'] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Chromosome':
        """Create from dictionary."""
        c = cls(
            hidden_1=d['hidden_1'],
            hidden_2=d['hidden_2'],
            dropout=d['dropout'],
            learning_rate=d['learning_rate'],
            weight_decay=d['weight_decay'],
            huber_delta=d['huber_delta'],
            loss_weight_severity=d['loss_weight_severity'],
            loss_weight_resolution=d['loss_weight_resolution'],
            feasible=d.get('feasible', True),
            generation=d.get('generation', 0)
        )
        if d.get('fitness'):
            c.fitness = FitnessVector(**d['fitness'])
        return c

    def encode(self) -> list[float]:
        """Encode to float vector."""
        return [
            float(self.hidden_1), float(self.hidden_2), self.dropout,
            self.learning_rate, self.weight_decay, self.huber_delta,
            self.loss_weight_severity, self.loss_weight_resolution
        ]

    def decode(self, vec: list[float]) -> None:
        """Decode from float vector."""
        self.hidden_1 = int(vec[0])
        self.hidden_2 = int(vec[1])
        self.dropout = vec[2]
        self.learning_rate = vec[3]
        self.weight_decay = vec[4]
        self.huber_delta = vec[5]
        self.loss_weight_severity = vec[6]
        self.loss_weight_resolution = vec[7]
