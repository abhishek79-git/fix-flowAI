import torch
import torch.nn as nn
import time
from typing import List, Tuple

class FixFlowMLP(nn.Module):
    """Multi-task MLP with shared backbone and task-specific heads."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2, num_categories: int = 8):
        """
        Initialize the multi-task MLP.
        
        Args:
            input_dim: Dimension of input features.
            hidden_dims: List of dimensions for hidden layers.
            dropout: Dropout probability.
            num_categories: Number of output categories for classification.
        """
        super(FixFlowMLP, self).__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Category head (classification)
        self.category_head = nn.Linear(hidden_dims[1], num_categories)
        
        # Severity head (regression)
        self.severity_head = nn.Linear(hidden_dims[1], 1)
        
        # Resolution head (regression)
        self.resolution_head = nn.Linear(hidden_dims[1], 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the model."""
        shared_features = self.backbone(x)
        
        category_out = self.category_head(shared_features)
        severity_out = self.severity_head(shared_features)
        resolution_out = self.resolution_head(shared_features)
        
        return category_out, severity_out, resolution_out


class HuberLossSmooth(nn.Module):
    """Huber loss with configurable delta."""
    
    def __init__(self, delta: float = 1.0):
        """
        Args:
            delta: The threshold at which to change between L1 and L2 loss.
        """
        super(HuberLossSmooth, self).__init__()
        self.delta = delta
        self.loss_fn = nn.HuberLoss(delta=self.delta)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Huber loss."""
        return self.loss_fn(pred, target)


class MultiTaskLoss(nn.Module):
    """Combines cross-entropy (with label smoothing) + Huber losses with configurable weights."""
    
    def __init__(self, cat_weight: float = 1.0, sev_weight: float = 1.0, res_weight: float = 1.0, label_smoothing: float = 0.1, delta: float = 1.0):
        """
        Initialize the multi-task loss.
        
        Args:
            cat_weight: Weight for category classification loss.
            sev_weight: Weight for severity regression loss.
            res_weight: Weight for resolution regression loss.
            label_smoothing: Smoothing parameter for CrossEntropyLoss.
            delta: Delta for HuberLoss.
        """
        super(MultiTaskLoss, self).__init__()
        self.cat_weight = cat_weight
        self.sev_weight = sev_weight
        self.res_weight = res_weight
        
        self.cat_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.sev_loss_fn = HuberLossSmooth(delta=delta)
        self.res_loss_fn = HuberLossSmooth(delta=delta)
        
    def forward(self, 
                cat_pred: torch.Tensor, cat_target: torch.Tensor,
                sev_pred: torch.Tensor, sev_target: torch.Tensor,
                res_pred: torch.Tensor, res_target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss."""
        cat_loss = self.cat_loss_fn(cat_pred, cat_target)
        sev_loss = self.sev_loss_fn(sev_pred.squeeze(), sev_target.squeeze())
        res_loss = self.res_loss_fn(res_pred.squeeze(), res_target.squeeze())
        
        total_loss = (self.cat_weight * cat_loss) + (self.sev_weight * sev_loss) + (self.res_weight * res_loss)
        return total_loss


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch module.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model: nn.Module, sample_input: torch.Tensor, runs: int = 100) -> float:
    """
    Measure average inference latency in milliseconds.
    
    Args:
        model: PyTorch model.
        sample_input: Input tensor.
        runs: Number of runs to average.
        
    Returns:
        Average latency in milliseconds.
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(sample_input)
            
    # Measure
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            model(sample_input)
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / runs) * 1000
    return float(avg_time_ms)
