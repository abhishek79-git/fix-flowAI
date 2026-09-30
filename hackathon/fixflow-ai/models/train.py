"""FixFlow AI Training Pipeline.

Multi-task training with robust Huber loss, label smoothing,
gradient clipping, early stopping, and deterministic seeding.
"""

import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from models.features import create_feature_pipeline, transform_features, encode_categories
from models.multitask_model import FixFlowMLP, MultiTaskLoss, count_parameters, measure_latency


def set_deterministic(seed: int = 42) -> None:
    """Set all random seeds and enable deterministic mode.

    Args:
        seed: The master seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def create_dataloaders(
    X: np.ndarray,
    y_cat: np.ndarray,
    y_sev: np.ndarray,
    y_res: np.ndarray,
    batch_size: int = 64,
    seed: int = 42,
    shuffle: bool = True
) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y_cat: Integer category labels.
        y_sev: Float severity values [0, 1].
        y_res: Float normalized resolution values [0, 1].
        batch_size: Training batch size.
        seed: Seed for DataLoader shuffle.
        shuffle: Whether to shuffle data.

    Returns:
        PyTorch DataLoader.
    """
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_cat = torch.tensor(y_cat, dtype=torch.long)
    tensor_sev = torch.tensor(y_sev, dtype=torch.float32)
    tensor_res = torch.tensor(y_res, dtype=torch.float32)

    dataset = TensorDataset(tensor_x, tensor_cat, tensor_sev, tensor_res)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any]
) -> dict[str, list]:
    """Train the multi-task model with robust loss and regularization.

    Implements gradient clipping, early stopping, and records full history.

    Args:
        model: FixFlowMLP model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Dict with epochs, learning_rate, weight_decay, etc.

    Returns:
        Training history with per-epoch train_loss, val_loss, macro_f1, etc.
    """
    model_cfg = config.get('model', config)
    epochs = model_cfg.get('max_epochs', model_cfg.get('epochs', 50))
    lr = model_cfg.get('learning_rate', 1e-3)
    weight_decay = model_cfg.get('weight_decay', 1e-4)
    max_norm = model_cfg.get('gradient_clip_norm', 5.0)
    patience = model_cfg.get('patience', 10)
    label_smoothing = model_cfg.get('label_smoothing', 0.1)
    huber_delta = model_cfg.get('huber_delta', 1.0)

    loss_weights = model_cfg.get('loss_weights', {})
    cat_w = loss_weights.get('category', 1.0)
    sev_w = loss_weights.get('severity', 0.5)
    res_w = loss_weights.get('resolution', 0.3)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MultiTaskLoss(
        cat_weight=cat_w, sev_weight=sev_w, res_weight=res_w,
        label_smoothing=label_smoothing, delta=huber_delta
    )

    history: dict[str, list] = {
        'train_loss': [], 'val_loss': [], 'macro_f1': [],
        'mae_severity': [], 'mae_resolution': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # === Training ===
        model.train()
        train_loss_total = 0.0
        n_train = 0

        for batch in train_loader:
            x, y_cat, y_sev, y_res = batch
            optimizer.zero_grad()

            cat_out, sev_out, res_out = model(x)
            loss = criterion(cat_out, y_cat, sev_out, y_sev, res_out, y_res)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            train_loss_total += loss.item() * x.size(0)
            n_train += x.size(0)

        train_loss_avg = train_loss_total / max(n_train, 1)
        history['train_loss'].append(train_loss_avg)

        # === Validation ===
        model.eval()
        val_loss_total = 0.0
        n_val = 0
        all_cat_preds: list[int] = []
        all_cat_targets: list[int] = []
        all_sev_preds: list[float] = []
        all_sev_targets: list[float] = []
        all_res_preds: list[float] = []
        all_res_targets: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                x, y_cat, y_sev, y_res = batch
                cat_out, sev_out, res_out = model(x)
                loss = criterion(cat_out, y_cat, sev_out, y_sev, res_out, y_res)

                val_loss_total += loss.item() * x.size(0)
                n_val += x.size(0)

                all_cat_preds.extend(torch.argmax(cat_out, dim=1).cpu().numpy().tolist())
                all_cat_targets.extend(y_cat.cpu().numpy().tolist())
                all_sev_preds.extend(sev_out.squeeze().cpu().numpy().tolist())
                all_sev_targets.extend(y_sev.cpu().numpy().tolist())
                all_res_preds.extend(res_out.squeeze().cpu().numpy().tolist())
                all_res_targets.extend(y_res.cpu().numpy().tolist())

        val_loss_avg = val_loss_total / max(n_val, 1)
        macro_f1 = float(f1_score(all_cat_targets, all_cat_preds, average='macro', zero_division=0))
        mae_sev = float(mean_absolute_error(all_sev_targets, all_sev_preds))
        mae_res = float(mean_absolute_error(all_res_targets, all_res_preds))

        history['val_loss'].append(val_loss_avg)
        history['macro_f1'].append(macro_f1)
        history['mae_severity'].append(mae_sev)
        history['mae_resolution'].append(mae_res)

        # === Early Stopping ===
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def save_model(model: nn.Module, pipeline: dict[str, Any], path: str) -> None:
    """Save model weights and feature pipeline.

    Args:
        model: Trained PyTorch model.
        pipeline: Feature pipeline dict (contains vectorizer, encoders, etc.).
        path: File path to save to.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # Extract serializable pipeline info
    save_pipeline = {
        k: v for k, v in pipeline.items()
        if k not in ('vectorizer',)  # vectorizer saved separately
    }

    state = {
        'model_state': model.state_dict(),
        'pipeline': save_pipeline,
        'vectorizer': pipeline.get('vectorizer'),
        'input_dim': pipeline.get('total_feature_dim', 318),
    }
    torch.save(state, path)


def load_model(path: str) -> tuple[nn.Module, dict[str, Any]]:
    """Load model and feature pipeline from file.

    Args:
        path: Path to saved model file.

    Returns:
        Tuple of (model, pipeline_dict).

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    state = torch.load(path, weights_only=False)
    input_dim = state.get('input_dim', 318)
    num_categories = state.get('pipeline', {}).get('num_categories', 8)

    model = FixFlowMLP(input_dim=input_dim, num_categories=num_categories)
    model.load_state_dict(state['model_state'])
    model.eval()

    pipeline = state.get('pipeline', {})
    if 'vectorizer' in state:
        pipeline['vectorizer'] = state['vectorizer']

    return model, pipeline


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config_path: str = 'config.yaml',
    save_path: str = 'models/artifacts/best_model.pt'
) -> dict[str, Any]:
    """Complete baseline training pipeline from DataFrames to saved model.

    Args:
        train_df: Training DataFrame with complaint_text, category, severity, etc.
        val_df: Validation DataFrame.
        config_path: Path to config.yaml.
        save_path: Path to save the trained model.

    Returns:
        Dict with training history and final metrics.
    """
    set_deterministic(42)

    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}

    # Feature pipeline
    pipeline = create_feature_pipeline(train_df)
    X_train = transform_features(train_df, pipeline)
    X_val = transform_features(val_df, pipeline)

    # Target encoding
    y_cat_train = encode_categories(train_df['category'], pipeline)
    y_sev_train = train_df['severity'].values.astype(np.float32)
    y_res_train = (train_df['estimated_resolution_minutes'].values / 300.0).astype(np.float32)

    y_cat_val = encode_categories(val_df['category'], pipeline)
    y_sev_val = val_df['severity'].values.astype(np.float32)
    y_res_val = (val_df['estimated_resolution_minutes'].values / 300.0).astype(np.float32)

    batch_size = config.get('model', {}).get('batch_size', 64)

    train_loader = create_dataloaders(X_train, y_cat_train, y_sev_train, y_res_train, batch_size=batch_size)
    val_loader = create_dataloaders(X_val, y_cat_val, y_sev_val, y_res_val, batch_size=batch_size, shuffle=False)

    # Create model
    model = FixFlowMLP(
        input_dim=pipeline['total_feature_dim'],
        hidden_dims=config.get('model', {}).get('hidden_layers', [128, 64]),
        dropout=config.get('model', {}).get('dropout', 0.2),
        num_categories=pipeline['num_categories']
    )

    # Train
    history = train_model(model, train_loader, val_loader, config)

    # Save
    save_model(model, pipeline, save_path)

    # Final metrics
    param_count = count_parameters(model)
    sample = torch.randn(1, pipeline['total_feature_dim'])
    latency = measure_latency(model, sample, runs=50)

    return {
        'history': history,
        'final_macro_f1': history['macro_f1'][-1] if history['macro_f1'] else 0.0,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
        'param_count': param_count,
        'latency_ms': round(latency, 2),
        'epochs_trained': len(history['train_loss']),
        'model_path': save_path,
    }
