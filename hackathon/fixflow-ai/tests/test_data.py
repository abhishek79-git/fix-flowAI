"""Tests for data generation."""
import pytest

def test_dataset_generation_deterministic() -> None:
    """generate twice with seed=42, assert equal."""
    pass

def test_dataset_schema() -> None:
    """check all required columns present."""
    pass

def test_dataset_size() -> None:
    """check >= 5000 records."""
    pass

def test_category_distribution() -> None:
    """check all categories present."""
    pass

def test_correlations() -> None:
    """check meaningful correlations exist."""
    pass

def test_split_sizes() -> None:
    """check train/val/ood ratios."""
    pass

def test_ood_distribution_differs() -> None:
    """OOD should differ from training distribution."""
    pass
