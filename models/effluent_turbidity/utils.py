"""
Utility functions for Effluent Turbidity Prediction Model
"""
import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Any


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with DateTime as index
    """
    df = pd.read_csv(path)
    
    # Parse DateTime column and set as index
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df.set_index("DateTime", inplace=True)
    
    # Sort by index to ensure temporal order
    df.sort_index(inplace=True)
    
    return df


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.
    Data is split chronologically (no shuffling).
    
    Args:
        df: Input DataFrame (must be sorted by time)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Data split:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df


def save_model(model: Any, path: str) -> None:
    """
    Save model to disk using joblib.
    
    Args:
        model: Trained model object
        path: Path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path: str) -> Any:
    """
    Load model from disk.
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded model object
    """
    return joblib.load(path)


def save_results(results: pd.DataFrame, path: str) -> None:
    """
    Save results DataFrame to CSV.
    
    Args:
        results: Results DataFrame
        path: Path to save the CSV
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path, index=False)
    print(f"Results saved to: {path}")


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: pd.DatetimeIndex,
    horizon: int,
    path: str
) -> None:
    """
    Save predictions with true values to CSV.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        index: DateTime index
        horizon: Number of prediction steps
        path: Path to save the CSV
    """
    # Create column names for each prediction step
    true_cols = [f"true_t+{i+1}" for i in range(horizon)]
    pred_cols = [f"pred_t+{i+1}" for i in range(horizon)]
    
    # Create DataFrame
    df = pd.DataFrame(index=index)
    
    for i in range(horizon):
        df[true_cols[i]] = y_true[:, i]
        df[pred_cols[i]] = y_pred[:, i]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Predictions saved to: {path}")
