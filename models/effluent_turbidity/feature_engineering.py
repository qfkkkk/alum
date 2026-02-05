"""
Feature Engineering for Effluent Turbidity Prediction Model
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from .config import Config


class FeatureEngineer:
    """Feature engineering for time series prediction."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or Config()
        self.feature_names: List[str] = []
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            df: Input DataFrame
            cols: Columns to create lags for. Uses config.feature_cols if not provided.
            lags: List of lag values. Uses range(1, config.lag_window+1) if not provided.
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        cols = cols or self.config.feature_cols
        lags = lags or list(range(1, self.config.lag_window + 1))
        
        for col in cols:
            if col not in df.columns:
                continue
            for lag in lags:
                lag_col_name = f"{col}_lag{lag}"
                df[lag_col_name] = df[col].shift(lag)
                
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create rolling statistical features (mean, std) for specified columns.
        
        Args:
            df: Input DataFrame
            cols: Columns to create rolling features for
            windows: Window sizes for rolling calculations
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        cols = cols or self.config.feature_cols
        windows = windows or self.config.rolling_windows
        
        for col in cols:
            if col not in df.columns:
                continue
            for window in windows:
                # Rolling mean
                mean_col_name = f"{col}_roll{window}_mean"
                df[mean_col_name] = df[col].shift(1).rolling(window=window).mean()
                
                # Rolling std
                std_col_name = f"{col}_roll{window}_std"
                df[std_col_name] = df[col].shift(1).rolling(window=window).std()
                
        return df
    
    def create_diff_features(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create difference features (first-order diff) for trend capture.
        
        Args:
            df: Input DataFrame
            cols: Columns to create diff features for
            
        Returns:
            DataFrame with diff features added
        """
        df = df.copy()
        cols = cols or self.config.feature_cols
        
        for col in cols:
            if col not in df.columns:
                continue
            diff_col_name = f"{col}_diff1"
            df[diff_col_name] = df[col].diff(1)
            
        return df
    
    def create_target_columns(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        horizon: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create multi-step target columns (t+1, t+2, ..., t+horizon).
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            horizon: Number of future steps to predict
            
        Returns:
            DataFrame with target columns added
        """
        df = df.copy()
        target_col = target_col or self.config.target_col
        horizon = horizon or self.config.horizon
        
        for h in range(1, horizon + 1):
            target_name = f"target_t+{h}"
            df[target_name] = df[target_col].shift(-h)
            
        return df
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        include_rolling: bool = True,
        include_diff: bool = False
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Full feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            include_rolling: Whether to include rolling features
            include_diff: Whether to include diff features
            
        Returns:
            Tuple of (processed_df, feature_names, target_names)
        """
        # Step 1: Create lag features
        df = self.create_lag_features(df)
        
        # Step 2: Create rolling features (optional)
        if include_rolling:
            df = self.create_rolling_features(df)
        
        # Step 3: Create diff features (optional)
        if include_diff:
            df = self.create_diff_features(df)
        
        # Step 4: Create target columns
        df = self.create_target_columns(df)
        
        # Step 5: Drop rows with NaN values (from lagging/rolling)
        df = df.dropna()
        
        # Step 6: Identify feature and target columns
        target_names = [f"target_t+{h}" for h in range(1, self.config.horizon + 1)]
        
        # Feature columns: all columns except targets and original feature columns
        exclude_cols = set(target_names) | set(self.config.feature_cols) | {"low_flow", "high_turb"}
        feature_names = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_names = feature_names
        
        print(f"Feature engineering complete:")
        print(f"  Total samples: {len(df):,}")
        print(f"  Number of features: {len(feature_names)}")
        print(f"  Target columns: {target_names}")
        
        return df, feature_names, target_names
    
    def get_X_y(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract X (features) and y (targets) arrays from DataFrame.
        
        Args:
            df: Processed DataFrame
            feature_names: List of feature column names
            target_names: List of target column names
            
        Returns:
            Tuple of (X, y) as numpy arrays
        """
        X = df[feature_names].values
        y = df[target_names].values
        
        return X, y
