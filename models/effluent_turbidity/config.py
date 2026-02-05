"""
Configuration for Effluent Turbidity Prediction Model
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


@dataclass
class Config:
    """Configuration class for the effluent turbidity prediction model."""
    
    # Target column
    target_col: str = "turb_chushui_1"
    
    # Feature columns (excluding target and event flags)
    feature_cols: List[str] = field(default_factory=lambda: [
        "dose_1",
        "turb_chushui_1",  # Also used as feature (lagged)
        "turb_jinshui_1",
        "flow_1",
        "pH",
        "temp_down",
        "temp_shuimian",
    ])
    
    # Prediction horizon (number of future steps to predict)
    horizon: int = 6
    
    # Lag window size for feature engineering
    lag_window: int = 12
    
    # Rolling window sizes for statistical features
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])
    
    # Data split ratios (must sum to 1.0)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Output directory
    output_dir: str = "output/effluent_turbidity"
    
    def get_candidate_models(self) -> Dict[str, Any]:
        """
        Returns a dictionary of candidate models.
        Includes XGBoost and LightGBM by default (if installed).
        """
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            "Ridge": MultiOutputRegressor(
                Ridge(alpha=1.0, random_state=self.random_seed)
            ),
        }
        
        # Add XGBoost if installed
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_seed,
                    n_jobs=-1
                )
            )
        except ImportError:
            print("Warning: XGBoost not installed, skipping...")
        
        # Add LightGBM if installed
        try:
            from lightgbm import LGBMRegressor
            models["LightGBM"] = MultiOutputRegressor(
                LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_seed,
                    n_jobs=-1,
                    verbose=-1
                )
            )
        except ImportError:
            print("Warning: LightGBM not installed, skipping...")
        
        return models
