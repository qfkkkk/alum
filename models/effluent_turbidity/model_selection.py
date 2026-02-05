"""
Model Selection for Effluent Turbidity Prediction Model
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from .config import Config


class ModelSelector:
    """Model training, evaluation and selection."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize ModelSelector.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or Config()
        self.scaler: Optional[StandardScaler] = None
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.results: List[Dict[str, Any]] = []
    
    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler on training data and transform.
        
        Args:
            X: Training features
            
        Returns:
            Scaled features
        """
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            Scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(X)
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Features
            y: True targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X)
        
        # Overall metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Per-step R² scores
        per_step_r2 = {}
        for i in range(y.shape[1]):
            step_r2 = r2_score(y[:, i], y_pred[:, i])
            per_step_r2[f"r2_t+{i+1}"] = step_r2
        
        return {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            **per_step_r2
        }
    
    def train_and_evaluate(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> pd.DataFrame:
        """
        Train all candidate models and evaluate on validation set.
        
        Args:
            models: Dictionary of model name -> model object
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            DataFrame with evaluation results for all models
        """
        self.results = []
        
        print("\n" + "=" * 60)
        print("Training and Evaluating Models")
        print("=" * 60)
        
        for name, model in models.items():
            print(f"\n[{name}] Training...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_metrics = self.evaluate_model(model, X_val, y_val)
                
                result = {
                    "model": name,
                    "val_r2": val_metrics["r2"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                }
                
                # Add per-step R² scores
                for key, value in val_metrics.items():
                    if key.startswith("r2_t+"):
                        result[f"val_{key}"] = value
                
                self.results.append(result)
                
                print(f"[{name}] Val R²: {val_metrics['r2']:.4f}, "
                      f"MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
                
            except Exception as e:
                print(f"[{name}] Training failed: {e}")
                continue
        
        # Create results DataFrame and sort by R²
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values("val_r2", ascending=False).reset_index(drop=True)
        
        return results_df
    
    def select_best_model(
        self,
        models: Dict[str, Any],
        results_df: pd.DataFrame
    ) -> Tuple[str, Any]:
        """
        Select the best model based on validation R².
        
        Args:
            models: Dictionary of model name -> model object
            results_df: Results DataFrame from train_and_evaluate
            
        Returns:
            Tuple of (best_model_name, best_model)
        """
        best_model_name = results_df.iloc[0]["model"]
        best_model = models[best_model_name]
        
        self.best_model_name = best_model_name
        self.best_model = best_model
        
        print(f"\n{'=' * 60}")
        print(f"Best Model: {best_model_name}")
        print(f"Validation R²: {results_df.iloc[0]['val_r2']:.4f}")
        print(f"{'=' * 60}")
        
        return best_model_name, best_model
    
    def retrain_on_full_data(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Any:
        """
        Retrain the best model on train + validation data.
        
        Args:
            model: Model to retrain
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Retrained model
        """
        print("\nRetraining best model on Train + Val data...")
        
        # Combine train and validation data
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.vstack([y_train, y_val])
        
        # Clone and retrain the model
        from sklearn.base import clone
        model_retrained = clone(model)
        model_retrained.fit(X_combined, y_combined)
        
        self.best_model = model_retrained
        
        print(f"Retrained on {len(X_combined):,} samples")
        
        return model_retrained
    
    def final_evaluation(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Final evaluation on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with test metrics
        """
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)
        
        test_metrics = self.evaluate_model(model, X_test, y_test)
        
        print(f"\nTest R²:   {test_metrics['r2']:.4f}")
        print(f"Test MAE:  {test_metrics['mae']:.4f}")
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        
        print("\nPer-step R² scores:")
        for i in range(self.config.horizon):
            key = f"r2_t+{i+1}"
            print(f"  t+{i+1}: {test_metrics[key]:.4f}")
        
        return test_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X: Features (will be scaled if scaler is fitted)
            
        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("No model trained. Run train_and_evaluate first.")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.best_model.predict(X)
