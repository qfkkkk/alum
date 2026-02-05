#!/usr/bin/env python3
"""
Training script for Effluent Turbidity Prediction Model

Usage:
    python scripts/train_effluent_turbidity.py
    
    # With custom data path
    python scripts/train_effluent_turbidity.py --data data/processed_data.csv
    
    # Include boosting models (XGBoost, LightGBM)
    python scripts/train_effluent_turbidity.py --use-boosting
"""
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.effluent_turbidity import (
    Config,
    FeatureEngineer,
    ModelSelector,
    load_data,
    time_series_split,
    save_model,
    save_results,
)
from models.effluent_turbidity.utils import save_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Effluent Turbidity Prediction Model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed_data.csv",
        help="Path to input data CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/effluent_turbidity",
        help="Output directory"
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip retraining on train+val after model selection"
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("Effluent Turbidity Prediction Model - Training")
    print("=" * 60)
    
    # Initialize configuration
    config = Config(output_dir=args.output)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    df = load_data(args.data)
    print(f"Loaded {len(df):,} samples from {args.data}")
    
    # Step 2: Feature engineering
    print("\n[Step 2] Feature engineering...")
    fe = FeatureEngineer(config)
    df_processed, feature_names, target_names = fe.prepare_dataset(df)
    
    # Step 3: Time series split
    print("\n[Step 3] Splitting data (60/20/20)...")
    train_df, val_df, test_df = time_series_split(
        df_processed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    # Extract X and y
    X_train, y_train = fe.get_X_y(train_df, feature_names, target_names)
    X_val, y_val = fe.get_X_y(val_df, feature_names, target_names)
    X_test, y_test = fe.get_X_y(test_df, feature_names, target_names)
    
    print(f"\nFeature matrix shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    # Step 4: Initialize model selector and scale features
    print("\n[Step 4] Scaling features...")
    selector = ModelSelector(config)
    X_train_scaled = selector.fit_scaler(X_train)
    X_val_scaled = selector.transform(X_val)
    X_test_scaled = selector.transform(X_test)
    
    # Step 5: Get candidate models (includes XGBoost/LightGBM if installed)
    print("\n[Step 5] Preparing candidate models...")
    models = config.get_candidate_models()
    
    print(f"Candidate models: {list(models.keys())}")
    
    # Step 6: Train and evaluate all models
    print("\n[Step 6] Training and evaluating models...")
    results_df = selector.train_and_evaluate(
        models,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val
    )
    
    # Step 7: Select best model
    print("\n[Step 7] Selecting best model...")
    best_name, best_model = selector.select_best_model(models, results_df)
    
    # Step 8: Retrain on train + val (optional)
    if not args.no_retrain:
        print("\n[Step 8] Retraining best model on Train + Val...")
        best_model = selector.retrain_on_full_data(
            best_model,
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val
        )
    
    # Step 9: Final evaluation on test set
    print("\n[Step 9] Final evaluation on test set...")
    test_metrics = selector.final_evaluation(best_model, X_test_scaled, y_test)
    
    # Add test metrics to results
    results_df.loc[results_df["model"] == best_name, "test_r2"] = test_metrics["r2"]
    results_df.loc[results_df["model"] == best_name, "test_mae"] = test_metrics["mae"]
    results_df.loc[results_df["model"] == best_name, "test_rmse"] = test_metrics["rmse"]
    
    # Step 10: Save outputs
    print("\n[Step 10] Saving outputs...")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "models"), exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output, "models", "best_model.pkl")
    save_model(best_model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(args.output, "models", "scaler.pkl")
    save_model(selector.scaler, scaler_path)
    
    # Save model comparison results
    results_path = os.path.join(args.output, "model_comparison.csv")
    save_results(results_df, results_path)
    
    # Save test predictions
    y_pred = best_model.predict(X_test_scaled)
    predictions_path = os.path.join(args.output, "test_predictions.csv")
    save_predictions(
        y_test,
        y_pred,
        test_df.index,
        config.horizon,
        predictions_path
    )
    
    # Save feature names for inference
    import json
    feature_info = {
        "feature_names": feature_names,
        "target_names": target_names,
        "best_model": best_name,
        "test_r2": test_metrics["r2"]
    }
    feature_info_path = os.path.join(args.output, "feature_info.json")
    with open(feature_info_path, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"Feature info saved to: {feature_info_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest Model: {best_name}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"\nOutputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
