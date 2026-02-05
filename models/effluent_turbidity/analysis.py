"""
Model Analysis and Visualization for Effluent Turbidity Prediction
图表使用中文标签
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from .config import Config


# Set matplotlib style with Chinese font
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
# Chinese font for macOS
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# Variable name mapping to Chinese
VARIABLE_NAMES_CN = {
    'dose_1': '投药量',
    'turb_chushui_1': '出水浊度',
    'turb_jinshui_1': '进水浊度',
    'flow_1': '流量',
    'pH': 'pH值',
    'temp_down': '水下温度',
    'temp_shuimian': '水面温度',
}


def get_cn_name(var_name: str) -> str:
    """Get Chinese name for a variable."""
    # Check if it's a base variable
    for key, cn_name in VARIABLE_NAMES_CN.items():
        if var_name == key:
            return cn_name
        if var_name.startswith(f"{key}_lag"):
            lag = var_name.split('_lag')[1]
            return f"{cn_name}_滞后{lag}"
        if var_name.startswith(f"{key}_roll"):
            parts = var_name.split('_roll')[1]
            if '_mean' in parts:
                window = parts.split('_mean')[0]
                return f"{cn_name}_滚动{window}均值"
            elif '_std' in parts:
                window = parts.split('_std')[0]
                return f"{cn_name}_滚动{window}标准差"
    return var_name


class ModelAnalyzer:
    """Model analysis and visualization tools."""
    
    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_names: List[str],
        config: Optional[Config] = None
    ):
        """
        Initialize ModelAnalyzer.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            feature_names: List of feature names
            config: Configuration object
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.config = config or Config()
        self.output_dir = os.path.join(self.config.output_dir, "figures")
        os.makedirs(self.output_dir, exist_ok=True)
    
    # ==================== Model Performance ====================
    
    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        index: pd.DatetimeIndex,
        step: int = 1,
        n_samples: int = 500,
        save_path: Optional[str] = None
    ) -> None:
        """Plot prediction vs actual values for a specific step."""
        fig, ax = plt.subplots(figsize=(14, 5))
        
        step_idx = step - 1
        y_true_step = y_true[-n_samples:, step_idx]
        y_pred_step = y_pred[-n_samples:, step_idx]
        time_idx = index[-n_samples:]
        
        ax.plot(time_idx, y_true_step, label='真实值', alpha=0.8, linewidth=1)
        ax.plot(time_idx, y_pred_step, label='预测值', alpha=0.8, linewidth=1)
        
        ax.set_xlabel('时间')
        ax.set_ylabel('浊度')
        ax.set_title(f'预测值 vs 真实值 (t+{step}步)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    def plot_r2_by_step(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> None:
        """Plot R² score for each prediction step."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        steps = list(range(1, self.config.horizon + 1))
        r2_values = [metrics[f"r2_t+{i}"] for i in steps]
        
        bars = ax.bar(steps, r2_values, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, r2_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('预测步数')
        ax.set_ylabel('R² 分数')
        ax.set_title('R² 分数随预测步数的衰减')
        ax.set_xticks(steps)
        ax.set_xticklabels([f't+{i}' for i in steps])
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    def plot_residual_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot residual distribution for all steps."""
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        
        for i in range(self.config.horizon):
            residuals = y_true[:, i] - y_pred[:, i]
            
            ax = axes[i]
            ax.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax.axvline(x=residuals.mean(), color='orange', linestyle='-', linewidth=1,
                      label=f'均值: {residuals.mean():.4f}')
            
            ax.set_xlabel('残差')
            ax.set_ylabel('频数')
            ax.set_title(f't+{i+1}步 残差分布')
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    # ==================== Feature Importance ====================
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from model."""
        # Handle different model types
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            importances_list = []
            for est in self.model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances_list.append(est.feature_importances_)
            if importances_list:
                importances = np.mean(importances_list, axis=0)
            else:
                if hasattr(self.model.estimators_[0], 'coef_'):
                    coefs = np.array([est.coef_ for est in self.model.estimators_])
                    importances = np.mean(np.abs(coefs), axis=0)
                else:
                    raise ValueError("Model does not support feature importance extraction")
        else:
            raise ValueError("Model does not support feature importance extraction")
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'feature_cn': [get_cn_name(f) for f in self.feature_names],
            'importance': importances
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Plot top N feature importances."""
        importance_df = self.get_feature_importance()
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature_cn'])
        ax.invert_yaxis()
        ax.set_xlabel('重要性')
        ax.set_title(f'特征重要性排名 (Top {top_n})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
        return importance_df
    
    def plot_importance_by_variable(
        self,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Plot feature importance grouped by original variable."""
        importance_df = self.get_feature_importance()
        
        def get_variable_name(feature_name: str) -> str:
            for suffix in ['_lag', '_roll', '_diff']:
                if suffix in feature_name:
                    return feature_name.split(suffix)[0]
            return feature_name
        
        importance_df['variable'] = importance_df['feature'].apply(get_variable_name)
        grouped = importance_df.groupby('variable')['importance'].sum().sort_values(ascending=False)
        
        # Convert to Chinese names
        grouped.index = [VARIABLE_NAMES_CN.get(v, v) for v in grouped.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        grouped.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
        ax.set_xlabel('原始变量')
        ax.set_ylabel('总重要性')
        ax.set_title('按原始变量分组的特征重要性')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
        return grouped.reset_index()
    
    def plot_lag_importance(
        self,
        variable: str,
        save_path: Optional[str] = None
    ) -> None:
        """Plot importance of different lags for a specific variable."""
        importance_df = self.get_feature_importance()
        
        lag_features = importance_df[
            importance_df['feature'].str.startswith(f"{variable}_lag")
        ].copy()
        
        if lag_features.empty:
            print(f"No lag features found for {variable}")
            return
        
        lag_features['lag'] = lag_features['feature'].apply(
            lambda x: int(x.split('_lag')[1])
        )
        lag_features = lag_features.sort_values('lag')
        
        var_cn = VARIABLE_NAMES_CN.get(variable, variable)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.bar(lag_features['lag'], lag_features['importance'], 
               color='steelblue', alpha=0.8)
        ax.set_xlabel('滞后步数')
        ax.set_ylabel('重要性')
        ax.set_title(f'{var_cn} 滞后特征重要性分析')
        ax.set_xticks(lag_features['lag'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    # ==================== Sensitivity Analysis ====================
    
    def sensitivity_analysis(
        self,
        X_base: np.ndarray,
        feature_idx: int,
        values: np.ndarray
    ) -> np.ndarray:
        """Perform sensitivity analysis on a single feature."""
        predictions = []
        
        for val in values:
            X_test = X_base.copy()
            X_test[0, feature_idx] = val
            X_scaled = self.scaler.transform(X_test)
            pred = self.model.predict(X_scaled)
            predictions.append(pred[0])
        
        return np.array(predictions)
    
    def plot_sensitivity_curve(
        self,
        X_base: np.ndarray,
        feature_name: str,
        n_points: int = 50,
        save_path: Optional[str] = None
    ) -> None:
        """Plot sensitivity curve for a feature."""
        if feature_name not in self.feature_names:
            print(f"Feature {feature_name} not found")
            return
        
        feature_idx = self.feature_names.index(feature_name)
        
        base_val = X_base[0, feature_idx]
        feature_std = self.scaler.scale_[feature_idx] if hasattr(self.scaler, 'scale_') else 1.0
        values = np.linspace(base_val - 2*feature_std, base_val + 2*feature_std, n_points)
        
        predictions = self.sensitivity_analysis(X_base, feature_idx, values)
        
        feature_cn = get_cn_name(feature_name)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for step in range(self.config.horizon):
            ax.plot(values, predictions[:, step], label=f't+{step+1}步', alpha=0.8)
        
        ax.axvline(x=base_val, color='red', linestyle='--', alpha=0.5, label='基准值')
        ax.set_xlabel(feature_cn)
        ax.set_ylabel('预测浊度')
        ax.set_title(f'敏感性分析: {feature_cn}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    # ==================== Run All Analysis ====================
    
    def run_all_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X_test: np.ndarray,
        index: pd.DatetimeIndex,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Run all analysis and save figures. Returns metrics dict for report."""
        print("\n" + "=" * 60)
        print("Running Model Analysis")
        print("=" * 60)
        
        # 1. Prediction vs Actual
        print("\n[1/5] Generating prediction vs actual plots...")
        for step in [1, 3, 6]:
            self.plot_prediction_vs_actual(
                y_true, y_pred, index, step=step,
                save_path=os.path.join(self.output_dir, f"prediction_vs_actual_t{step}.png")
            )
        
        # 2. R² by step
        print("[2/5] Generating R² decay plot...")
        self.plot_r2_by_step(
            metrics,
            save_path=os.path.join(self.output_dir, "r2_by_step.png")
        )
        
        # 3. Residual distribution
        print("[3/5] Generating residual distribution plot...")
        self.plot_residual_distribution(
            y_true, y_pred,
            save_path=os.path.join(self.output_dir, "residual_distribution.png")
        )
        
        # 4. Feature importance
        print("[4/5] Generating feature importance plots...")
        importance_df = self.plot_feature_importance(
            top_n=20,
            save_path=os.path.join(self.output_dir, "feature_importance_top20.png")
        )
        importance_df.to_csv(
            os.path.join(self.config.output_dir, "feature_importance.csv"),
            index=False
        )
        
        self.plot_importance_by_variable(
            save_path=os.path.join(self.output_dir, "importance_by_variable.png")
        )
        
        # Lag importance for key variables
        for var in ['turb_jinshui_1', 'dose_1', 'turb_chushui_1']:
            self.plot_lag_importance(
                var,
                save_path=os.path.join(self.output_dir, f"lag_importance_{var}.png")
            )
        
        # 5. Sensitivity analysis
        print("[5/5] Generating sensitivity analysis plots...")
        X_base = X_test[len(X_test)//2:len(X_test)//2+1].copy()
        
        for var in ['dose_1_lag1', 'turb_jinshui_1_lag1', 'flow_1_lag1']:
            if var in self.feature_names:
                self.plot_sensitivity_curve(
                    X_base, var,
                    save_path=os.path.join(self.output_dir, f"sensitivity_{var}.png")
                )
        
        print(f"\n✓ All figures saved to: {self.output_dir}/")
        
        return {
            'metrics': metrics,
            'importance_df': importance_df,
            'figures_dir': self.output_dir
        }
