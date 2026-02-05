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
    """Get Chinese name for a variable.
    
    滞后特征说明：
    - 滞后N (t-N) 表示使用 N 个时间步之前的历史数据作为输入
    - 例如 "出水浊度(t-1)" 表示上一个时间点的出水浊度值作为当前预测的输入
    """
    for key, cn_name in VARIABLE_NAMES_CN.items():
        if var_name == key:
            return cn_name
        if var_name.startswith(f"{key}_lag"):
            lag = var_name.split('_lag')[1]
            return f"{cn_name}(t-{lag})"  # 改为 t-N 格式
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
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.config = config or Config()
        self.output_dir = os.path.join(self.config.output_dir, "figures")
        os.makedirs(self.output_dir, exist_ok=True)
    
    # ==================== Model Performance ====================
    
    def plot_prediction_examples(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        index: pd.DatetimeIndex,
        n_examples: int = 3,
        window_size: int = 50,
        save_path: Optional[str] = None
    ) -> None:
        """Plot specific prediction examples (short windows) instead of overall comparison.
        
        选择几个典型的时间窗口展示预测效果，而不是整体对比。
        """
        fig, axes = plt.subplots(n_examples, 2, figsize=(14, 4 * n_examples))
        
        # Select example positions: start, middle, end of test set
        n_total = len(y_true)
        positions = [
            int(n_total * 0.1),   # 开始
            int(n_total * 0.5),   # 中间
            int(n_total * 0.9) - window_size,  # 结束
        ]
        
        for row, pos in enumerate(positions[:n_examples]):
            end_pos = min(pos + window_size, n_total)
            
            # t+1 prediction
            ax1 = axes[row, 0]
            ax1.plot(range(window_size), y_true[pos:end_pos, 0], 
                    'b-', label='真实值', linewidth=1.5)
            ax1.plot(range(window_size), y_pred[pos:end_pos, 0], 
                    'r--', label='预测值', linewidth=1.5, alpha=0.8)
            start_time = index[pos].strftime('%m-%d %H:%M')
            ax1.set_title(f'示例{row+1}: t+1步预测 (起始: {start_time})')
            ax1.set_xlabel('时间步')
            ax1.set_ylabel('浊度')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # t+6 prediction
            ax2 = axes[row, 1]
            ax2.plot(range(window_size), y_true[pos:end_pos, 5], 
                    'b-', label='真实值', linewidth=1.5)
            ax2.plot(range(window_size), y_pred[pos:end_pos, 5], 
                    'r--', label='预测值', linewidth=1.5, alpha=0.8)
            ax2.set_title(f'示例{row+1}: t+6步预测 (起始: {start_time})')
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('浊度')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
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
        """Plot residual distribution for all steps with appropriate x-axis limits."""
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        
        for i in range(self.config.horizon):
            residuals = y_true[:, i] - y_pred[:, i]
            
            ax = axes[i]
            
            # 使用合理的x轴范围：基于数据的分位数
            q1, q99 = np.percentile(residuals, [1, 99])
            xlim = max(abs(q1), abs(q99)) * 1.2
            
            # 过滤掉极端值进行绑定
            filtered_residuals = residuals[(residuals >= -xlim) & (residuals <= xlim)]
            
            ax.hist(filtered_residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='零线')
            ax.axvline(x=residuals.mean(), color='orange', linestyle='-', linewidth=1.5,
                      label=f'均值: {residuals.mean():.4f}')
            
            ax.set_xlim(-xlim, xlim)
            ax.set_xlabel('残差 (真实值 - 预测值)')
            ax.set_ylabel('频数')
            ax.set_title(f't+{i+1}步 残差分布')
            ax.legend(fontsize=8)
        
        plt.suptitle('残差分布分析\n(残差 = 真实值 - 预测值，接近0表示预测准确)', fontsize=12, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    # ==================== Feature Importance ====================
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from model."""
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
        ax.set_title(f'特征重要性排名 (Top {top_n})\n\n说明: t-N 表示使用 N 个时间步之前的历史数据作为输入')
        
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
        """Plot importance of different lags for a specific variable.
        
        横坐标改为 t-N 格式，表示 N 个时间步之前的输入。
        """
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
        ax.set_xlabel('输入时间点 (t-N 表示 N 步之前的数据)')
        ax.set_ylabel('重要性')
        ax.set_title(f'{var_cn} 历史输入的重要性分析\n(越靠近当前时刻 t-1，表示越近期的数据)')
        ax.set_xticks(lag_features['lag'])
        ax.set_xticklabels([f't-{lag}' for lag in lag_features['lag']])
        
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
    ) -> Tuple[float, float]:
        """Plot sensitivity curve for a feature.
        
        敏感性分析方法说明：
        1. 固定其他所有输入特征不变
        2. 仅改变目标特征的值（在 ±2个标准差范围内变化）
        3. 观察模型预测输出如何随之变化
        
        返回: (基准值, 变化范围)
        """
        if feature_name not in self.feature_names:
            print(f"Feature {feature_name} not found")
            return (0, 0)
        
        feature_idx = self.feature_names.index(feature_name)
        
        base_val = X_base[0, feature_idx]
        feature_std = self.scaler.scale_[feature_idx] if hasattr(self.scaler, 'scale_') else 1.0
        
        # 变化范围：基准值 ± 2个标准差
        val_min = base_val - 2 * feature_std
        val_max = base_val + 2 * feature_std
        values = np.linspace(val_min, val_max, n_points)
        
        predictions = self.sensitivity_analysis(X_base, feature_idx, values)
        
        feature_cn = get_cn_name(feature_name)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for step in range(self.config.horizon):
            ax.plot(values, predictions[:, step], label=f't+{step+1}步', alpha=0.8)
        
        ax.axvline(x=base_val, color='red', linestyle='--', alpha=0.7, 
                  linewidth=2, label=f'基准值: {base_val:.3f}')
        ax.set_xlabel(f'{feature_cn} 值')
        ax.set_ylabel('预测浊度')
        ax.set_title(f'敏感性分析: {feature_cn}\n'
                    f'(固定其他输入，仅改变此特征，观察预测变化)')
        ax.legend()
        
        # 添加说明文本
        ax.text(0.02, 0.98, 
                f'变化范围: {val_min:.3f} ~ {val_max:.3f}\n(基准值 ± 2倍标准差)',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
        return (base_val, 2 * feature_std)
    
    # ==================== Run All Analysis ====================
    
    def run_all_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X_test: np.ndarray,
        index: pd.DatetimeIndex,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Run all analysis and save figures."""
        print("\n" + "=" * 60)
        print("Running Model Analysis")
        print("=" * 60)
        
        # 1. Prediction examples (改为具体示例)
        print("\n[1/5] Generating prediction example plots...")
        self.plot_prediction_examples(
            y_true, y_pred, index,
            save_path=os.path.join(self.output_dir, "prediction_examples.png")
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
        
        # 5. Sensitivity analysis (分析所有滞后 t-12 到 t-1)
        print("[5/5] Generating sensitivity analysis plots...")
        X_base = X_test[len(X_test)//2:len(X_test)//2+1].copy()
        
        sensitivity_info = {}
        # 对每个变量分析所有滞后
        for var_base in ['dose_1', 'turb_jinshui_1', 'flow_1']:
            self.plot_sensitivity_all_lags(
                X_base, var_base,
                save_path=os.path.join(self.output_dir, f"sensitivity_{var_base}_all_lags.png")
            )
        
        print(f"\n✓ All figures saved to: {self.output_dir}/")
        
        return {
            'metrics': metrics,
            'importance_df': importance_df,
            'figures_dir': self.output_dir,
            'sensitivity_info': sensitivity_info
        }
    
    def plot_sensitivity_all_lags(
        self,
        X_base: np.ndarray,
        variable: str,
        n_points: int = 30,
        save_path: Optional[str] = None
    ) -> None:
        """Plot sensitivity analysis for all lags (t-12 to t-1) of a variable.
        
        生成热力图，展示不同滞后时刻的敏感性。
        """
        # 找到该变量的所有滞后特征
        lag_features = []
        for i in range(1, self.config.lag_window + 1):
            feature_name = f"{variable}_lag{i}"
            if feature_name in self.feature_names:
                lag_features.append((i, feature_name))
        
        if not lag_features:
            print(f"No lag features found for {variable}")
            return
        
        var_cn = VARIABLE_NAMES_CN.get(variable, variable)
        
        # 创建子图: 2行6列 (对应 t-1 到 t-12)
        n_lags = len(lag_features)
        n_cols = min(6, n_lags)
        n_rows = (n_lags + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (lag, feature_name) in enumerate(lag_features):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            feature_idx = self.feature_names.index(feature_name)
            base_val = X_base[0, feature_idx]
            feature_std = self.scaler.scale_[feature_idx] if hasattr(self.scaler, 'scale_') else 1.0
            
            val_min = base_val - 2 * feature_std
            val_max = base_val + 2 * feature_std
            values = np.linspace(val_min, val_max, n_points)
            
            predictions = self.sensitivity_analysis(X_base, feature_idx, values)
            
            # 只绘制 t+1 和 t+6 的预测变化
            ax.plot(values, predictions[:, 0], 'b-', label='t+1步', linewidth=1.5, alpha=0.8)
            ax.plot(values, predictions[:, 5], 'r--', label='t+6步', linewidth=1.5, alpha=0.8)
            ax.axvline(x=base_val, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f't-{lag}', fontsize=11)
            ax.set_xlabel(f'{var_cn}值')
            ax.set_ylabel('预测浊度')
            if idx == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(lag_features), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'{var_cn} 各时刻历史输入的敏感性分析\n(t-N 表示 N 步之前的输入值)', 
                     fontsize=12, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
