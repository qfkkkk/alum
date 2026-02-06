"""
Data Analysis for NaClO Turbidity Segmentation
进水浊度分段分析模块
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from .config import Config


# 设置中文字体
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TurbidityAnalyzer:
    """Turbidity segmentation analyzer for NaClO dosing."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        os.makedirs(self.config.figures_dir, exist_ok=True)
        os.makedirs(self.config.docs_dir, exist_ok=True)
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess data."""
        df = pd.read_csv(filepath)
        
        # 确保必要的列存在
        required_cols = [self.config.turbidity_col, self.config.dose_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 移除缺失值
        df = df.dropna(subset=required_cols)
        
        return df
    
    def get_data_overview(self, df: pd.DataFrame) -> Dict:
        """Get data overview statistics."""
        turb = df[self.config.turbidity_col]
        dose = df[self.config.dose_col]
        
        overview = {
            'total_samples': len(df),
            'turbidity_min': turb.min(),
            'turbidity_max': turb.max(),
            'turbidity_mean': turb.mean(),
            'turbidity_std': turb.std(),
            'dose_min': dose.min(),
            'dose_max': dose.max(),
            'dose_mean': dose.mean(),
            'dose_std': dose.std(),
            'quantiles': {
                f'p{int(q*100)}': turb.quantile(q) 
                for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            }
        }
        return overview
    
    def segment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add segment labels to data."""
        df = df.copy()
        df['segment'] = pd.cut(
            df[self.config.turbidity_col],
            bins=self.config.segment_bins,
            labels=self.config.segment_labels
        )
        return df
    
    def analyze_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze each segment for decision making."""
        df = self.segment_data(df)
        
        results = []
        for segment in self.config.segment_labels:
            seg_data = df[df['segment'] == segment]
            
            if len(seg_data) < 10:
                results.append({
                    'segment': segment,
                    'sample_count': len(seg_data),
                    'percentage': len(seg_data) / len(df) * 100,
                    'correlation': np.nan,
                    'r2_linear': np.nan,
                    'cv_dose': np.nan,
                    'recommendation': '样本不足'
                })
                continue
            
            turb = seg_data[self.config.turbidity_col].values
            dose = seg_data[self.config.dose_col].values
            
            # 1. Correlation
            corr, _ = stats.pearsonr(turb, dose)
            
            # 2. Linear R²
            model = LinearRegression()
            model.fit(turb.reshape(-1, 1), dose)
            r2 = model.score(turb.reshape(-1, 1), dose)
            
            # 3. Coefficient of Variation
            cv = dose.std() / dose.mean() if dose.mean() > 0 else np.nan
            
            # 4. Decision logic
            recommendation = self._get_recommendation(
                sample_count=len(seg_data),
                correlation=corr,
                r2=r2,
                cv=cv,
                segment=segment
            )
            
            results.append({
                'segment': segment,
                'sample_count': len(seg_data),
                'percentage': len(seg_data) / len(df) * 100,
                'turbidity_mean': turb.mean(),
                'dose_mean': dose.mean(),
                'dose_std': dose.std(),
                'correlation': corr,
                'r2_linear': r2,
                'cv_dose': cv,
                'recommendation': recommendation
            })
        
        return pd.DataFrame(results)
    
    def _get_recommendation(
        self, 
        sample_count: int, 
        correlation: float, 
        r2: float, 
        cv: float,
        segment: str = ""
    ) -> str:
        """基于数据指标推荐建模方法。
        
        决策逻辑：
        1. 样本量不足 → 烧杯实验
        2. 线性关系明确 (R² > 0.5) → 机理模型
        3. 有相关性但非线性 → 数据驱动
        4. 投药稳定 (CV < 0.3) → 固定规则
        5. 关系复杂且波动大 → 数据驱动 + 约束
        """
        # 处理 NaN
        if np.isnan(correlation):
            correlation = 0
        if np.isnan(r2):
            r2 = 0
        if np.isnan(cv):
            cv = 1
        
        # 1. 样本量检查
        if sample_count < 500:
            return "烧杯实验 + 专家规则（样本不足）"
        
        # 2. 线性关系明确 → 机理模型
        if r2 > 0.5:
            if cv < 0.3:
                return "机理模型（线性关系明确且稳定）"
            else:
                return "机理模型 + 动态调整（线性但有波动）"
        
        # 3. 有一定相关性 → 数据驱动
        if abs(correlation) > 0.3:
            return "数据驱动（有相关性但非线性）"
        
        # 4. 投药稳定 → 固定规则
        if cv < 0.3:
            return "固定投药规则（投药稳定）"
        
        # 5. 关系复杂且波动大
        if cv > 1.0:
            return "数据驱动 + 专家约束（波动大，需边界控制）"
        
        return "数据驱动（关系复杂）"
    
    # ==================== Visualization ====================
    
    def plot_turbidity_distribution(
        self, 
        df: pd.DataFrame, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot turbidity distribution histogram with clipped x-axis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        turb = df[self.config.turbidity_col]
        
        # 裁剪到 99% 分位数以内，避免极端值拉宽坐标轴
        x_max = turb.quantile(0.99)
        turb_clipped = turb[turb <= x_max]
        
        ax.hist(turb_clipped, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(turb.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {turb.mean():.2f}')
        ax.axvline(turb.median(), color='orange', linestyle='-', linewidth=2, label=f'中位数: {turb.median():.2f}')
        
        # 添加分段线（只显示在可见范围内的）
        for threshold in self.config.segment_bins[1:-1]:
            if threshold <= x_max:
                ax.axvline(threshold, color='gray', linestyle=':', alpha=0.7)
                ax.text(threshold, ax.get_ylim()[1] * 0.9, f'{int(threshold)}', 
                       ha='center', fontsize=8, color='gray')
        
        ax.set_xlabel('进水浊度 (NTU)')
        ax.set_ylabel('频数')
        ax.set_title(f'进水浊度分布（显示 99% 分位数以内，≤{x_max:.0f} NTU）')
        ax.legend()
        ax.set_xlim(0, x_max * 1.05)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    def plot_segment_counts(
        self, 
        analysis_df: pd.DataFrame, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot segment sample counts (bar chart only, no pie chart)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 柱状图
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(analysis_df)))
        bars = ax.bar(analysis_df['segment'], analysis_df['sample_count'], color=colors)
        ax.set_xlabel('进水浊度段')
        ax.set_ylabel('样本数量')
        ax.set_title('各段样本数量分布')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # 添加数值标签（样本数和占比）
        for bar, cnt, pct in zip(bars, analysis_df['sample_count'], analysis_df['percentage']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{int(cnt):,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        # 设置 y 轴范围留出标签空间
        ax.set_ylim(0, analysis_df['sample_count'].max() * 1.2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    def plot_dose_vs_turbidity(
        self, 
        df: pd.DataFrame, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot dose vs turbidity scatter with clipped axes and improved legend."""
        df = self.segment_data(df)
        
        # 计算裁剪范围（99%分位数）
        x_max = df[self.config.turbidity_col].quantile(0.99)
        y_max = df[self.config.dose_col].quantile(0.99)
        
        # 只保留范围内的数据
        df_clipped = df[
            (df[self.config.turbidity_col] <= x_max) & 
            (df[self.config.dose_col] <= y_max)
        ]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 使用更鲜明的颜色
        segment_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        segments = self.config.segment_labels
        
        for seg, color in zip(segments, segment_colors):
            seg_data = df_clipped[df_clipped['segment'] == seg]
            if len(seg_data) > 0:
                ax.scatter(
                    seg_data[self.config.turbidity_col],
                    seg_data[self.config.dose_col],
                    alpha=0.4, s=15, label=f'{seg} ({len(seg_data):,})', 
                    color=color, edgecolors='none'
                )
        
        ax.set_xlabel('进水浊度 (NTU)', fontsize=11)
        ax.set_ylabel('投药量', fontsize=11)
        ax.set_title(f'投药量 vs 进水浊度（显示 99% 分位数以内）', fontsize=12)
        
        # 图例放在图外右侧，避免重叠
        ax.legend(title='浊度段', bbox_to_anchor=(1.02, 1), loc='upper left',
                 fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(0, x_max * 1.05)
        ax.set_ylim(0, y_max * 1.05)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    def plot_segment_metrics(
        self, 
        analysis_df: pd.DataFrame, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot segment analysis metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        valid_df = analysis_df.dropna(subset=['correlation'])
        
        # 1. Correlation
        ax1 = axes[0]
        bars1 = ax1.bar(valid_df['segment'], valid_df['correlation'], color='steelblue', alpha=0.8)
        ax1.axhline(y=self.config.high_correlation_threshold, color='red', linestyle='--', 
                   label=f'阈值 ({self.config.high_correlation_threshold})')
        ax1.set_xlabel('进水浊度段')
        ax1.set_ylabel('相关系数')
        ax1.set_title('各段相关系数')
        ax1.legend()
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        # 2. R²
        ax2 = axes[1]
        bars2 = ax2.bar(valid_df['segment'], valid_df['r2_linear'], color='green', alpha=0.8)
        ax2.axhline(y=self.config.high_r2_threshold, color='red', linestyle='--',
                   label=f'阈值 ({self.config.high_r2_threshold})')
        ax2.set_xlabel('进水浊度段')
        ax2.set_ylabel('线性R²')
        ax2.set_title('各段线性R²')
        ax2.legend()
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        # 3. CV
        ax3 = axes[2]
        bars3 = ax3.bar(valid_df['segment'], valid_df['cv_dose'], color='orange', alpha=0.8)
        ax3.axhline(y=self.config.low_cv_threshold, color='red', linestyle='--',
                   label=f'阈值 ({self.config.low_cv_threshold})')
        ax3.set_xlabel('进水浊度段')
        ax3.set_ylabel('变异系数 (CV)')
        ax3.set_title('各段投药变异系数')
        ax3.legend()
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    # ==================== Report Generation ====================
    
    def run_full_analysis(self, df: pd.DataFrame) -> Dict:
        """Run complete analysis and generate all outputs."""
        print("=" * 60)
        print("NaClO 进水浊度分段分析")
        print("=" * 60)
        
        # 1. Data overview
        print("\n[1/5] 数据概览...")
        overview = self.get_data_overview(df)
        
        # 2. Segment analysis
        print("[2/5] 分段分析...")
        analysis_df = self.analyze_segments(df)
        
        # 3. Generate figures
        print("[3/5] 生成图表...")
        self.plot_turbidity_distribution(
            df, 
            save_path=os.path.join(self.config.figures_dir, "turbidity_distribution.png")
        )
        self.plot_segment_counts(
            analysis_df,
            save_path=os.path.join(self.config.figures_dir, "segment_counts.png")
        )
        self.plot_dose_vs_turbidity(
            df,
            save_path=os.path.join(self.config.figures_dir, "dose_vs_turbidity.png")
        )
        self.plot_segment_metrics(
            analysis_df,
            save_path=os.path.join(self.config.figures_dir, "segment_metrics.png")
        )
        
        # 4. 多特征分析（如果有流量数据）
        print("[4/6] 多特征分析...")
        if 'flow_1' in df.columns:
            self.plot_multifeature_analysis(
                df,
                save_path=os.path.join(self.config.figures_dir, "multifeature_analysis.png")
            )
        else:
            print("  (跳过：未找到 flow_1 列)")
        
        # 5. 水温分析
        print("[5/6] 水温分析...")
        if 'temp_down' in df.columns or 'temp_shuimian' in df.columns:
            self.plot_temperature_analysis(
                df,
                save_path=os.path.join(self.config.figures_dir, "temperature_analysis.png")
            )
        else:
            print("  (跳过：未找到温度列)")
        
        # 6. Save analysis results
        print("[6/6] 保存分析结果...")
        analysis_df.to_csv(
            os.path.join(self.config.output_dir, "segment_analysis.csv"),
            index=False
        )
        
        print(f"\n✓ 分析完成！图表保存至: {self.config.figures_dir}/")
        
        return {
            'overview': overview,
            'analysis_df': analysis_df,
            'figures_dir': self.config.figures_dir
        }
    
    def plot_multifeature_analysis(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot multi-feature analysis: turbidity, flow, and dose relationships."""
        df = self.segment_data(df)
        
        # 裁剪到 99% 分位数
        turb_max = df[self.config.turbidity_col].quantile(0.99)
        dose_max = df[self.config.dose_col].quantile(0.99)
        flow_max = df['flow_1'].quantile(0.99) if 'flow_1' in df.columns else 1
        
        df_clipped = df[
            (df[self.config.turbidity_col] <= turb_max) & 
            (df[self.config.dose_col] <= dose_max)
        ]
        if 'flow_1' in df.columns:
            df_clipped = df_clipped[df_clipped['flow_1'] <= flow_max]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        segment_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        segments = self.config.segment_labels
        
        # 1. 流量 vs 投药量
        ax1 = axes[0]
        for seg, color in zip(segments, segment_colors):
            seg_data = df_clipped[df_clipped['segment'] == seg]
            if len(seg_data) > 0 and 'flow_1' in seg_data.columns:
                ax1.scatter(
                    seg_data['flow_1'], seg_data[self.config.dose_col],
                    alpha=0.3, s=10, label=seg, color=color
                )
        ax1.set_xlabel('流量', fontsize=11)
        ax1.set_ylabel('投药量', fontsize=11)
        ax1.set_title('投药量 vs 流量（分段着色）')
        
        # 2. 浊度 × 流量 vs 投药量（交互特征）
        ax2 = axes[1]
        if 'flow_1' in df_clipped.columns:
            df_clipped['turb_x_flow'] = df_clipped[self.config.turbidity_col] * df_clipped['flow_1']
            interaction_max = df_clipped['turb_x_flow'].quantile(0.99)
            df_clipped = df_clipped[df_clipped['turb_x_flow'] <= interaction_max]
            
            for seg, color in zip(segments, segment_colors):
                seg_data = df_clipped[df_clipped['segment'] == seg]
                if len(seg_data) > 0:
                    ax2.scatter(
                        seg_data['turb_x_flow'], seg_data[self.config.dose_col],
                        alpha=0.3, s=10, label=seg, color=color
                    )
            ax2.set_xlabel('浊度 × 流量', fontsize=11)
            ax2.set_ylabel('投药量', fontsize=11)
            ax2.set_title('投药量 vs 浊度×流量（交互特征）')
        
        # 添加共享图例在图外右侧
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, title='浊度段', 
                  bbox_to_anchor=(1.02, 0.5), loc='center left',
                  fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0, 0.88, 1])  # 为图例留出空间
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    def plot_temperature_analysis(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot temperature vs dose analysis."""
        df = self.segment_data(df)
        
        # 选择水温列
        temp_cols = []
        if 'temp_down' in df.columns:
            temp_cols.append(('temp_down', '水下温度'))
        if 'temp_shuimian' in df.columns:
            temp_cols.append(('temp_shuimian', '水面温度'))
        
        if not temp_cols:
            return
        
        # 裁剪到 99% 分位数
        dose_max = df[self.config.dose_col].quantile(0.99)
        df_clipped = df[df[self.config.dose_col] <= dose_max]
        
        fig, axes = plt.subplots(1, len(temp_cols), figsize=(7 * len(temp_cols), 6))
        if len(temp_cols) == 1:
            axes = [axes]
        
        segment_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        segments = self.config.segment_labels
        
        for ax, (temp_col, temp_name) in zip(axes, temp_cols):
            # 裁剪温度
            if temp_col in df_clipped.columns:
                temp_max = df_clipped[temp_col].quantile(0.99)
                temp_min = df_clipped[temp_col].quantile(0.01)
                df_temp = df_clipped[
                    (df_clipped[temp_col] >= temp_min) & 
                    (df_clipped[temp_col] <= temp_max)
                ]
                
                for seg, color in zip(segments, segment_colors):
                    seg_data = df_temp[df_temp['segment'] == seg]
                    if len(seg_data) > 0:
                        ax.scatter(
                            seg_data[temp_col], seg_data[self.config.dose_col],
                            alpha=0.3, s=10, label=seg, color=color
                        )
                
                ax.set_xlabel(f'{temp_name} (°C)', fontsize=11)
                ax.set_ylabel('投药量', fontsize=11)
                ax.set_title(f'投药量 vs {temp_name}（分段着色）')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
