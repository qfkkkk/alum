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
                cv=cv
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
        cv: float
    ) -> str:
        """Determine recommendation based on analysis."""
        cfg = self.config
        
        # 样本量检查
        if sample_count < cfg.min_samples_data_driven:
            if r2 > cfg.high_r2_threshold:
                return "机理模型（样本少但线性关系明确）"
            else:
                return "烧杯实验 + 专家规则"
        
        # 相关性和R²检查
        if abs(correlation) > cfg.high_correlation_threshold and r2 > cfg.high_r2_threshold:
            if cv < cfg.low_cv_threshold:
                return "机理模型（关系明确且稳定）"
            else:
                return "数据驱动（关系明确但有波动）"
        
        if r2 > cfg.high_r2_threshold:
            return "数据驱动（线性关系较好）"
        
        if cv < cfg.low_cv_threshold:
            return "机理模型（投药稳定）"
        
        return "数据驱动 + 机理约束（关系复杂）"
    
    # ==================== Visualization ====================
    
    def plot_turbidity_distribution(
        self, 
        df: pd.DataFrame, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot turbidity distribution histogram."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        turb = df[self.config.turbidity_col]
        
        ax.hist(turb, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(turb.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {turb.mean():.2f}')
        ax.axvline(turb.median(), color='orange', linestyle='-', linewidth=2, label=f'中位数: {turb.median():.2f}')
        
        # 添加分段线
        for threshold in self.config.segment_bins[1:-1]:
            ax.axvline(threshold, color='gray', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('进水浊度 (NTU)')
        ax.set_ylabel('频数')
        ax.set_title('进水浊度分布')
        ax.legend()
        
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
        """Plot segment sample counts."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 柱状图
        ax1 = axes[0]
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(analysis_df)))
        bars = ax1.bar(analysis_df['segment'], analysis_df['sample_count'], color=colors)
        ax1.set_xlabel('进水浊度段')
        ax1.set_ylabel('样本数量')
        ax1.set_title('各段样本数量')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        # 添加数值标签
        for bar, pct in zip(bars, analysis_df['percentage']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 饼图
        ax2 = axes[1]
        valid_data = analysis_df[analysis_df['sample_count'] > 0]
        ax2.pie(valid_data['sample_count'], labels=valid_data['segment'], 
                autopct='%1.1f%%', colors=colors[:len(valid_data)])
        ax2.set_title('各段占比')
        
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
        """Plot dose vs turbidity scatter with segments colored."""
        df = self.segment_data(df)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 分段着色
        segments = df['segment'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
        
        for seg, color in zip(segments, colors):
            seg_data = df[df['segment'] == seg]
            ax.scatter(
                seg_data[self.config.turbidity_col],
                seg_data[self.config.dose_col],
                alpha=0.3, s=10, label=seg, color=color
            )
        
        ax.set_xlabel('进水浊度 (NTU)')
        ax.set_ylabel('投药量')
        ax.set_title('投药量 vs 进水浊度（分段着色）')
        ax.legend(title='浊度段', loc='upper left')
        
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
        
        # 4. Save analysis results
        print("[4/5] 保存分析结果...")
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
