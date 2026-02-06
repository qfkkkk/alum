"""
Configuration for NaClO Turbidity Segmentation Analysis
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """Configuration class for turbidity segmentation analysis."""
    
    # Data columns
    turbidity_col: str = "turb_jinshui_1"  # 进水浊度列名
    dose_col: str = "dose_1"               # 投药量列名
    time_col: str = "time"                 # 时间列名
    
    # Segmentation thresholds (NTU) - 可根据分析结果调整
    segment_bins: List[float] = field(default_factory=lambda: [0, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')])
    segment_labels: List[str] = field(default_factory=lambda: [
        '极低(<0.5)', '低(0.5-1)', '中低(1-2)', '中(2-5)', '高(5-10)', '极高(>10)'
    ])
    
    # Decision thresholds
    min_samples_data_driven: int = 1000    # 数据驱动最小样本量
    high_correlation_threshold: float = 0.7  # 高相关性阈值
    high_r2_threshold: float = 0.6          # 高R²阈值
    low_cv_threshold: float = 0.2           # 低变异系数阈值
    
    # Output paths
    output_dir: str = "output/naclo_segmentation"
    figures_dir: str = "output/naclo_segmentation/figures"
    docs_dir: str = "docs/naclo_segmentation"
    
    # Analysis parameters
    random_seed: int = 42
