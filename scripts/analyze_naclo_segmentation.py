#!/usr/bin/env python3
"""
NaClO Turbidity Segmentation Analysis Script
次氯酸钠投加量分段分析脚本

Usage:
    python scripts/analyze_naclo_segmentation.py
    
    # With custom data path
    python scripts/analyze_naclo_segmentation.py --data data/processed_data.csv
    
    # Generate markdown report
    python scripts/analyze_naclo_segmentation.py --generate-report
"""
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.naclo_segmentation import Config, TurbidityAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze turbidity segmentation for NaClO dosing"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed_data.csv",
        help="Path to input data CSV"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate markdown analysis report"
    )
    return parser.parse_args()


def generate_markdown_report(overview: dict, analysis_df, figures_dir: str, output_path: str):
    """Generate markdown analysis report."""
    
    docs_dir = os.path.dirname(output_path)
    figures_rel = os.path.relpath(figures_dir, docs_dir)
    
    report = f"""# 次氯酸钠投加量分段分析报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. 数据概览

| 项目 | 值 |
|------|-----|
| 总样本数 | {overview['total_samples']:,} |
| 进水浊度范围 | {overview['turbidity_min']:.3f} ~ {overview['turbidity_max']:.3f} NTU |
| 进水浊度均值 | {overview['turbidity_mean']:.3f} NTU |
| 投药量范围 | {overview['dose_min']:.3f} ~ {overview['dose_max']:.3f} |
| 投药量均值 | {overview['dose_mean']:.3f} |

**进水浊度分位数：**
"""
    for k, v in overview['quantiles'].items():
        report += f"- {k}: {v:.3f} NTU\n"
    
    report += f"""
---

## 2. 进水浊度分布

![进水浊度分布]({figures_rel}/turbidity_distribution.png)

**分析：**
- 数据表明进水浊度呈现特定分布特征
- 灰色虚线为分段阈值

---

## 3. 分段统计

![分段统计]({figures_rel}/segment_counts.png)

**各段样本量：**

| 浊度段 | 样本数 | 占比 |
|-------|-------|------|
"""
    for _, row in analysis_df.iterrows():
        report += f"| {row['segment']} | {int(row['sample_count']):,} | {row['percentage']:.1f}% |\n"
    
    report += f"""
---

## 4. 投药量 vs 进水浊度关系

![散点图]({figures_rel}/dose_vs_turbidity.png)

**分析：**
- 不同颜色代表不同的浊度分段
- 可观察各段内投药量与浊度的关系模式

---

## 5. 关系明确性分析

![分析指标]({figures_rel}/segment_metrics.png)

**指标说明：**
- **相关系数**：衡量投药量与浊度的线性相关强度（> 0.7 为高相关）
- **线性R²**：线性模型拟合优度（> 0.6 表示线性关系明确）
- **变异系数 (CV)**：投药量的相对波动程度（< 0.2 表示投药稳定）

---

## 6. 多特征分析

### 6.1 流量与投药量关系

![多特征分析]({figures_rel}/multifeature_analysis.png)

**分析：**
- 左图：投药量 vs 流量，不同颜色表示不同浊度段
- 右图：投药量 vs 浊度×流量（交互特征），探索组合效应

### 6.2 水温与投药量关系

![水温分析]({figures_rel}/temperature_analysis.png)

**分析：**
- 水温可能影响药剂反应速率
- 低温时可能需要增加投药量

---

## 7. 决策表格

根据分析结果，各段建议使用的建模方法：

| 浊度段 | 样本量 | 相关系数 | 线性R² | 变异系数 | 推荐方法 | 理由 |
|-------|-------|---------|-------|---------|---------|------|
"""
    for _, row in analysis_df.iterrows():
        corr = f"{row['correlation']:.3f}" if not pd.isna(row['correlation']) else "-"
        r2 = f"{row['r2_linear']:.3f}" if not pd.isna(row['r2_linear']) else "-"
        cv = f"{row['cv_dose']:.3f}" if not pd.isna(row['cv_dose']) else "-"
        
        # 生成理由
        cv_val = row['cv_dose'] if not pd.isna(row['cv_dose']) else 1
        r2_val = row['r2_linear'] if not pd.isna(row['r2_linear']) else 0
        corr_val = abs(row['correlation']) if not pd.isna(row['correlation']) else 0
        
        if row['sample_count'] < 500:
            reason = "样本量不足，需补充实验"
        elif r2_val > 0.5:
            reason = f"R²={r2_val:.2f}>0.5，线性关系明确"
        elif corr_val > 0.3:
            reason = f"相关系数={corr_val:.2f}>0.3，有一定相关性"
        elif cv_val < 0.3:
            reason = f"CV={cv_val:.2f}<0.3，投药稳定"
        elif cv_val > 1.0:
            reason = f"CV={cv_val:.2f}>1.0，波动非常大"
        else:
            reason = f"R²={r2_val:.2f}<0.5，相关系数低，关系复杂"
        
        report += f"| {row['segment']} | {int(row['sample_count']):,} | {corr} | {r2} | {cv} | {row['recommendation']} | {reason} |\n"
    
    report += f"""
---

## 8. 决策逻辑说明

### 什么时候用数据驱动？
- 样本量充足（> 1000）
- 关系复杂，难以用简单公式描述
- 存在非线性或交互效应

### 什么时候用机理模型？
- 样本量少但关系明确（线性R² > 0.6）
- 投药行为稳定（CV < 0.2）
- 有明确的物理/化学原理支撑

### 什么时候需要烧杯实验？
- 样本量不足
- 关系不明确（低相关、低R²）
- 极端工况下的安全验证

---
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")


def main():
    """Main analysis pipeline."""
    args = parse_args()
    
    # Initialize
    config = Config()
    analyzer = TurbidityAnalyzer(config)
    
    # Load data
    print(f"Loading data from: {args.data}")
    df = analyzer.load_data(args.data)
    
    # Run analysis
    results = analyzer.run_full_analysis(df)
    
    # Generate report if requested
    if args.generate_report:
        print("\n[5/5] 生成Markdown报告...")
        import pandas as pd  # Import here for report generation
        generate_markdown_report(
            overview=results['overview'],
            analysis_df=results['analysis_df'],
            figures_dir=results['figures_dir'],
            output_path=os.path.join(config.docs_dir, "SEGMENTATION_ANALYSIS.md")
        )
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    import pandas as pd  # Import at module level for report function
    main()
