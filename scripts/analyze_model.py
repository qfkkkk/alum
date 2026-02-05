#!/usr/bin/env python3
"""
Model Analysis Script for Effluent Turbidity Prediction

Usage:
    python scripts/analyze_model.py
    
    # Generate full analysis with markdown report
    python scripts/analyze_model.py --generate-report
    
    # Only generate feature importance
    python scripts/analyze_model.py --only-importance
    
    # Sensitivity analysis for specific feature
    python scripts/analyze_model.py --sensitivity dose_1_lag1
"""
import os
import sys
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.effluent_turbidity import (
    Config,
    FeatureEngineer,
    load_data,
    time_series_split,
)
from models.effluent_turbidity.utils import load_model
from models.effluent_turbidity.analysis import ModelAnalyzer, VARIABLE_NAMES_CN
from models.effluent_turbidity.model_selection import ModelSelector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Effluent Turbidity Prediction Model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed_data.csv",
        help="Path to input data CSV"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="output/effluent_turbidity",
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--only-importance",
        action="store_true",
        help="Only generate feature importance analysis"
    )
    parser.add_argument(
        "--sensitivity",
        type=str,
        default=None,
        help="Feature name for sensitivity analysis"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate markdown analysis report in docs/"
    )
    return parser.parse_args()


def generate_markdown_report(
    feature_info: dict,
    metrics: dict,
    importance_df,
    figures_dir: str,
    output_path: str
):
    """Generate markdown analysis report with embedded figures."""
    
    # Get relative path from docs to figures
    docs_dir = os.path.dirname(output_path)
    figures_rel = os.path.relpath(figures_dir, docs_dir)
    
    best_model = feature_info['best_model']
    test_r2 = metrics['r2']
    
    # Top 10 features
    top_features = importance_df.head(10)
    
    report = f"""# 出水浊度预测模型 - 分析报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. 模型概述

| 项目 | 值 |
|------|-----|
| 最佳模型 | **{best_model}** |
| 测试集 R² | **{test_r2:.4f}** |
| 特征数量 | {len(feature_info['feature_names'])} |
| 预测步数 | 6 步 |

---

## 2. 预测性能分析

### 2.1 预测值 vs 真实值（典型示例）

从测试集中选取了 3 个典型时间窗口，展示模型的预测效果：

![预测示例]({figures_rel}/prediction_examples.png)

**说明：**
- 蓝色实线：真实值
- 红色虚线：预测值
- 每行展示一个时间窗口，分别来自测试集的开始、中间和结束时段

### 2.2 R² 分数衰减

![R² 衰减]({figures_rel}/r2_by_step.png)

**分析：**
- t+1 步 R²: **{metrics.get('r2_t+1', 0):.4f}**
- t+6 步 R²: **{metrics.get('r2_t+6', 0):.4f}**
- 整体衰减幅度约 {((metrics.get('r2_t+1', 0) - metrics.get('r2_t+6', 0)) / metrics.get('r2_t+1', 1) * 100):.1f}%

### 2.3 残差分布

![残差分布]({figures_rel}/residual_distribution.png)

**说明：**
- 残差 = 真实值 - 预测值
- 残差接近 0 表示预测准确
- 红色虚线为零线，橙色线为残差均值

---

## 3. 特征重要性分析

### 3.1 什么是滞后特征？

**滞后特征 (t-N)** 表示使用 **N 个时间步之前**的历史数据作为输入。

例如：
- `出水浊度(t-1)`：上一个时间点的出水浊度值
- `进水浊度(t-6)`：6 个时间步之前的进水浊度值（约 30 分钟前）

**为什么需要滞后特征？**
水处理过程存在时间延迟，当前的进水浊度需要经过一段时间才能影响出水浊度。通过分析不同滞后的重要性，可以推断出这个延迟时间。

### 3.2 Top 20 特征重要性

![特征重要性]({figures_rel}/feature_importance_top20.png)

**Top 10 重要特征：**

| 排名 | 特征 | 重要性 |
|:---:|------|-------:|
"""
    
    for i, row in top_features.iterrows():
        report += f"| {i+1} | {row['feature_cn']} | {row['importance']:.4f} |\n"
    
    report += f"""
### 3.3 按原始变量分组

![按变量分组]({figures_rel}/importance_by_variable.png)

**分析：**
- **出水浊度历史值**对预测影响最大，体现了时序数据的自相关性
- **进水浊度**是第二重要因素，反映了进出水的因果关系
- **投药量**的影响体现了控制变量对出水质量的调节作用

### 3.4 滞后效应分析

分析不同时间点的历史输入对预测的影响程度：

| 进水浊度 | 投药量 | 出水浊度 |
|:---:|:---:|:---:|
| ![进水浊度滞后]({figures_rel}/lag_importance_turb_jinshui_1.png) | ![投药量滞后]({figures_rel}/lag_importance_dose_1.png) | ![出水浊度滞后]({figures_rel}/lag_importance_turb_chushui_1.png) |

**如何解读？**
- 横坐标 `t-N` 表示 N 个时间步之前的数据
- 柱子越高，表示该时间点的历史数据对预测越重要
- 例如：如果 t-6 的进水浊度重要性高，说明进水约 30 分钟后影响出水

---

## 4. 敏感性分析

### 4.1 什么是敏感性分析？

敏感性分析用于回答：**如果某个输入变化，模型预测会如何变化？**

**方法：**
1. 固定其他所有输入特征不变
2. 仅改变目标特征的值（在基准值 ± 2倍标准差范围内变化）
3. 观察模型预测输出如何随之变化

### 4.2 各变量敏感性（t-12 到 t-1）

每个变量的所有历史时刻输入的敏感性分析：

**投药量敏感性：**
![投药量敏感性]({figures_rel}/sensitivity_dose_1_all_lags.png)

**进水浊度敏感性：**
![进水浊度敏感性]({figures_rel}/sensitivity_turb_jinshui_1_all_lags.png)

**流量敏感性：**
![流量敏感性]({figures_rel}/sensitivity_flow_1_all_lags.png)

**如何解读？**
- 每个子图对应一个历史时刻 (t-1 到 t-12)
- 蓝色实线：t+1 步预测，红色虚线：t+6 步预测
- 曲线斜率越陡，表示该历史输入对预测影响越大
- 可用于评估"增加/减少某时刻的投药量 X 单位，出水浊度大约变化多少"

---

## 5. 结论与建议

### 5.1 模型性能

- 模型整体 R² 达到 **{test_r2:.4f}**，预测效果良好
- 短期预测（t+1 ~ t+3）精度较高，适合实时预警
- 长期预测（t+4 ~ t+6）可用于趋势判断

### 5.2 数据洞察

1. **时序自相关**：出水浊度历史值是最重要的预测因子
2. **因果关系**：进水浊度经过一定时间延迟后影响出水浊度
3. **控制效果**：投药量对出水浊度有调节作用

### 5.3 应用建议

- **实时预测**：使用 t+1 ~ t+3 步预测进行实时监控和预警
- **投药优化**：结合敏感性分析，优化投药策略
- **异常检测**：当预测残差超过阈值时，可触发异常告警
"""
    
    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")


def main():
    """Main analysis pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("Effluent Turbidity Model - Analysis")
    print("=" * 60)
    
    # Initialize configuration
    config = Config(output_dir=args.model_dir)
    
    # Step 1: Load trained model and scaler
    print("\n[Step 1] Loading trained model...")
    model_path = os.path.join(args.model_dir, "models", "best_model.pkl")
    scaler_path = os.path.join(args.model_dir, "models", "scaler.pkl")
    feature_info_path = os.path.join(args.model_dir, "feature_info.json")
    
    model = load_model(model_path)
    scaler = load_model(scaler_path)
    
    with open(feature_info_path, 'r') as f:
        feature_info = json.load(f)
    
    feature_names = feature_info['feature_names']
    target_names = feature_info['target_names']
    
    print(f"Loaded model: {feature_info['best_model']}")
    print(f"Number of features: {len(feature_names)}")
    
    # Step 2: Load and process data
    print("\n[Step 2] Loading and processing data...")
    df = load_data(args.data)
    
    fe = FeatureEngineer(config)
    df_processed, _, _ = fe.prepare_dataset(df)
    
    # Get test set
    train_df, val_df, test_df = time_series_split(
        df_processed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    X_test, y_test = fe.get_X_y(test_df, feature_names, target_names)
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    selector = ModelSelector(config)
    metrics = selector.evaluate_model(model, X_test_scaled, y_test)
    
    print(f"\nTest R²: {metrics['r2']:.4f}")
    
    # Step 3: Initialize analyzer
    print("\n[Step 3] Initializing analyzer...")
    analyzer = ModelAnalyzer(model, scaler, feature_names, config)
    
    # Step 4: Run analysis
    if args.only_importance:
        print("\n[Step 4] Running feature importance analysis only...")
        importance_df = analyzer.plot_feature_importance(
            top_n=20,
            save_path=os.path.join(analyzer.output_dir, "feature_importance_top20.png")
        )
        importance_df.to_csv(
            os.path.join(config.output_dir, "feature_importance.csv"),
            index=False
        )
        analyzer.plot_importance_by_variable(
            save_path=os.path.join(analyzer.output_dir, "importance_by_variable.png")
        )
        print(f"\n✓ Feature importance saved to: {analyzer.output_dir}/")
        
    elif args.sensitivity:
        print(f"\n[Step 4] Running sensitivity analysis for: {args.sensitivity}")
        X_base = X_test[len(X_test)//2:len(X_test)//2+1].copy()
        analyzer.plot_sensitivity_curve(
            X_base, args.sensitivity,
            save_path=os.path.join(analyzer.output_dir, f"sensitivity_{args.sensitivity}.png")
        )
        
    else:
        print("\n[Step 4] Running full analysis...")
        result = analyzer.run_all_analysis(
            y_true=y_test,
            y_pred=y_pred,
            X_test=X_test,
            index=test_df.index,
            metrics=metrics
        )
        
        # Generate markdown report if requested
        if args.generate_report:
            print("\n[Step 5] Generating markdown report...")
            report_path = os.path.join("docs", "effluent_turbidity", "ANALYSIS_REPORT.md")
            generate_markdown_report(
                feature_info=feature_info,
                metrics=metrics,
                importance_df=result['importance_df'],
                figures_dir=result['figures_dir'],
                output_path=report_path
            )
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
