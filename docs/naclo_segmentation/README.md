# 次氯酸钠投加量分段分析模块

## 目标

根据进水浊度分段，分析各段适合使用**数据驱动**还是**机理模型**进行投药优化。

## 运行分析

```bash
# 基础分析
python scripts/analyze_naclo_segmentation.py

# 生成完整报告
python scripts/analyze_naclo_segmentation.py --generate-report
```

## 分段策略

| 浊度段 | 可能的建模方法 |
|-------|---------------|
| 极低 (< 0.5 NTU) | 机理模型 / 固定比例 |
| 低 (0.5-1 NTU) | 根据分析结果决定 |
| 中低 (1-2 NTU) | 根据分析结果决定 |
| 中 (2-5 NTU) | 根据分析结果决定 |
| 高 (5-10 NTU) | 根据分析结果决定 |
| 极高 (> 10 NTU) | 机理模型 + 安全约束 |

## 决策依据

| 指标 | 数据驱动 | 机理模型 |
|-----|---------|---------|
| 样本量 | > 1000 | < 1000 |
| 相关系数 | - | > 0.7 |
| 线性 R² | < 0.6 | > 0.6 |
| 变异系数 | > 0.2 | < 0.2 |

## 输出文件

```
output/naclo_segmentation/
├── figures/
│   ├── turbidity_distribution.png
│   ├── segment_counts.png
│   ├── dose_vs_turbidity.png
│   └── segment_metrics.png
└── segment_analysis.csv

docs/naclo_segmentation/
├── README.md
└── SEGMENTATION_ANALYSIS.md
```
