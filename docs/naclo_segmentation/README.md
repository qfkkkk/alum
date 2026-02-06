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

## 分段策略（水厂标准）

| 浊度段 | 浊度范围 |
|-------|---------|
| 低浊 | 0-20 NTU |
| 中低浊 | 20-50 NTU |
| 中浊 | 50-100 NTU |
| 中高浊 | 100-200 NTU |
| 高浊 | > 200 NTU |

## 决策逻辑（基于数据指标）

| 条件 | 推荐方法 |
|-----|---------|
| 样本量 < 500 | 烧杯实验 + 专家规则 |
| R² > 0.5 | 机理模型（线性关系明确） |
| 相关系数 > 0.3 | 数据驱动（有相关性但非线性） |
| CV < 0.3 | 固定投药规则（投药稳定） |
| CV > 1.0 | 数据驱动 + 专家约束（波动大） |

## 分析内容

1. **进水浊度分布** - 直方图 + 分段阈值
2. **分段统计** - 各段样本量和占比
3. **投药量 vs 浊度关系** - 分段着色散点图
4. **关系明确性分析** - 相关系数、R²、CV
5. **多特征分析** - 流量、浊度×流量交互
6. **水温分析** - 水下温度、水面温度

## 输出文件

```
output/naclo_segmentation/
├── figures/
│   ├── turbidity_distribution.png
│   ├── segment_counts.png
│   ├── dose_vs_turbidity.png
│   ├── segment_metrics.png
│   ├── multifeature_analysis.png
│   └── temperature_analysis.png
└── segment_analysis.csv

docs/naclo_segmentation/
├── README.md
└── SEGMENTATION_ANALYSIS.md
```
