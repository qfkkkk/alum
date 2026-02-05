# 出水浊度预测模型

多模型时序预测系统，使用 MIMO 策略同时预测未来 **6 步** 的出水浊度。

## 模型配置

### 输入特征 (7 个原始特征)

| 特征 | 说明 |
|------|------|
| `dose_1` | 投药量 |
| `turb_chushui_1` | 出水浊度 |
| `turb_jinshui_1` | 进水浊度 |
| `flow_1` | 流量 |
| `pH` | pH值 |
| `temp_down` | 水下温度 |
| `temp_shuimian` | 水面温度 |

### 特征工程参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `lag_window` | 12 | 滞后窗口，使用过去 12 个时间点的数据 |
| `rolling_windows` | [3, 6, 12] | 滚动统计窗口大小 |
| `horizon` | 6 | 预测未来 6 步 |

### Rolling Windows 说明

**滚动窗口特征**是对过去一段时间内的数据计算统计量，用于捕捉时序数据的**趋势**和**波动性**。

对于每个原始特征，会生成以下滚动统计特征：
- `{feature}_roll{window}_mean`：过去 N 个点的均值（趋势）
- `{feature}_roll{window}_std`：过去 N 个点的标准差（波动性）

**示例：**
```
假设当前时间点为 t，rolling_window = 3

turb_chushui_1_roll3_mean = mean(t-3, t-2, t-1)  # 过去 3 点均值
turb_chushui_1_roll3_std  = std(t-3, t-2, t-1)   # 过去 3 点标准差
```

### 总特征数：126 个

| 特征类型 | 计算方式 | 数量 |
|---------|---------|------|
| Lag 特征 | 7 特征 × 12 滞后 | 84 |
| Rolling 特征 | 7 特征 × 3 窗口 × 2 统计量 | 42 |
| **合计** | | **126** |

---

## 候选模型

- RandomForest
- Ridge
- XGBoost
- LightGBM

---

## 运行训练

```bash
python scripts/train_effluent_turbidity.py
```

**数据划分：** Train 60% / Val 20% / Test 20%

---

## 模型分析

训练完成后，运行分析脚本生成可视化：

```bash
# 完整分析（包含报告生成）
python scripts/analyze_model.py --generate-report

# 仅特征重要性
python scripts/analyze_model.py --only-importance

# 特定变量敏感性分析
python scripts/analyze_model.py --sensitivity dose_1_lag1
```

**生成图表：**

| 图表 | 说明 |
|------|------|
| `prediction_vs_actual_t*.png` | 预测 vs 真实值对比 |
| `r2_by_step.png` | R² 随预测步数衰减 |
| `residual_distribution.png` | 残差分布 |
| `feature_importance_top20.png` | Top 20 特征重要性 |
| `importance_by_variable.png` | 按原始变量分组的重要性 |
| `lag_importance_*.png` | 各变量的滞后重要性 |
| `sensitivity_*.png` | 敏感性分析曲线 |

---

## 输出文件

```
output/effluent_turbidity/
├── models/
│   ├── best_model.pkl      # 最佳模型
│   └── scaler.pkl          # 特征缩放器
├── figures/                # 可视化图表
├── model_comparison.csv    # 模型对比结果
├── test_predictions.csv    # 测试集预测
├── feature_importance.csv  # 特征重要性
└── feature_info.json       # 特征信息
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```
