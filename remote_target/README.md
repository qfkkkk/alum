# xPatch 出水浊度预测

基于 xPatch 模型的水厂出水浊度多步预测系统。输入 60 个时间步（5min/步）的历史数据，预测未来 6 步的出水浊度。

*结果及模型：https://drive.google.com/drive/folders/1Eo2uLMbdD6wXgURYTUr64zjW4ya7qn-N?dmr=1&ec=wgc-drive-hero-goto*

## 项目结构

```
remote_target/
├── train.py          # 训练脚本
├── evaluate.py       # 评估 + 可视化
├── predict.py        # 推理/预测
├── dataset.py        # 数据加载与预处理
├── requirements.txt  # 依赖
├── models/
│   └── xPatch.py     # xPatch 模型
└── layers/
    ├── revin.py       # RevIN 归一化
    ├── decomp.py      # 序列分解
    ├── network.py     # 网络模块
    ├── ema.py         # 指数移动平均
    └── dema.py        # 双指数移动平均
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练

```bash
# 训练单个池子（默认差分目标）
python train.py --pool 1 --epochs 100 --csv ../train_data.csv

# 训练所有池子
python train.py --all --epochs 100

# CPU 快速测试
python train.py --pool 1 --epochs 3 --test-mode

# 直接预测模式（不使用差分）
python train.py --pool 1 --epochs 100 --no-diff
```

### 3. 评估

```bash
# 评估单个池子
python evaluate.py --pool 1

# 评估所有池子（显示汇总表）
python evaluate.py --all

# 自定义抽样数
python evaluate.py --pool 1 --n-samples 8
```

### 4. 预测

```bash
# 从 CSV 取最后 30 行预测
python predict.py --pool 1 --csv ../train_data.csv

# 指定起始行
python predict.py --pool 1 --csv ../train_data.csv --start-row 1000

# 从 JSON 文件输入
python predict.py --pool 1 --json input.json
```

## 输入特征（每个池子 6 个）

| 特征 | 说明 |
|------|------|
| `dose_X` | 加药量 |
| `turb_chushui_X` | 出水浊度（预测目标） |
| `turb_jinshui_X` | 进水浊度 |
| `flow_X` | 流量 |
| `pH` | pH 值（共享） |
| `temp_shuimian` | 水面温度（共享） |

## 输出结构

```
output/
├── pool_1/
│   ├── best_model.pt        # 模型权重
│   ├── scaler.pkl           # StandardScaler
│   ├── train_history.json   # 训练损失记录
│   ├── train_curve.png      # 训练曲线
│   ├── test_metrics.json    # 测试集指标
│   └── figures/
│       ├── per_step_errors.png       # 分步误差柱状图
│       ├── sample_predictions.png    # 抽样预测对比图
│       └── scatter_pred_vs_actual.png # 散点图
├── pool_2/ ...
├── pool_3/ ...
└── pool_4/ ...
```

## 训练结果

### 总体指标

| Pool | MAE | RMSE | R² | 方向准确率 |
|------|-------|-------|--------|-----------|
| 1 | 0.0722 | 0.2650 | 0.7340 | 45.20% |
| 2 | 0.1102 | 0.3256 | 0.7683 | 49.62% |
| 3 | 0.0808 | 0.1677 | 0.8919 | 51.35% |
| 4 | 0.0886 | 0.3124 | 0.9335 | 33.65% |

### 分步指标（MAE / RMSE）

| Step | Pool 1 | Pool 2 | Pool 3 | Pool 4 |
|------|--------|--------|--------|--------|
| t+1 | 0.0459 / 0.2212 | 0.0674 / 0.2683 | 0.0492 / 0.1134 | 0.0516 / 0.2119 |
| t+2 | 0.0598 / 0.2517 | 0.0917 / 0.3029 | 0.0664 / 0.1426 | 0.0731 / 0.2627 |
| t+3 | 0.0697 / 0.2620 | 0.1056 / 0.3214 | 0.0776 / 0.1607 | 0.0851 / 0.2935 |
| t+4 | 0.0781 / 0.2721 | 0.1215 / 0.3395 | 0.0886 / 0.1779 | 0.0970 / 0.3236 |
| t+5 | 0.0869 / 0.2848 | 0.1316 / 0.3482 | 0.0973 / 0.1911 | 0.1079 / 0.3586 |
| t+6 | 0.0931 / 0.2917 | 0.1434 / 0.3641 | 0.1058 / 0.2041 | 0.1166 / 0.3899 |

## 模型设计

### 防预测偏移策略

1. **差分目标**（默认开启）: 预测 Δy = y_{t+k} - y_t，而非绝对值，迫使模型学习实际变化
2. **RevIN**: 可逆实例归一化，消除输入分布漂移
3. **xPatch**: 基于 Patch 的时序预测，结合 EMA/DEMA 分解 + CNN + MLP

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seq-len` | 60 | 输入窗口长度 |
| `--pred-len` | 6 | 预测步数 |
| `--patch-len` | 6 | Patch 长度 |
| `--stride` | 3 | Patch 步长 |
| `--lr` | 1e-3 | 学习率 |
| `--patience` | 10 | 早停耐心值 |
| `--ma-type` | ema | 移动平均类型 (ema/dema/reg) |
| `--no-diff` | - | 关闭差分目标 |
| `--no-revin` | - | 关闭 RevIN |

## JSON 输入格式

```json
{
  "data": [
    {"dose": 15.3, "turb_chushui": 0.31, "turb_jinshui": 54.8, "flow": 4098.5, "pH": 5.99, "temp_shuimian": 15.26},
    ...
  ]
}
```

共 60 行，每行包含 6 个特征。


