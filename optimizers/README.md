# 水厂投矾系统 MPC 控制器

基于模型预测控制(MPC)的水厂投矾量优化控制系统，支持多种启发式优化算法。

## 功能特性

- **模型预测控制(MPC)**：基于预测模型的前馈控制，优化未来控制序列
- **多种优化算法**：
  - PSO（粒子群优化）- 推荐用于实时控制
  - SA（模拟退火）- 适合精确优化
  - DE（差分进化）- 平衡速度和精度
- **多池子支持**：同时控制4个池子的投矾量
- **约束处理**：
  - 投矾量上下限约束
  - 投矾变化率限制
  - 目标浊度跟踪
- **灵活配置**：可调整预测时域、控制时域、权重参数等

## 系统架构

```
optimizers/
├── config.py          # 配置类：MPCConfig, PoolState
├── optimizer.py       # 优化算法：PSO, SA, DE
├── controller.py      # MPC控制器核心
├── example_usage.py   # 使用示例
├── test_mpc.py        # 单元测试
└── README.md          # 本文档
```

## 快速开始

### 1. 环境要求

```bash
# Python >= 3.8
# 依赖包
pip install numpy torch scikit-learn
```

### 2. 基本使用

```python
import numpy as np
import torch
from config import MPCConfig, PoolState
from controller import create_mpc_controller

# 1. 配置MPC参数
config = MPCConfig(
    prediction_horizon=6,      # 预测时域: 6步 (30分钟)
    control_horizon=5,         # 控制时域: 5步
    target_turbidity=0.5,      # 目标浊度: 0.5 NTU
    lambda_du=0.1,             # 控制变化量权重
    optimizer_type='pso',      # 优化算法
    models_dir='../models',    # 模型目录
)

# 2. 创建控制器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
controller = create_mpc_controller(config, device)

# 3. 准备池子状态
# 历史数据：最近30步的数据 [30, 6]
# 特征顺序: [dose, turb_chushui, turb_jinshui, flow, pH, temp_shuimian]
history_data = ... # 从数据库或CSV读取

pool_state = PoolState(
    pool_id=1,
    current_dose=15.0,
    history_data=history_data
)

# 4. 执行控制
result = controller.control_step(pool_state, verbose=True)

# 5. 获取控制结果
next_dose = result['next_dose']              # 建议投矾量
predicted_turb = result['predicted_turbidity']  # 预测浊度序列
```

## MPC 原理

### 目标函数

MPC 控制器求解以下优化问题：

$$
\min \sum_{k=1}^{N_p} \|y(t+k) - y_{set}\|^2 + \lambda \sum_{k=1}^{N_c} \|\Delta u(t+k)\|^2
$$

其中：
- $N_p$: 预测时域（6步）
- $N_c$: 控制时域（5步）
- $y(t+k)$: 预测的出水浊度
- $y_{set}$: 目标浊度（0.5 NTU）
- $\Delta u(t+k)$: 控制变化量（投矾量变化）
- $\lambda$: 控制变化量权重系数

### 约束条件

1. **投矾量范围约束**：
   - $u_{min} \leq u(t+k) \leq u_{max}$
   - 默认：5-40 mg/L

2. **投矾变化率约束**：
   - $|\Delta u(t+k)| \leq \Delta u_{max}$
   - 默认：5 mg/L/步

## 配置参数说明

### MPCConfig 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `prediction_horizon` | int | 6 | 预测时域（步数） |
| `control_horizon` | int | 5 | 控制时域（步数） |
| `time_step` | float | 5.0 | 每步时间（分钟） |
| `target_turbidity` | float | 0.5 | 目标浊度（NTU） |
| `lambda_du` | float | 0.1 | 控制变化量权重 |
| `dose_min` | float | 5.0 | 投矾量下限（mg/L） |
| `dose_max` | float | 40.0 | 投矾量上限（mg/L） |
| `dose_rate_limit` | float | 5.0 | 投矾变化率限制（mg/L） |
| `optimizer_type` | str | 'pso' | 优化器类型 |
| `max_iterations` | int | 100 | 最大迭代次数 |
| `population_size` | int | 50 | 种群大小 |

### 优化器参数调优建议

#### PSO（粒子群优化）- 推荐
```python
config = MPCConfig(
    optimizer_type='pso',
    max_iterations=100,
    population_size=50,
    pso_w=0.7,    # 惯性权重
    pso_c1=1.5,   # 个体学习因子
    pso_c2=1.5,   # 社会学习因子
)
```

#### SA（模拟退火）
```python
config = MPCConfig(
    optimizer_type='sa',
    max_iterations=100,
    sa_temp_init=100.0,   # 初始温度
    sa_temp_min=1e-3,     # 最小温度
    sa_alpha=0.95,        # 降温系数
)
```

#### DE（差分进化）
```python
config = MPCConfig(
    optimizer_type='de',
    max_iterations=100,
    population_size=50,
    de_f=0.8,     # 缩放因子
    de_cr=0.9,    # 交叉概率
)
```
