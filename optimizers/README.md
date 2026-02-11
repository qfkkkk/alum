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

## 使用示例

### 示例1：单步控制

```bash
python example_usage.py 1
```

演示如何对单个池子执行一步MPC控制。

### 示例2：多步仿真

```bash
python example_usage.py 2
```

演示连续10步的MPC控制仿真，展示控制效果。

### 示例3：优化算法比较

```bash
python example_usage.py 3
```

比较PSO、SA、DE三种优化算法的性能。

### 示例4：多池子控制

```bash
python example_usage.py 4
```

演示同时控制4个池子的投矾量。

## 测试

运行单元测试验证功能：

```bash
# 运行所有测试
python test_mpc.py

# 运行单个测试
python test_mpc.py 1  # 测试配置类
python test_mpc.py 2  # 测试优化算法
python test_mpc.py 3  # 测试池子预测器
python test_mpc.py 4  # 测试MPC控制器
python test_mpc.py 5  # 测试多池子控制
```

## 实际应用指南

### 1. 数据准备

确保历史数据包含30个时间步（150分钟），6个特征：
- `dose`: 投矾量（mg/L）
- `turb_chushui`: 出水浊度（NTU）
- `turb_jinshui`: 进水浊度（NTU）
- `flow`: 流量（m³/h）
- `pH`: pH值
- `temp_shuimian`: 水面温度（℃）

### 2. 实时控制流程

```python
# 伪代码示例
while True:
    # 1. 从数据库获取最近30步数据
    history_data = fetch_history_from_db(pool_id, steps=30)
    
    # 2. 构造池子状态
    pool_state = PoolState(
        pool_id=pool_id,
        current_dose=current_dose,
        history_data=history_data
    )
    
    # 3. MPC控制计算
    result = controller.control_step(pool_state, verbose=False)
    next_dose = result['next_dose']
    
    # 4. 执行控制动作
    apply_dose_to_pool(pool_id, next_dose)
    
    # 5. 等待下一个控制周期（5分钟）
    time.sleep(5 * 60)
```

### 3. 参数调优建议

**快速响应场景**（进水浊度波动大）：
```python
config = MPCConfig(
    prediction_horizon=6,
    control_horizon=5,
    lambda_du=0.05,  # 降低控制惩罚，允许更大变化
)
```

**平稳控制场景**（进水浊度稳定）：
```python
config = MPCConfig(
    prediction_horizon=8,
    control_horizon=4,
    lambda_du=0.2,   # 增大控制惩罚，减少波动
)
```

## 性能优化

### GPU 加速

```python
# 使用GPU加速模型预测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
controller = create_mpc_controller(config, device)
```

### 减少优化时间

```python
config = MPCConfig(
    max_iterations=50,      # 减少迭代次数
    population_size=30,     # 减少种群大小
)
```

## 故障排除

### 问题1：模型文件找不到
```
FileNotFoundError: 模型文件不存在: ../models/pool_1/best_model.pt
```

**解决**：确保已训练模型，或修改`models_dir`路径。

### 问题2：预测值异常
```
预测浊度为负数或过大
```

**解决**：检查历史数据是否包含异常值，进行数据清洗。

### 问题3：优化时间过长
```
单次控制耗时超过30秒
```

**解决**：减少`max_iterations`和`population_size`，或使用GPU加速。

## API 参考

### MPCController

主要方法：

#### `control_step(pool_state, verbose=True) -> dict`

执行一步MPC控制。

**参数**：
- `pool_state`: PoolState对象，包含池子当前状态
- `verbose`: 是否打印详细信息

**返回**：
```python
{
    'next_dose': float,                    # 建议投矾量
    'dose_change': float,                  # 投矾变化量
    'optimal_cost': float,                 # 目标函数值
    'optimal_du_sequence': np.ndarray,     # 最优控制序列 [Nc]
    'optimal_dose_sequence': np.ndarray,   # 最优投矾序列 [Np]
    'predicted_turbidity': np.ndarray,     # 预测浊度 [Np]
}
```

### PoolPredictor

#### `predict(history_data, future_doses) -> np.ndarray`

预测未来出水浊度。

**参数**：
- `history_data`: 历史数据 [30, 6]
- `future_doses`: 未来投矾量 [6]

**返回**：
- 预测浊度序列 [6]

## 许可证

本项目仅供研究和学习使用。

## 联系方式

如有问题或建议，请联系开发团队。
