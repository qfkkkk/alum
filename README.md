# 投药优化模块 (Alum Dosing)

## 功能概述

基于出水浊度预测进行最佳投药量优化。

- **输入**：历史工况数据（进水浊度、流量、pH、温度等）
- **输出**：未来时间点的浊度预测 + 最优投药量推荐

## 目录结构

```
alum/
├── __init__.py                # 模块入口
├── README.md                  # 本文件
│
├── models/                    # 模型结构定义（nn.Module 等）
│   └── xPatch/                #   xPatch 模型架构
│       ├── layers/            #     网络层组件
│       └── models/            #     模型类定义
│
├── checkpoints/               # 训练好的模型权重（.pt/.pkl）
│   └── xPatch/                #   xPatch 各池权重
│       ├── pool_1/            #     1号池模型
│       ├── pool_2/            #     2号池模型
│       ├── pool_3/            #     3号池模型
│       └── pool_4/            #     4号池模型
│
├── predictors/                # 预测器（模型加载 + 推理流程）
│   ├── __init__.py            #   工厂函数 + 导出
│   ├── base_predictor.py      #   预测器抽象基类（模板方法）
│   ├── turbidity_predictor.py #   出水浊度预测器
│   └── predictor_manager.py   #   多池预测管理器
│
├── optimizers/                # 优化器（投药量优化算法）
│   ├── __init__.py            #   工厂函数 create_optimizer()
│   ├── base_optimizer.py      #   优化器抽象基类
│   └── dosing_optimizer.py    #   投药优化器实现
│
├── services/                  # 业务服务层
│   ├── __init__.py
│   ├── dosing_pipeline.py     #   管道封装（预测→优化）
│   ├── dosing_api.py          #   纯API服务
│   └── dosing_scheduler.py    #   API + 定时调度
│
├── configs/                   # 配置文件
│   └── app.yaml               #   应用配置（模型+优化+服务）
│
├── dataio/                    # 数据加载与获取
│   ├── __init__.py
│   ├── data_factory.py        #   数据工厂（读取、上传）
│   └── data_loader.py         #   平台数据接口
│
├── utils/                     # 通用工具函数
│   ├── __init__.py
│   └── data_helper.py         #   数据处理辅助
│
└── tests/                     # 单元测试
```

### 核心分层

| 层级 | 目录 | 职责 |
|------|------|------|
| **配置** | `configs/` | 应用配置（模型架构、步长、特征、池子开关、优化参数、路由） |
| **模型定义** | `models/` | 网络结构定义（`nn.Module` 子类），纯代码，不含权重 |
| **模型权重** | `checkpoints/` | 训练产出的 `.pt`/`.pkl` 文件，通过 `.gitignore` 排除 |
| **预测器** | `predictors/` | 从 `models/` 导入结构 → 从 `checkpoints/` 加载权重 → 执行推理 |
| **优化器** | `optimizers/` | 基于预测结果计算最优投药方案 |
| **服务层** | `services/` | 组合预测器和优化器，对外提供 API |

### 依赖关系

```
models/  ←  底层，无依赖
  ↑
predictors/ / optimizers/  ←  导入 models 中的类
  ↑
services/  ←  组合 predictors 和 optimizers
```

## 设计模式

| 模式 | 位置 | 作用 |
|------|------|------|
| **策略模式** | `BasePredictor`, `BaseOptimizer` | 不同算法（xPatch/LSTM等）可互换 |
| **模板方法** | `BasePredictor.predict()` | 通用流程骨架，子类只实现 `_infer()` |
| **工厂模式** | `create_predictor()`, `create_manager()` | 统一创建对象 |
| **管道模式** | `DosingPipeline` | 流程串联 |

## API 路由

```
POST /alum_dosing/full_optimization    # 主接口：预测+优化
POST /alum_dosing/predict_turbidity    # 仅预测浊度
POST /alum_dosing/optimize_dosing      # 仅优化投药
GET  /alum_dosing/health               # 健康检查

GET  /alum_dosing/scheduler/status     # 调度器状态
POST /alum_dosing/scheduler/start      # 启动调度器
POST /alum_dosing/scheduler/stop       # 停止调度器
GET  /alum_dosing/latest_result        # 最新结果
```

## 快速使用

### 预测（多池）
```python
from predictors import create_manager

manager = create_manager()  # 自动读取 configs/app.yaml
result = manager.predict_all(data_dict, last_datetime)
# result = {"pool_1": {"2026-02-11 10:05": 0.32, ...}, ...}
```

### 预测（单池）
```python
from predictors import create_predictor
import yaml

with open('configs/app.yaml') as f:
    config = yaml.safe_load(f)

predictor = create_predictor('xPatch', pool_id=1, config=config)
predictions = predictor.predict(input_data)  # [60, 6] -> [6]
```

### 投药优化管道 (Pipeline)
```python
from services.dosing_pipeline import DosingPipeline
from datetime import datetime

# 初始化
pipeline = DosingPipeline()  # 自动加载配置

# 执行全流程 (数据适配 -> 预测 -> 特征提取 -> 优化)
result = pipeline.run(
    raw_data=input_data_dict, 
    last_dt=datetime(2026, 2, 11, 12, 0)
)

# 单独执行优化 (调试用)
# predictions: {pool_id: {time: val}}
recommendations = pipeline.optimize_only(predictions)

# 单独执行预测 (调试用)
predictions = pipeline.predict_only(
    input_data=input_data_dict,
    last_dt=datetime(2026, 2, 11, 12, 0)
)
```

### 启动服务
```bash
# 仅API服务
python -m alum.services.dosing_api

# API + 定时任务
python -m alum.services.dosing_scheduler
```
