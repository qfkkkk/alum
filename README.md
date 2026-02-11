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
├── checkpoints/               # 训练好的模型权重（.pt/.pkl，不入Git）
│   └── xPatch/                #   xPatch 各池权重
│       ├── pool_1/            #     1号池模型
│       ├── pool_2/            #     2号池模型
│       ├── pool_3/            #     3号池模型
│       └── pool_4/            #     4号池模型
│
├── predictors/                # 预测器（模型加载 + 推理流程）
│   ├── __init__.py            #   工厂函数 create_predictor()
│   ├── base_predictor.py      #   预测器抽象基类
│   └── turbidity_predictor.py #   浊度预测器实现
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
├── dataio/                      # 数据加载与获取
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
| **策略模式** | `base_predictor.py`, `base_optimizer.py` | 算法可替换 |
| **工厂模式** | `create_predictor()`, `create_optimizer()` | 统一创建对象 |
| **管道模式** | `DosingPipeline` | 流程串联 |
| **单例模式** | `get_pipeline()` | 避免重复加载模型 |

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

### 代码调用
```python
from alum import DosingPipeline

pipeline = DosingPipeline()
result = pipeline.run(history_data, current_flow=1000)

print(f"最优投药量: {result['dosing_result']['optimal_dosing']} mg/L")
```

### 启动服务
```bash
# 仅API服务
python -m alum.services.dosing_api

# API + 定时任务
python -m alum.services.dosing_scheduler
```

## 扩展指南

### 添加新模型架构
1. 在 `models/` 下新建子目录（如 `models/lstm/`）
2. 定义 `nn.Module` 子类
3. 训练后将权重放到 `checkpoints/<模型名>/` 下

### 添加新预测器
1. 继承 `BasePredictor`
2. 实现 `predict()` 和 `_load_model()`
3. 在 `predictors/__init__.py` 注册

### 添加新优化器
1. 继承 `BaseOptimizer`
2. 实现 `optimize()` 和 `get_top_n_solutions()`
3. 在 `optimizers/__init__.py` 注册
