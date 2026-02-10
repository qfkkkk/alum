# 投药优化模块 (alum_dosing)

## 功能概述

基于出水浊度预测进行最佳投药量优化。

- **输入**：30个时间点的历史数据
- **输出**：5个时间点的浊度预测 + 最优投药量推荐

## 目录结构

```
modules/alum_dosing/
├── __init__.py              # 模块入口
├── README.md                # 本文件
│
├── predictors/              # 预测层
│   ├── __init__.py          # 工厂函数 create_predictor()
│   ├── base_predictor.py    # 抽象基类
│   └── turbidity_predictor.py  # 浊度预测实现
│
├── optimizers/              # 优化层
│   ├── __init__.py          # 工厂函数 create_optimizer()
│   ├── base_optimizer.py    # 抽象基类
│   └── dosing_optimizer.py  # 投药优化实现
│
├── services/                # 服务层
│   ├── __init__.py
│   ├── dosing_pipeline.py   # 管道封装（预测→优化）
│   ├── dosing_api.py        # 纯API服务
│   └── dosing_scheduler.py  # API + 定时调度
│
├── utils/                   # 工具函数
│   ├── __init__.py
│   └── data_helper.py       # 数据处理辅助
│
└── models/                  # 模型文件（权重等）
    └── .gitkeep
```

## 设计模式

| 模式 | 位置 | 作用 |
|------|------|------|
| **策略模式** | `base_predictor.py`, `base_optimizer.py` | 算法可替换 |
| **工厂模式** | `create_predictor()`, `create_optimizer()` | 统一创建对象 |
| **管道模式** | `DosingPipeline` | 流程串联 |
| **单例模式** | `get_pipeline()` | 避免重复加载模型 |

## API路由

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
from modules.alum_dosing import DosingPipeline

pipeline = DosingPipeline()
result = pipeline.run(history_data, current_flow=1000)

print(f"最优投药量: {result['dosing_result']['optimal_dosing']} mg/L")
```

### 启动服务
```bash
# 仅API服务
python -m modules.alum_dosing.services.dosing_api

# API + 定时任务
python -m modules.alum_dosing.services.dosing_scheduler
```

## 复用的外部模块

- `data.data_factory.load_agg_data()` - 数据读取
- `data.data_factory.upload_recommend_message()` - 结果写回
- `utils.logger.Logger` - 日志
- `utils.read_config.get_config()` - 配置读取

## 扩展指南

### 添加新预测器
1. 继承 `BasePredictor`
2. 实现 `predict()` 和 `_load_model()`
3. 在 `predictors/__init__.py` 注册

### 添加新优化器
1. 继承 `BaseOptimizer`
2. 实现 `optimize()` 和 `get_top_n_solutions()`
3. 在 `optimizers/__init__.py` 注册
