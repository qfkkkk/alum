# Alum Dosing 本地联调与远程对接说明

## 1. 本地测试需要修改的配置

文件：`configs/app.yaml`

### 1.1 数据 IO（本地假数据 + 本地打印）

```yaml
dataio:
  mode: local
  debug: false
  local_seed: 42
  read_model_name: optimized_dose
  write_model_name: optimized_dose
  remote_config_file: configs/alum_dosing.yaml
```

说明：
- `mode: local` 表示读数据用本地假数据，写结果只打印日志，不走远程库。

### 1.2 调度器（整点对齐每 5 分钟）

```yaml
scheduler:
  enabled: true
  auto_start: true
  task_names:
    - predict
    - optimize
  fallback_minute_by_task:
    predict: 5
    optimize: 5
  tasks:
    predict:
      enabled: true
      frequency:
        type: hourly
        interval_hours: 1
        minute: 5
        minute_step: 5
    optimize:
      enabled: true
      frequency:
        type: hourly
        interval_hours: 1
        minute: 5
        minute_step: 5
```

说明：
- 上述配置会在每小时的 `:05/:10/:15/.../:55` 触发。
- 该模式与服务启动时间无关，按整点分钟对齐。

## 2. 启动命令

项目根目录执行：

```bash
# 仅启动 API
python -m services.run_dosing_service api-only
```

```bash
# 仅启动定时任务
python -m services.run_dosing_service scheduler-only
```

```bash
# 同时启动 API + 定时任务（推荐联调）
python -m services.run_dosing_service full
```

```bash
# 查看调度器状态（需要服务已启动）
python -m services.run_dosing_service status
```

默认端口：`5001`

## 3. API 调用示例（curl）

### 3.1 预测器接口

```bash
curl -X POST "http://127.0.0.1:5001/alum_dosing/predict" \
  -H "Content-Type: application/json" \
  -d '{"mode":"online"}'
```

### 3.2 优化器接口

```bash
# POST：mode=online，内部读取实时数据并执行全流程
curl -X POST "http://127.0.0.1:5001/alum_dosing/optimize" \
  -H "Content-Type: application/json" \
  -d '{"mode":"online"}'
```

## 4. 常用状态接口

```bash
# 调度器状态
curl "http://127.0.0.1:5001/alum_dosing/scheduler/status"
```

```bash
# 最新任务结果（预测 + 优化）
curl "http://127.0.0.1:5001/alum_dosing/latest_result"
```

## 5. 快速排查

- `predict/optimize` 请求报错：先看服务是否启动、端口是否 `5001`。
- 定时任务没触发：确认 `scheduler.enabled=true`、任务 `enabled=true`、`frequency.type=hourly`、`minute=5`、`minute_step=5`。
- 本地模式没有写库：这是预期行为，`mode: local` 只打印结果日志。

## 6. 远程部署时需要修改哪些 YAML

文件：`configs/app.yaml`

把 `dataio` 改成远程模式，并配置远程写入模型名：

```yaml
dataio:
  mode: remote
  debug: false
  read_model_name: optimized_dose
  write_model_name_optimize: optimized_dose
  write_model_name_predict: effluent_turbidity
  remote_config_file: configs/alum_dosing.yaml
```

说明：
- `mode: remote`：读写都走远程数据源。
- `remote_config_file`：远程平台连接配置文件路径。
- `write_model_name_optimize`：优化结果写入的模型名。
- `write_model_name_predict`：预测结果写入的模型名。

调度频率按“每小时每 5 分钟一次（整点对齐）”示例（推荐）：

```yaml
scheduler:
  enabled: true
  auto_start: true
  task_names:
    - predict
    - optimize
  fallback_minute_by_task:
    predict: 5
    optimize: 5
  tasks:
    predict:
      enabled: true
      frequency:
        type: hourly
        interval_hours: 1
        minute: 5
        minute_step: 5
    optimize:
      enabled: true
      frequency:
        type: hourly
        interval_hours: 1
        minute: 5
        minute_step: 5
```

## 7. API 对接文档（请求 + 响应）

Base URL（示例）：
- `http://<host>:5001`

统一响应外层：

```json
{
  "success": true,
  "code": "OK",
  "message": "success",
  "data": {},
  "meta": {
    "timestamp": "2026-02-13 12:00:00"
  }
}
```

统一错误外层：

```json
{
  "success": false,
  "code": "XXX_ERROR",
  "message": "错误描述",
  "error": {
    "detail": "异常详情"
  },
  "meta": {
    "timestamp": "2026-02-13 12:00:00"
  }
}
```

### 7.1 健康检查

请求：

```bash
curl -X GET "http://127.0.0.1:5001/alum_dosing/health"
```

成功响应示例：

```json
{
  "success": true,
  "code": "OK",
  "message": "healthy",
  "data": {
    "service": "alum_dosing",
    "health": "healthy"
  },
  "meta": {
    "timestamp": "2026-02-13 12:00:00"
  }
}
```

### 7.2 预测接口

请求：
`mode` 为必填字段。

```bash
# POST：mode=external，接收外部输入数据后预测
curl -X POST "http://127.0.0.1:5001/alum_dosing/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "external",
    "last_dt": "2026-02-13 12:00:00",
    "data_dict": {
      "pool_1": [[1.0, 2.0], [3.0, 4.0]]
    }
  }'
```

```bash
# POST：mode=online 时，忽略外部 data，内部读取实时数据
curl -X POST "http://127.0.0.1:5001/alum_dosing/predict" \
  -H "Content-Type: application/json" \
  -d '{"mode":"online"}'
```

成功响应示例：

```json
{
  "success": true,
  "code": "OK",
  "message": "success",
  "data": {
    "task": "predict",
    "executed_at": "2026-02-13 12:00:00",
    "pool_count": 4,
    "point_count": 24,
    "pools": [
      {
        "pool_id": "pool_1",
        "forecast": [
          { "datetime": "2026-02-13 12:05:00", "turbidity_pred": 4.98 }
        ]
      }
    ]
  },
  "meta": {
    "timestamp": "2026-02-13 12:00:01"
  }
}
```

### 7.3 优化接口

请求：
`mode` 为必填字段。

```bash
# POST：mode=external，接收预测结果 + 当前特征，仅执行优化（不再做预测）
curl -X POST "http://127.0.0.1:5001/alum_dosing/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "external",
    "predictions": {
      "pool_1": {
        "2026-02-13 12:05:00": 1.11,
        "2026-02-13 12:10:00": 1.15
      }
    },
    "current_features": {
      "pool_1": {
        "current_dose": 10.0,
        "ph": 7.1,
        "flow": 1200
      }
    }
  }'
```

```bash
# POST：mode=online，内部读取实时数据并执行“预测+优化”全流程
curl -X POST "http://127.0.0.1:5001/alum_dosing/optimize" \
  -H "Content-Type: application/json" \
  -d '{"mode":"online"}'
```

成功响应示例：

```json
{
  "success": true,
  "code": "OK",
  "message": "success",
  "data": {
    "task": "optimize",
    "executed_at": "2026-02-13 12:00:00",
    "pool_count": 4,
    "point_count": 20,
    "pools": [
      {
        "pool_id": "pool_1",
        "status": "success",
        "executed_at": "2026-02-13 12:00:00",
        "recommendations": [
          { "datetime": "2026-02-13 12:05:00", "value": 7.86 }
        ]
      }
    ]
  },
  "meta": {
    "timestamp": "2026-02-13 12:00:01"
  }
}
```

### 7.4 调度器状态接口

请求：

```bash
curl -X GET "http://127.0.0.1:5001/alum_dosing/scheduler/status"
```

成功响应示例：

```json
{
  "success": true,
  "code": "OK",
  "message": "success",
  "data": {
    "scheduler": {
      "running": true,
      "tasks": {
        "predict": {
          "enabled": true,
          "next_run_at": "2026-02-13 12:00:10",
          "has_latest_result": true,
          "last_executed_at": "2026-02-13 12:00:00"
        },
        "optimize": {
          "enabled": true,
          "next_run_at": "2026-02-13 12:00:15",
          "has_latest_result": true,
          "last_executed_at": "2026-02-13 12:00:00"
        }
      }
    }
  },
  "meta": {
    "timestamp": "2026-02-13 12:00:02"
  }
}
```

### 7.5 调度器启停接口

启动请求：

```bash
curl -X POST "http://127.0.0.1:5001/alum_dosing/scheduler/start"
```

停止请求：

```bash
curl -X POST "http://127.0.0.1:5001/alum_dosing/scheduler/stop"
```

### 7.6 最新结果接口

请求：

```bash
curl -X GET "http://127.0.0.1:5001/alum_dosing/latest_result"
```

成功响应示例（节选）：

```json
{
  "success": true,
  "code": "OK",
  "message": "success",
  "data": {
    "latest": {
      "predict": {
        "task": "predict",
        "status": "success",
        "executed_at": "2026-02-13 12:00:00",
        "duration_ms": 20
      },
      "optimize": {
        "task": "optimize",
        "status": "success",
        "executed_at": "2026-02-13 12:00:00",
        "duration_ms": 150
      }
    }
  },
  "meta": {
    "timestamp": "2026-02-13 12:00:03"
  }
}
```
