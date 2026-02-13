# API 对接文档

## Base URL

- 预测服务（推荐）：`http://<host>:5001`
- 优化服务（推荐）：`http://<host>:5002`

说明：
- 两个服务都包含完整 Flask 路由；推荐按职责区分端口调用。

## 统一响应格式

成功：

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

失败：

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

## 1. 健康检查

```bash
curl -X GET "http://127.0.0.1:5001/alum_dosing/health"
```

## 2. 预测接口

路由：`POST /alum_dosing/predict`  
必填：`mode`

- `mode=online`：内部读取实时数据
- `mode=external/agent/multisim/mulitsim`：使用外部输入数据

### online 示例

```bash
curl -X POST "http://127.0.0.1:5001/alum_dosing/predict" \
  -H "Content-Type: application/json" \
  -d '{"mode":"online"}'
```

### external 示例

```bash
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

## 3. 优化接口

路由：`POST /alum_dosing/optimize`  
必填：`mode`

- `mode=online`：执行 `pipeline.run`（预测+特征提取+优化）
- `mode=external/agent/multisim/mulitsim`：直接 `optimize_only`（需传 `predictions` + `current_features`）

### online 示例

```bash
curl -X POST "http://127.0.0.1:5002/alum_dosing/optimize" \
  -H "Content-Type: application/json" \
  -d '{"mode":"online"}'
```

### external 示例

```bash
curl -X POST "http://127.0.0.1:5002/alum_dosing/optimize" \
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

## 4. 调度器接口

### 状态

```bash
curl -X GET "http://127.0.0.1:5001/alum_dosing/scheduler/status"
```

### 启动

```bash
curl -X POST "http://127.0.0.1:5001/alum_dosing/scheduler/start"
```

### 停止

```bash
curl -X POST "http://127.0.0.1:5001/alum_dosing/scheduler/stop"
```

## 5. 最新结果接口

```bash
curl -X GET "http://127.0.0.1:5001/alum_dosing/latest_result"
```

