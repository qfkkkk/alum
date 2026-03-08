# 投矾系统服务 API 接口文档

---

## 1、浊度预测服务

**接口地址：**

```
http://<host>:5001/alum_dosing/predict
```

**访问方式：**

POST

**调用说明：**

服务端在每次调用时自动从实时数据源拉取最新的 60 步基线数据，调用方通过 `patches` 字段对指定时间步的指定特征进行**局部替换**，无需传入完整数据。

- `online` 模式：无需任何额外参数，直接使用实时数据进行预测。
- `external` / `agent` 模式：在实时基线数据之上，按 `patches` 内容进行局部替换后预测。

**输入参数：**

| 名称 | 含义 | 类型 | 必须 | 说明 |
|------|------|------|------|------|
| mode | 调用模式 | string | 是 | `online`：纯实时数据预测；`external` / `agent`：实时基线 + patches 替换 |
| patches | 局部替换数据 | dict | 否 | `external`/`agent` 模式下使用；键为池子 ID，值为 patch 列表；`online` 模式下忽略 |

**patches 单条结构：**

| 字段 | 含义 | 类型 | 必须 | 说明 |
|------|------|------|------|------|
| datetime | 替换时间点 | string | 是 | 格式 `YYYY-MM-DD HH:MM:SS`，必须精确命中当前数据窗口内的整分钟时间点 |
| features | 替换特征值 | dict | 是 | 键为特征名（大小写不敏感），值为数值；可同时替换多个特征 |

> **可用特征名：** `dose`（投药量）、`turb_chushui`（出水浊度）、`turb_jinshui`（进水浊度）、`flow`（进水流量）、`pH`（进水pH）、`temp_shuimian`（水温）

**输入数据示例（external / agent 模式）：**

```json
{
  "mode": "agent",
  "patches": {
    "pool_1": [
      {
        "datetime": "2026-03-07 20:05:00",
        "features": {"dose": 9.9, "pH": 7.2}
      },
      {
        "datetime": "2026-03-07 20:10:00",
        "features": {"flow": 1200}
      }
    ],
    "pool_2": [
      {
        "datetime": "2026-03-07 20:15:00",
        "features": {"dose": 10.5}
      }
    ]
  }
}
```

**输入数据示例（online 模式）：**

```json
{
  "mode": "online"
}
```

**返回参数：**

| 名称 | 含义 | 类型 | 说明 |
|------|------|------|------|
| success | 调用状态 | bool | `true` 表示成功，`false` 表示失败 |
| code | 返回编码 | string | 成功返回 `OK`，失败返回错误编码 |
| message | 返回信息 | string | 成功返回 `success`，失败返回错误原因 |
| data | 返回数据 | object | 预测结果，含各池子预测列表 |
| data.executed_at | 预测基准时间 | string | 本次预测所用数据的最后时间点 |
| data.pool_count | 池子数量 | int | 本次预测涵盖的池子总数 |
| data.point_count | 预测点数 | int | 所有池子预测时间点总数 |
| data.pools | 各池预测 | list | 各池子预测列表 |
| data.pools[].pool_id | 池子 ID | string | 如 `pool_1` |
| data.pools[].forecast | 预测序列 | list | 每条含 `datetime`（预测时间）和 `turbidity_pred`（预测浊度，NTU） |
| meta.timestamp | 响应时间 | string | 服务端返回时间 |

**返回示例：**

```json
{
  "success": true,
  "code": "OK",
  "message": "success",
  "data": {
    "task": "predict",
    "executed_at": "2026-03-07 20:15:00",
    "pool_count": 4,
    "point_count": 24,
    "pools": [
      {
        "pool_id": "pool_1",
        "forecast": [
          {"datetime": "2026-03-07 20:20:00", "turbidity_pred": 4.9839},
          {"datetime": "2026-03-07 20:25:00", "turbidity_pred": 3.4721},
          {"datetime": "2026-03-07 20:30:00", "turbidity_pred": 4.7588},
          {"datetime": "2026-03-07 20:35:00", "turbidity_pred": 4.1887},
          {"datetime": "2026-03-07 20:40:00", "turbidity_pred": 3.9993},
          {"datetime": "2026-03-07 20:45:00", "turbidity_pred": 4.4515}
        ]
      },
      {
        "pool_id": "pool_2",
        "forecast": [
          {"datetime": "2026-03-07 20:20:00", "turbidity_pred": 3.83},
          {"datetime": "2026-03-07 20:25:00", "turbidity_pred": 3.1036}
        ]
      }
    ]
  },
  "meta": {
    "timestamp": "2026-03-07 20:15:07"
  }
}
```

**错误码说明：**

| code | 含义 | HTTP 状态码 |
|------|------|-------------|
| `PREDICT_API_BAD_REQUEST` | 通用参数错误（如缺少 mode、patches 格式非法等） | 400 |
| `PREDICT_PATCH_DATETIME_FORMAT_INVALID` | patch datetime 格式非法，必须为 `YYYY-MM-DD HH:MM:SS` | 400 |
| `PREDICT_PATCH_DATETIME_OUT_OF_WINDOW` | patch datetime 不在当前数据窗口范围内 | 400 |
| `PREDICT_PATCH_POOL_NOT_FOUND` | patches 中指定的池子 ID 不存在 | 400 |
| `PREDICT_PATCH_FEATURE_NOT_FOUND` | patches 中指定的特征名不存在 | 400 |
| `PREDICT_PATCH_FEATURE_VALUE_INVALID` | patch 特征值无法转为数值 | 400 |
| `PREDICT_API_ERROR` | 服务端预测执行异常 | 500 |

---

## 2、投矾优化服务

**接口地址：**

```
http://<host>:5002/alum_dosing/optimize
```

**访问方式：**

POST

**输入参数：**

| 名称 | 含义 | 类型 | 必须 | 说明 |
|------|------|------|------|------|
| mode | 调用模式 | string | 是 | 有 `online`（实时数据）、`external`（外部数据）两种方式；外部调用使用 `external` |
| predictions | 浊度预测结果 | dict | 是 | 浊度预测服务返回的预测数据；键为池子 ID，值为以时间字符串为键、预测浊度值为数字的对象 |
| current_features | 当前实时特征 | dict | 是 | 各池子当前实时特征；键为池子 ID，值为特征字典（必须包含 `current_dose` 或其别名 `dose`）；若无输入，默认使用实时数据 |

**输入数据示例：**

```json
{
  "mode": "external",
  "predictions": {
    "pool_1": {
      "2026-02-13 12:05:00": 1.11,
      "2026-02-13 12:10:00": 1.15,
      "2026-02-13 12:15:00": 1.18,
      "2026-02-13 12:20:00": 1.20,
      "2026-02-13 12:25:00": 1.21,
      "2026-02-13 12:30:00": 1.22
    },
    "pool_2": {
      "2026-02-13 12:05:00": 1.11,
      "2026-02-13 12:10:00": 1.15,
      "2026-02-13 12:15:00": 1.18,
      "2026-02-13 12:20:00": 1.20,
      "2026-02-13 12:25:00": 1.21,
      "2026-02-13 12:30:00": 1.22
    },
    "pool_3": {
      "2026-02-13 12:05:00": 1.11,
      "2026-02-13 12:10:00": 1.15,
      "2026-02-13 12:15:00": 1.18,
      "2026-02-13 12:20:00": 1.20,
      "2026-02-13 12:25:00": 1.21,
      "2026-02-13 12:30:00": 1.22
    },
    "pool_4": {
      "2026-02-13 12:05:00": 1.11,
      "2026-02-13 12:10:00": 1.15,
      "2026-02-13 12:15:00": 1.18,
      "2026-02-13 12:20:00": 1.20,
      "2026-02-13 12:25:00": 1.21,
      "2026-02-13 12:30:00": 1.22
    }
  },
  "current_features": {
    "pool_1": {
      "current_dose": 10.0,
      "ph": 7.10,
      "flow": 1200.0
    },
    "pool_2": {
      "current_dose": 11.0,
      "ph": 7.05,
      "flow": 1180.0
    },
    "pool_3": {
      "current_dose": 11.0,
      "ph": 7.05,
      "flow": 1180.0
    },
    "pool_4": {
      "current_dose": 11.0,
      "ph": 7.05,
      "flow": 1180.0
    }
  }
}
```


**返回参数：**

| 名称 | 含义 | 类型 | 说明 |
|------|------|------|------|
| success | 调用状态 | string | `true` 表示成功，`false` 表示失败 |
| code | 返回编码 | string | 成功返回"ok",失败返回"错误编码" |
| message | 返回信息 | string | 成功返回 'success' ,失败返回'错误原因' |
| data | 返回数据 | list | 各池子投矾剂量推荐结果列表 |
| meta | 返回时间 | string | 示例 ："2026-02-13 12:00:00" |

**返回示例：**

```json
{
  "success": "true",
  "code": "OK",
  "message": "success",
  "data": [
    {
      "pool_id": "pool_1",
      "status": "success",
      "recommendations": [
        {"datetime": "2026-02-13 12:05:00", "value": 9.80},
        {"datetime": "2026-02-13 12:10:00", "value": 9.75},
        {"datetime": "2026-02-13 12:15:00", "value": 9.70},
        {"datetime": "2026-02-13 12:20:00", "value": 9.68}
      ]
    },
    {
      "pool_id": "pool_2",
      "status": "success",
      "recommendations": [
        {"datetime": "2026-02-13 12:05:00", "value": 10.50},
        {"datetime": "2026-02-13 12:10:00", "value": 10.45},
        {"datetime": "2026-02-13 12:15:00", "value": 10.40},
        {"datetime": "2026-02-13 12:20:00", "value": 10.38}
      ]
    }
  ],
  "meta": {
    "timestamp": "2026-02-13 12:00:00"
  }
}
```