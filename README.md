# Alum Dosing

投药预测与优化服务（Flask API + 定时调度）。

## 快速开始

### 1) 启动预测服务（默认 5001）

```bash
python -m services.run_predict_service full
```

### 2) 启动优化服务（默认 5002）

```bash
python -m services.run_optimize_service full
```

### 3) 健康检查

```bash
curl -X GET "http://127.0.0.1:5001/alum_dosing/health"
curl -X GET "http://127.0.0.1:5002/alum_dosing/health"
```

## 常用调用

### 预测（推荐走 5001）

```bash
curl -X POST "http://127.0.0.1:5001/alum_dosing/predict" \
  -H "Content-Type: application/json" \
  -d '{"mode":"online"}'
```

### 优化（推荐走 5002）

```bash
curl -X POST "http://127.0.0.1:5002/alum_dosing/optimize" \
  -H "Content-Type: application/json" \
  -d '{"mode":"online"}'
```

## 调度说明

- 调度频率由 `configs/app.yaml` 的 `scheduler` 配置控制。
- `mode=online` 的优化请求会走 `pipeline.run`（预测+特征提取+优化）。
- `mode=external` 的优化请求会直接走 `optimize_only`（不再做预测）。

## 文档导航

- API 对接文档：`docs/api.md`
- 部署与配置文档：`docs/deploy.md`

