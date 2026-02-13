# 部署与配置说明

## 1. 本地联调配置

文件：`configs/app.yaml`

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
- `mode: local` 使用本地假数据读取，结果仅日志输出。

## 2. 远程部署配置

文件：`configs/app.yaml`

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
- `mode: remote` 使用远程读写。
- `write_model_name_optimize`/`write_model_name_predict` 分别对应优化和预测写入模型名。

## 3. 调度器配置（整点对齐）

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
- 触发点为每小时 `:05/:10/:15/.../:55`。
- 与服务启动时刻无关，按整点分钟对齐。

## 4. 启动命令

### 预测服务（5001）

```bash
python -m services.run_predict_service full
```

### 优化服务（5002）

```bash
python -m services.run_optimize_service full
```
