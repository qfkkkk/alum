# -*- encoding: utf-8 -*-
"""
投药优化 API 服务（触发式）。

路由：
    GET  /alum_dosing/predict   - 仅预测（内部拉数）
    POST /alum_dosing/predict   - 仅预测（接收外部输入）
    GET  /alum_dosing/optimize  - 优化（全流程）
    POST /alum_dosing/optimize  - 优化（接收预测结果+特征）
    GET  /alum_dosing/health    - 健康检查
"""

import traceback
from datetime import datetime
from typing import Any, Dict

import numpy as np
from flask import Flask, request

from .dosing_pipeline import DosingPipeline
from .io_adapter import read_data
from .response import error_response, ok_response
from utils.logger import Logger

# 全局配置
time_format = "%Y-%m-%d %H:%M:%S"
logger = Logger()
app = Flask(__name__)

# 模块级单例（延迟初始化）
_pipeline = None


def get_pipeline() -> DosingPipeline:
    """
    获取管道单例，避免重复加载模型。
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = DosingPipeline()
    return _pipeline


def _parse_datetime(value: Any) -> datetime:
    """
    解析外部传入时间字段为 datetime。

    作用：
    - 统一处理 API 入参中的时间格式。
    - 减少业务逻辑里重复的时间解析代码。

    支持格式：
    - YYYY-mm-dd HH:MM:SS
    - YYYY-mm-ddTHH:MM:SS
    - YYYY-mm-dd HH:MM
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError("datetime 不能为空字符串")
        for fmt in (time_format, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
    raise ValueError("datetime 格式非法，支持 YYYY-mm-dd HH:MM:SS")


def _normalize_input_data(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    规范化 predict POST 的外部输入数据。

    作用：
    - 从 payload 的 data_dict/input_data/raw_data 中提取池子数据。
    - 将列表结构统一转换为 numpy 2D 数组，作为预测器标准输入。

    返回：
    - {pool_id: np.ndarray[seq_len, n_features]}
    """
    raw = payload.get("data_dict")
    if raw is None:
        raw = payload.get("input_data")
    if raw is None:
        raw = payload.get("raw_data")

    if not isinstance(raw, dict) or not raw:
        raise ValueError("POST 请求体缺少 data_dict/input_data/raw_data，或类型非法")

    data_dict = {}
    for pool_id, values in raw.items():
        try:
            arr = np.asarray(values, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"{pool_id} 数据无法转换为数值数组: {exc}") from exc
        if arr.ndim != 2:
            raise ValueError(f"{pool_id} 输入必须是二维数组，当前 ndim={arr.ndim}")
        data_dict[str(pool_id)] = arr
    return data_dict


def _normalize_predictions_from_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    规范化 optimize POST 的 predictions 字段。

    作用：
    - 兼容两类输入：
      1) 标准映射：{pool_id: {datetime: value}}
      2) predict 风格列表：pools[].forecast[]
    - 统一输出给 optimize_only 使用的预测结构。

    返回：
    - {pool_id: {datetime: float_value}}
    """
    predictions = payload.get("predictions")
    if predictions is None and isinstance(payload.get("data"), dict):
        data_section = payload["data"]
        predictions = data_section.get("predictions")
        if predictions is None and isinstance(data_section.get("pools"), list):
            predictions = data_section.get("pools")
    if predictions is None and isinstance(payload.get("pools"), list):
        predictions = payload.get("pools")

    if isinstance(predictions, dict):
        normalized = {}
        for pool_id, series in predictions.items():
            if not isinstance(series, dict):
                raise ValueError(f"{pool_id} 的 predictions 必须是对象映射")
            normalized[str(pool_id)] = {str(k): float(v) for k, v in series.items()}
        if not normalized:
            raise ValueError("predictions 不能为空")
        return normalized

    if isinstance(predictions, list):
        normalized = {}
        for pool_item in predictions:
            if not isinstance(pool_item, dict):
                continue
            pool_id = pool_item.get("pool_id")
            forecast = pool_item.get("forecast")
            if not pool_id or not isinstance(forecast, list):
                continue
            series = {}
            for item in forecast:
                if not isinstance(item, dict):
                    continue
                dt_str = item.get("datetime")
                if not dt_str:
                    continue
                if "turbidity_pred" in item:
                    value = item["turbidity_pred"]
                else:
                    value = item.get("value")
                if value is None:
                    continue
                series[str(dt_str)] = float(value)
            if series:
                normalized[str(pool_id)] = series
        if not normalized:
            raise ValueError("predictions 列表格式非法，必须包含 pools[].pool_id + forecast[]")
        return normalized

    raise ValueError("POST optimize 缺少 predictions（支持 dict 或 pools 列表格式）")


def _normalize_current_features(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    规范化 optimize POST 的当前特征字段。

    作用：
    - 提取 current_features/features。
    - 将数值字段统一为 float。
    - 兼容 dose -> current_dose，并强制校验 current_dose 必填。

    返回：
    - {pool_id: {feature_name: float_value}}
    """
    current_features = payload.get("current_features")
    if current_features is None:
        current_features = payload.get("features")
    if not isinstance(current_features, dict) or not current_features:
        raise ValueError("POST optimize 缺少 current_features/features")

    normalized = {}
    for pool_id, features in current_features.items():
        if not isinstance(features, dict):
            raise ValueError(f"{pool_id} 的 current_features 必须是对象")
        pool_feat = {}
        for key, value in features.items():
            pool_feat[str(key)] = float(value)
        if "current_dose" not in pool_feat and "dose" in pool_feat:
            pool_feat["current_dose"] = float(pool_feat["dose"])
        if "current_dose" not in pool_feat:
            raise ValueError(f"{pool_id} 缺少 current_dose（或 dose）")
        normalized[str(pool_id)] = pool_feat
    return normalized


def _infer_last_dt(predictions: Dict[str, Dict[str, float]]) -> datetime:
    """
    从预测时间戳推断本次 optimize 的执行时间。

    作用：
    - POST optimize 只传 predictions 时，没有显式 last_dt。
    - 这里从 predictions 中取可解析时间的最大值，作为 executed_at。
    """
    timestamps = []
    for series in predictions.values():
        for dt_str in series.keys():
            try:
                timestamps.append(_parse_datetime(dt_str))
            except ValueError:
                continue
    if not timestamps:
        return datetime.now()
    return max(timestamps)


def _format_predict_response(predictions: Dict[str, Dict[str, float]], last_dt: datetime):
    """
    将内部预测结果格式化为统一 API 响应结构。

    作用：
    - 输出对外约定的 pools[].forecast[] 结构。
    - 统一补充 task/pool_count/point_count/executed_at 等字段。
    """
    formatted_preds = []
    count = 0
    for pool_id, time_dict in predictions.items():
        pool_data = {"pool_id": pool_id, "forecast": []}
        sorted_items = sorted(time_dict.items(), key=lambda x: x[0])
        for dt_str, val in sorted_items:
            pool_data["forecast"].append(
                {"datetime": dt_str, "turbidity_pred": round(float(val), 4)}
            )

        formatted_preds.append(pool_data)
        count += len(time_dict)

    return ok_response(
        data={
            "task": "predict",
            "executed_at": last_dt.strftime(time_format),
            "pool_count": len(formatted_preds),
            "point_count": count,
            "pools": formatted_preds,
        }
    )


def _format_optimize_response(results: Dict[str, Dict[str, Any]], last_dt: datetime):
    """
    将内部优化结果格式化为统一 API 响应结构。

    作用：
    - 输出对外约定的 pools[].recommendations[] 结构。
    - 统一补充 task/pool_count/point_count/executed_at 等字段。
    """
    formatted_results = []
    point_count = 0
    for pool_id, res in results.items():
        pool_data = {
            "pool_id": pool_id,
            "status": res.get("status", "success"),
            "executed_at": last_dt.strftime(time_format),
            "recommendations": [],
        }

        recs = res.get("recommendations", {})
        if recs:
            sorted_recs = sorted(recs.items(), key=lambda x: x[0])
            pool_data["recommendations"] = [
                {"datetime": k, "value": round(float(v), 2)} for k, v in sorted_recs
            ]
            point_count += len(sorted_recs)

        formatted_results.append(pool_data)

    return ok_response(
        data={
            "task": "optimize",
            "executed_at": last_dt.strftime(time_format),
            "pool_count": len(formatted_results),
            "point_count": point_count,
            "pools": formatted_results,
        }
    )


@app.route("/alum_dosing/health", methods=["GET"])
def health_check():
    return ok_response(
        data={"service": "alum_dosing", "health": "healthy"},
        message="healthy",
    )


def _run_predict_for_request():
    """
    预测请求执行入口（供 GET/POST 路由复用）。

    行为：
    - GET: 走 read_data 拉实时数据，再调用 predict_only。
    - POST:
      - 有 payload: 用外部输入做预测；
      - 空 payload: 回退 read_data（兼容旧调用方）。

    错误处理：
    - 入参校验错误返回 400。
    - 其他异常返回 500。
    """
    try:
        pipeline = get_pipeline()
        config = pipeline.predictor_manager.config

        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            if not payload:
                data_dict, last_dt = read_data(config)
            else:
                data_dict = _normalize_input_data(payload)
                try:
                    last_dt = _parse_datetime(payload.get("last_dt"))
                except ValueError:
                    last_dt = datetime.now()
        else:
            data_dict, last_dt = read_data(config)

        predictions = pipeline.predict_only(data_dict, last_dt)
        return _format_predict_response(predictions, last_dt)
    except ValueError as exc:
        return error_response(
            code="PREDICT_API_BAD_REQUEST",
            message="预测请求参数非法",
            detail=str(exc),
            status_code=400,
        )
    except Exception as exc:
        logger.error(f"预测接口异常: {traceback.format_exc()}")
        return error_response(
            code="PREDICT_API_ERROR",
            message="预测接口执行失败",
            detail=str(exc),
            status_code=500,
        )


@app.route("/alum_dosing/predict", methods=["GET"])
def predict_get_api():
    return _run_predict_for_request()


@app.route("/alum_dosing/predict", methods=["POST"])
def predict_post_api():
    return _run_predict_for_request()


def _run_optimize_for_request():
    """
    优化请求执行入口（供 GET/POST 路由复用）。

    行为：
    - GET: 走 read_data + pipeline.run（预测+优化全流程）。
    - POST:
      - 有 payload: 解析 predictions/current_features，直接 optimize_only；
      - 空 payload: 回退 pipeline.run（兼容旧调用方）。

    错误处理：
    - 入参校验错误返回 400。
    - 其他异常返回 500。
    """
    try:
        pipeline = get_pipeline()
        config = pipeline.predictor_manager.config

        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            if not payload:
                data_dict, last_dt = read_data(config)
                results = pipeline.run(data_dict, last_dt)
            else:
                predictions = _normalize_predictions_from_payload(payload)
                current_features = _normalize_current_features(payload)
                recommendations = pipeline.optimize_only(
                    predictions,
                    current_features=current_features,
                )
                results = {
                    pool_id: {"status": "success", "recommendations": recs}
                    for pool_id, recs in recommendations.items()
                }
                last_dt = _infer_last_dt(predictions)
        else:
            data_dict, last_dt = read_data(config)
            results = pipeline.run(data_dict, last_dt)

        return _format_optimize_response(results, last_dt)
    except ValueError as exc:
        return error_response(
            code="OPTIMIZE_API_BAD_REQUEST",
            message="优化请求参数非法",
            detail=str(exc),
            status_code=400,
        )
    except Exception as exc:
        logger.error(f"优化接口异常: {traceback.format_exc()}")
        return error_response(
            code="OPTIMIZE_API_ERROR",
            message="优化接口执行失败",
            detail=str(exc),
            status_code=500,
        )


@app.route("/alum_dosing/optimize", methods=["GET"])
def optimize_get_api():
    return _run_optimize_for_request()


@app.route("/alum_dosing/optimize", methods=["POST"])
def optimize_post_api():
    return _run_optimize_for_request()


def run_flask_app(host: str = "0.0.0.0", port: int = 5001):
    """
    运行 Flask API 服务（阻塞模式）。

    作用：
    - 作为服务启动入口，供命令行脚本和直接执行使用。
    - 统一禁用 reloader，避免模型重复加载。
    """
    try:
        logger.info(f"启动投药优化API服务 @ {host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"Flask服务运行异常：\n{error_msg}")


if __name__ == "__main__":
    run_flask_app()
