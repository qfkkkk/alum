# -*- encoding: utf-8 -*-
"""
投药优化 API 服务（触发式）。

路由：
    POST /alum_dosing/predict   - 预测（mode=online/external）
    POST /alum_dosing/optimize  - 优化（mode=online/external）
    GET  /alum_dosing/health    - 健康检查
"""

import traceback
from datetime import datetime, timedelta
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


class PredictPatchError(ValueError):
    def __init__(self, error_code: str, detail: str):
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail


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


def _parse_patch_datetime_strict(value: Any) -> datetime:
    """
    严格解析 patch datetime，必须为 YYYY-mm-dd HH:MM:SS。

    注意：
    - 不接受缺少秒的格式（如 YYYY-mm-dd HH:MM）。
    - 不做就近匹配，仅允许精确命中窗口时间点。
    """
    if not isinstance(value, str):
        raise PredictPatchError(
            "PREDICT_PATCH_DATETIME_FORMAT_INVALID",
            "patch datetime 必须是字符串，格式 YYYY-mm-dd HH:MM:SS",
        )

    raw = value.strip()
    if not raw:
        raise PredictPatchError(
            "PREDICT_PATCH_DATETIME_FORMAT_INVALID",
            "patch datetime 不能为空字符串",
        )

    try:
        parsed = datetime.strptime(raw, time_format)
    except ValueError as exc:
        raise PredictPatchError(
            "PREDICT_PATCH_DATETIME_FORMAT_INVALID",
            "patch datetime 格式非法，必须是 YYYY-mm-dd HH:MM:SS",
        ) from exc

    # 解析后再格式化校验，保证输入为严格秒级格式
    if parsed.strftime(time_format) != raw:
        raise PredictPatchError(
            "PREDICT_PATCH_DATETIME_FORMAT_INVALID",
            "patch datetime 格式非法，必须是 YYYY-mm-dd HH:MM:SS",
        )
    return parsed


def _build_feature_index(config: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    features = config.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("predict 配置缺少 features，无法执行 patches")

    by_exact = {}
    by_lower = {}
    for idx, name in enumerate(features):
        feature_name = str(name).strip()
        if not feature_name:
            continue
        by_exact[feature_name] = idx
        by_lower[feature_name.lower()] = idx

    if not by_exact:
        raise ValueError("predict 配置中的 features 非法，无法执行 patches")
    return {"exact": by_exact, "lower": by_lower}


def _build_window_index(last_dt: datetime, seq_len: int, interval_minutes: int) -> Dict[str, int]:
    if seq_len < 1:
        raise ValueError("predict 配置 seq_len 必须 >= 1")
    if interval_minutes < 1:
        raise ValueError("predict 配置 time_interval_minutes 必须 >= 1")

    start_dt = last_dt - timedelta(minutes=interval_minutes * (seq_len - 1))
    return {
        (start_dt + timedelta(minutes=interval_minutes * offset)).strftime(time_format): offset
        for offset in range(seq_len)
    }


def _align_datetime_to_interval(last_dt: datetime, interval_minutes: int) -> datetime:
    """
    将时间向下对齐到 interval_minutes 网格，秒固定为 00。

    示例：
    - interval=5, 19:45:57 -> 19:45:00
    - interval=5, 19:44:57 -> 19:40:00
    """
    if interval_minutes < 1:
        raise ValueError("predict 配置 time_interval_minutes 必须 >= 1")

    dt = last_dt.replace(microsecond=0)
    aligned_minute = (dt.minute // interval_minutes) * interval_minutes
    return dt.replace(minute=aligned_minute, second=0)


def _apply_predict_patches(
    baseline_data: Dict[str, Any],
    last_dt: datetime,
    payload: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[Dict[str, np.ndarray], datetime]:
    patches = payload.get("patches")
    if not isinstance(patches, dict) or not patches:
        raise ValueError("predict 非 online 模式缺少 patches，格式必须为 {pool_id: [patch, ...]}")

    feature_index = _build_feature_index(config)
    feature_exact = feature_index["exact"]
    feature_lower = feature_index["lower"]
    seq_len = int(config.get("seq_len", 60))
    interval_minutes = int(config.get("time_interval_minutes", 5))
    aligned_last_dt = _align_datetime_to_interval(last_dt, interval_minutes)
    window_index = _build_window_index(aligned_last_dt, seq_len, interval_minutes)
    window_start = (aligned_last_dt - timedelta(minutes=interval_minutes * (seq_len - 1))).strftime(time_format)
    window_end = aligned_last_dt.strftime(time_format)

    patched_data = {}
    for pool_id, values in (baseline_data or {}).items():
        try:
            arr = np.asarray(values, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"在线基线数据 {pool_id} 无法转换为数值数组: {exc}") from exc
        if arr.ndim != 2:
            raise ValueError(f"在线基线数据 {pool_id} 必须是二维数组，当前 ndim={arr.ndim}")
        if arr.shape[1] != len(feature_exact):
            raise ValueError(
                f"在线基线数据 {pool_id} 特征数={arr.shape[1]}，与配置 features={len(feature_exact)} 不一致"
            )
        patched_data[str(pool_id)] = arr.copy()

    if not patched_data:
        raise ValueError("read_data 返回空数据，无法执行 patches")

    for pool_id_raw, patch_list in patches.items():
        pool_id = str(pool_id_raw)
        if pool_id not in patched_data:
            raise PredictPatchError("PREDICT_PATCH_POOL_NOT_FOUND", f"patches 包含未知池子: {pool_id}")
        if not isinstance(patch_list, list) or not patch_list:
            raise ValueError(f"{pool_id} 的 patches 必须是非空列表")

        for idx, patch_item in enumerate(patch_list):
            if not isinstance(patch_item, dict):
                raise ValueError(f"{pool_id} patch[{idx}] 必须是对象")

            patch_dt = _parse_patch_datetime_strict(patch_item.get("datetime"))
            dt_key = patch_dt.strftime(time_format)
            row_idx = window_index.get(dt_key)
            if row_idx is None:
                raise PredictPatchError(
                    "PREDICT_PATCH_DATETIME_OUT_OF_WINDOW",
                    f"{pool_id} patch[{idx}] datetime={dt_key} 不在当前窗口范围 [{window_start}, {window_end}]"
                )

            feature_values = patch_item.get("features")
            if not isinstance(feature_values, dict) or not feature_values:
                raise ValueError(f"{pool_id} patch[{idx}] 缺少非空 features 对象")

            for feature_name_raw, raw_value in feature_values.items():
                feature_name = str(feature_name_raw).strip()
                if not feature_name:
                    raise ValueError(f"{pool_id} patch[{idx}] 存在空特征名")

                col_idx = feature_exact.get(feature_name)
                if col_idx is None:
                    col_idx = feature_lower.get(feature_name.lower())
                if col_idx is None:
                    raise PredictPatchError(
                        "PREDICT_PATCH_FEATURE_NOT_FOUND",
                        f"{pool_id} patch[{idx}] 特征 {feature_name} 不存在，可选: {list(feature_exact.keys())}"
                    )
                try:
                    patched_data[pool_id][row_idx, col_idx] = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise PredictPatchError(
                        "PREDICT_PATCH_FEATURE_VALUE_INVALID",
                        f"{pool_id} patch[{idx}] 特征 {feature_name} 的值无法转为数值: {raw_value}"
                    ) from exc

    return patched_data, aligned_last_dt


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
    预测请求执行入口（POST）。

    行为：
    - POST:
      - mode=online: 走 read_data 拉实时数据；
      - mode=external/agent/multisim/mulitsim: 先读实时基线，再按 patches 做局部替换；
      - mode 必填，缺失或非法返回 400。

    错误处理：
    - 入参校验错误返回 400。
    - 其他异常返回 500。
    """
    try:
        pipeline = get_pipeline()
        config = pipeline.predictor_manager.config

        payload = request.get_json(silent=True) or {}
        mode_raw = payload.get("mode")
        if mode_raw is None:
            raise ValueError("predict POST 缺少必填参数 mode")

        mode = str(mode_raw).strip().lower()
        if mode == "online":
            data_dict, last_dt = read_data(config)
        elif mode in ("external", "agent", "multisim", "mulitsim"):
            data_dict, last_dt = read_data(config)
            data_dict, last_dt = _apply_predict_patches(data_dict, last_dt, payload, config)
        else:
            raise ValueError(
                f"predict mode 不支持: {mode}，可选 online/external/agent/multisim"
            )

        predictions = pipeline.predict_only(data_dict, last_dt)
        return _format_predict_response(predictions, last_dt)
    except PredictPatchError as exc:
        return error_response(
            code=exc.error_code,
            message="预测 patch 参数非法",
            detail=exc.detail,
            status_code=400,
        )
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


@app.route("/alum_dosing/predict", methods=["POST"])
def predict_post_api():
    return _run_predict_for_request()


def _run_optimize_for_request():
    """
    优化请求执行入口（POST）。

    行为：
    - POST:
      - mode=online: 走 read_data + pipeline.run（预测+优化全流程）；
      - mode=external/agent/multisim/mulitsim: 解析 predictions/current_features，直接 optimize_only；
      - mode 必填，缺失或非法返回 400。

    错误处理：
    - 入参校验错误返回 400。
    - 其他异常返回 500。
    """
    try:
        pipeline = get_pipeline()
        config = pipeline.predictor_manager.config

        payload = request.get_json(silent=True) or {}
        mode_raw = payload.get("mode")
        if mode_raw is None:
            raise ValueError("optimize POST 缺少必填参数 mode")

        mode = str(mode_raw).strip().lower()
        if mode == "online":
            data_dict, last_dt = read_data(config)
            results = pipeline.run(data_dict, last_dt)
        elif mode in ("external", "agent", "multisim", "mulitsim"):
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
            raise ValueError(
                f"optimize mode 不支持: {mode}，可选 online/external/agent/multisim"
            )

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
