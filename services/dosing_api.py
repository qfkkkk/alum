# -*- encoding: utf-8 -*-
"""
投药优化 API 服务（触发式）。

路由：
    POST /alum_dosing/predict   - 仅预测
    POST /alum_dosing/optimize  - 全流程（数据 -> 预测器 -> 优化器），仅返回优化结果
    GET  /alum_dosing/health    - 健康检查
"""

import traceback
from datetime import datetime

from flask import Flask, jsonify

from .dosing_pipeline import DosingPipeline
from .io_adapter import read_data
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


@app.route("/alum_dosing/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "service": "alum_dosing",
            "timestamp": datetime.now().strftime(time_format),
        }
    )


@app.route("/alum_dosing/predict", methods=["POST"])
def predict_api():
    """
    触发式预测接口。
    """
    try:
        pipeline = get_pipeline()
        config = pipeline.predictor_manager.config

        data_dict, last_dt = read_data(config)
        predictions = pipeline.predict_only(data_dict, last_dt)

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

        return jsonify(
            {
                "status": "success",
                "predictions": formatted_preds,
                "count": count,
                "timestamp": last_dt.strftime(time_format),
            }
        )
    except Exception as exc:
        logger.error(f"预测接口异常: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/alum_dosing/optimize", methods=["POST"])
def optimize_api():
    """
    触发式全流程优化接口。

    注意：对外响应不返回 predictions 字段，仅返回优化结果 recommendations。
    """
    try:
        pipeline = get_pipeline()
        config = pipeline.predictor_manager.config

        data_dict, last_dt = read_data(config)
        results = pipeline.run(data_dict, last_dt)

        formatted_results = []
        for pool_id, res in results.items():
            pool_data = {
                "pool_id": pool_id,
                "status": res.get("status", "success"),
                "generated_at": res.get("generated_at"),
                "recommendations": [],
            }

            recs = res.get("recommendations", {})
            if recs:
                sorted_recs = sorted(recs.items(), key=lambda x: x[0])
                pool_data["recommendations"] = [
                    {"datetime": k, "value": round(float(v), 2)} for k, v in sorted_recs
                ]

            formatted_results.append(pool_data)

        return jsonify(
            {
                "status": "success",
                "results": formatted_results,
                "timestamp": last_dt.strftime(time_format),
            }
        )
    except Exception as exc:
        logger.error(f"优化接口异常: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(exc)}), 500


def run_flask_app(host: str = "0.0.0.0", port: int = 5002):
    """
    运行 Flask API 服务（阻塞模式）。
    """
    try:
        logger.info(f"启动投药优化API服务 @ {host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"Flask服务运行异常：\n{error_msg}")


if __name__ == "__main__":
    run_flask_app()
