# -*- encoding: utf-8 -*-
"""
投药优化定时调度服务。

职责：
    1. 周期性执行预测任务（data -> predict_only）
    2. 周期性执行优化任务（data -> pipeline.run）
    3. 保存最近一次预测/优化任务结果
    4. 提供调度器状态/启停/最新结果 API
"""

import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from flask import jsonify
import schedule

from .dosing_api import app, get_pipeline
from .io_adapter import read_data, upload_recommend_message
from utils.config_loader import load_config
from utils.logger import Logger

time_format = "%Y-%m-%d %H:%M:%S"
logger = Logger()

SCHEDULER_TAG_PREDICT = "alum_dosing_scheduler_predict"
SCHEDULER_TAG_OPTIMIZE = "alum_dosing_scheduler_optimize"

TASK_NAMES = ("predict", "optimize")
FALLBACK_MINUTE_BY_TASK = {
    "predict": 0,
    "optimize": 5,
}

# 调度器状态
state_lock = threading.Lock()
result_lock = threading.Lock()
scheduler_running = False
scheduler_thread: Optional[threading.Thread] = None
scheduler_settings: Dict[str, Any] = {}

# 任务结果缓存
latest_predict_result: Optional[Dict[str, Any]] = None
latest_optimize_result: Optional[Dict[str, Any]] = None


def _now_str() -> str:
    return datetime.now().strftime(time_format)


def _default_frequency(task_name: str) -> Dict[str, int]:
    return {"type": "hourly", "interval_hours": 1, "minute": FALLBACK_MINUTE_BY_TASK[task_name]}


def _default_scheduler_settings() -> Dict[str, Any]:
    return {
        "enabled": True,
        "auto_start": True,
        "tasks": {
            task_name: {
                "enabled": True,
                "frequency": _default_frequency(task_name),
            }
            for task_name in TASK_NAMES
        },
    }


def _sanitize_frequency(
    raw_frequency: Any,
    task_name: str,
    warn_prefix: str,
) -> Dict[str, int]:
    frequency = dict(_default_frequency(task_name))
    if not isinstance(raw_frequency, dict):
        logger.warning(f"{warn_prefix} frequency 配置非法，回退默认")
        return frequency

    frequency_type = str(raw_frequency.get("type", "hourly")).lower()
    if frequency_type != "hourly":
        logger.warning(f"{warn_prefix} 仅支持 hourly，回退默认")
        return frequency

    try:
        interval_hours = int(raw_frequency.get("interval_hours", frequency["interval_hours"]))
    except (TypeError, ValueError):
        interval_hours = frequency["interval_hours"]
    if interval_hours < 1:
        logger.warning(f"{warn_prefix} interval_hours 必须 >= 1，回退默认")
        interval_hours = frequency["interval_hours"]

    try:
        minute = int(raw_frequency.get("minute", frequency["minute"]))
    except (TypeError, ValueError):
        minute = frequency["minute"]
    if minute < 0 or minute > 59:
        logger.warning(f"{warn_prefix} minute 必须在 [0, 59]，回退默认")
        minute = frequency["minute"]

    return {"type": "hourly", "interval_hours": interval_hours, "minute": minute}


def _sanitize_task_settings(task_name: str, raw_task: Any) -> Dict[str, Any]:
    task_settings = {
        "enabled": True,
        "frequency": _default_frequency(task_name),
    }
    if not isinstance(raw_task, dict):
        logger.warning(f"[Scheduler:{task_name}] task 配置非法，使用默认")
        return task_settings

    task_settings["enabled"] = bool(raw_task.get("enabled", task_settings["enabled"]))
    task_settings["frequency"] = _sanitize_frequency(
        raw_task.get("frequency", task_settings["frequency"]),
        task_name,
        f"[Scheduler:{task_name}]",
    )
    return task_settings


def _sanitize_scheduler_settings(raw_scheduler: Any) -> Dict[str, Any]:
    settings = _default_scheduler_settings()
    if not isinstance(raw_scheduler, dict):
        logger.warning("[Scheduler] 配置类型非法，使用默认配置")
        return settings

    settings["enabled"] = bool(raw_scheduler.get("enabled", settings["enabled"]))
    settings["auto_start"] = bool(raw_scheduler.get("auto_start", settings["auto_start"]))

    raw_tasks = raw_scheduler.get("tasks")
    if not isinstance(raw_tasks, dict):
        logger.warning("[Scheduler] 缺少 tasks 配置，使用默认任务配置")
        return settings

    for task_name in TASK_NAMES:
        settings["tasks"][task_name] = _sanitize_task_settings(task_name, raw_tasks.get(task_name))

    return settings


def _refresh_scheduler_settings() -> Dict[str, Any]:
    global scheduler_settings
    config = load_config()
    raw_scheduler = config.get("scheduler", {})
    scheduler_settings = _sanitize_scheduler_settings(raw_scheduler)
    return scheduler_settings


def _register_task_job(tag: str, task_name: str, frequency: Dict[str, int], job_func) -> None:
    interval_hours = frequency["interval_hours"]
    minute = frequency["minute"]
    schedule.clear(tag)
    schedule.every(interval_hours).hours.at(f":{minute:02d}").do(job_func).tag(tag)
    logger.info(
        f"[调度器] 已注册任务: task={task_name} type=hourly interval_hours={interval_hours} minute={minute}"
    )


def _register_scheduler_jobs() -> None:
    schedule.clear(SCHEDULER_TAG_PREDICT)
    schedule.clear(SCHEDULER_TAG_OPTIMIZE)

    tasks = scheduler_settings["tasks"]
    if tasks["predict"]["enabled"]:
        _register_task_job(
            SCHEDULER_TAG_PREDICT, 
            "predict", 
            tasks["predict"]["frequency"], 
            scheduled_predict_job
        )

    if tasks["optimize"]["enabled"]:
        _register_task_job(
            SCHEDULER_TAG_OPTIMIZE,
            "optimize",
            tasks["optimize"]["frequency"],
            scheduled_optimization_job,
        )

    if not tasks["predict"]["enabled"] and not tasks["optimize"]["enabled"]:
        logger.warning("[调度器] 所有任务均被禁用")


def _get_next_run_at(tag: str) -> Optional[str]:
    jobs = schedule.get_jobs(tag)
    if not jobs:
        return None
    next_run = jobs[0].next_run
    if next_run is None:
        return None
    return next_run.strftime(time_format)


def _save_latest_result(task_name: str, payload: Dict[str, Any]) -> None:
    global latest_predict_result, latest_optimize_result
    with result_lock:
        if task_name == "predict":
            latest_predict_result = payload
        elif task_name == "optimize":
            latest_optimize_result = payload


def scheduled_predict_job():
    """
    执行一次预测定时任务（不走优化器）。
    """
    start_ts = time.time()
    logger.info(f"[定时任务] 开始执行预测任务 - {_now_str()}")

    try:
        pipeline = get_pipeline()
        config = pipeline.predictor_manager.config
        payload = read_data(config)
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise ValueError("read_data() 必须返回 (data_dict, last_dt)")

        data_dict, last_dt = payload
        if not isinstance(data_dict, dict) or not data_dict:
            raise ValueError("read_data() 返回的 data_dict 非法或为空")

        predictions = pipeline.predict_only(data_dict, last_dt)
        duration_ms = int((time.time() - start_ts) * 1000)
        pool_count = len(predictions) if isinstance(predictions, dict) else 0

        latest_payload = {
            "task": "predict",
            "status": "success",
            "executed_at": _now_str(),
            "duration_ms": duration_ms,
            "upload_skipped": True,
            "pool_count": pool_count,
            "result": predictions,
        }
        _save_latest_result("predict", latest_payload)
        logger.info(f"[定时任务] 预测任务成功: duration_ms={duration_ms}, pool_count={pool_count}")
    except Exception as exc:
        duration_ms = int((time.time() - start_ts) * 1000)
        error_traceback = traceback.format_exc()
        latest_payload = {
            "task": "predict",
            "status": "error",
            "executed_at": _now_str(),
            "duration_ms": duration_ms,
            "upload_skipped": True,
            "pool_count": 0,
            "error": str(exc),
            "traceback": error_traceback,
        }
        _save_latest_result("predict", latest_payload)
        logger.error(f"[定时任务] 预测任务失败: {exc}\n{error_traceback}")


def scheduled_optimization_job():
    """
    执行一次优化定时任务（数据 -> 预测器 -> 优化器）。
    """
    start_ts = time.time()
    logger.info(f"[定时任务] 开始执行优化任务 - {_now_str()}")

    try:
        pipeline = get_pipeline()
        config = pipeline.predictor_manager.config
        payload = read_data(config)
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise ValueError("read_data() 必须返回 (data_dict, last_dt)")

        data_dict, last_dt = payload
        if not isinstance(data_dict, dict) or not data_dict:
            raise ValueError("read_data() 返回的 data_dict 非法或为空")

        result = pipeline.run(data_dict, last_dt)
        duration_ms = int((time.time() - start_ts) * 1000)
        pool_count = len(result) if isinstance(result, dict) else 0

        upload_result = upload_recommend_message(result if isinstance(result, dict) else {})
        upload_skipped = bool(upload_result.get("skipped", True))

        latest_payload = {
            "task": "optimize",
            "status": "success",
            "executed_at": _now_str(),
            "duration_ms": duration_ms,
            "upload_skipped": upload_skipped,
            "pool_count": pool_count,
            "result": result,
        }
        _save_latest_result("optimize", latest_payload)
        logger.info(
            f"[定时任务] 优化任务成功: duration_ms={duration_ms}, pool_count={pool_count}, upload_skipped={upload_skipped}"
        )
    except Exception as exc:
        duration_ms = int((time.time() - start_ts) * 1000)
        error_traceback = traceback.format_exc()
        latest_payload = {
            "task": "optimize",
            "status": "error",
            "executed_at": _now_str(),
            "duration_ms": duration_ms,
            "upload_skipped": True,
            "pool_count": 0,
            "error": str(exc),
            "traceback": error_traceback,
        }
        _save_latest_result("optimize", latest_payload)
        logger.error(f"[定时任务] 优化任务失败: {exc}\n{error_traceback}")


def _run_scheduler_loop():
    while True:
        with state_lock:
            if not scheduler_running:
                break

        try:
            schedule.run_pending()
        except Exception:
            logger.error(f"[调度器] run_pending 异常:\n{traceback.format_exc()}")

        time.sleep(1)


def start_scheduler(interval_minutes: Optional[int] = None) -> Tuple[bool, str]:
    global scheduler_running, scheduler_thread

    if interval_minutes is not None:
        logger.warning("[调度器] interval_minutes 参数已废弃，当前由配置文件 scheduler 配置控制")

    with state_lock:
        if scheduler_running:
            return False, "already_running"

        settings = _refresh_scheduler_settings()
        if not settings.get("enabled", True):
            logger.warning("[调度器] 已禁用（scheduler.enabled=false）")
            return False, "disabled"

        _register_scheduler_jobs()

        scheduler_running = True
        scheduler_thread = threading.Thread(target=_run_scheduler_loop, daemon=True)
        scheduler_thread.start()

    logger.info("[调度器] 已启动")
    return True, "started"


def stop_scheduler() -> Tuple[bool, str]:
    global scheduler_running, scheduler_thread

    with state_lock:
        if not scheduler_running:
            schedule.clear(SCHEDULER_TAG_PREDICT)
            schedule.clear(SCHEDULER_TAG_OPTIMIZE)
            return True, "already_stopped"

        scheduler_running = False
        local_thread = scheduler_thread
        scheduler_thread = None
        schedule.clear(SCHEDULER_TAG_PREDICT)
        schedule.clear(SCHEDULER_TAG_OPTIMIZE)

    if local_thread is not None and local_thread.is_alive():
        local_thread.join(timeout=2)

    logger.info("[调度器] 已停止")
    return True, "stopped"


def run_flask_app_non_blocking(host: str = "0.0.0.0", port: int = 5002):
    def flask_thread_func():
        app.run(host=host, port=port, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=flask_thread_func, daemon=True)
    flask_thread.start()
    logger.info(f"[API] Flask服务线程已启动 @ {host}:{port}")
    return flask_thread


@app.route("/alum_dosing/scheduler/status", methods=["GET"])
def scheduler_status_api():
    with state_lock:
        running = scheduler_running
        tasks_cfg = scheduler_settings.get("tasks", {})

    status = {
        "scheduler_running": running,
        "timestamp": _now_str(),
    }

    with result_lock:
        predict_latest = latest_predict_result
        optimize_latest = latest_optimize_result

    status["predict"] = {
        "enabled": bool(tasks_cfg.get("predict", {}).get("enabled", True)),
        "next_run_at": _get_next_run_at(SCHEDULER_TAG_PREDICT),
        "has_latest_result": predict_latest is not None,
        "last_executed_at": predict_latest.get("executed_at") if predict_latest else None,
    }
    status["optimize"] = {
        "enabled": bool(tasks_cfg.get("optimize", {}).get("enabled", True)),
        "next_run_at": _get_next_run_at(SCHEDULER_TAG_OPTIMIZE),
        "has_latest_result": optimize_latest is not None,
        "last_executed_at": optimize_latest.get("executed_at") if optimize_latest else None,
    }

    return jsonify(status)


@app.route("/alum_dosing/scheduler/start", methods=["POST"])
def start_scheduler_api():
    success, reason = start_scheduler()
    if success:
        status = "success"
        message = "调度器已启动"
    elif reason == "already_running":
        status = "already_running"
        message = "调度器已在运行中"
    elif reason == "disabled":
        status = "disabled"
        message = "调度器已禁用（scheduler.enabled=false）"
    else:
        status = "error"
        message = "调度器启动失败"

    return jsonify({"status": status, "message": message, "timestamp": _now_str()})


@app.route("/alum_dosing/scheduler/stop", methods=["POST"])
def stop_scheduler_api():
    success, reason = stop_scheduler()
    if reason == "already_stopped":
        message = "调度器未运行（已是停止状态）"
    else:
        message = "调度器已停止"

    return jsonify(
        {
            "status": "success" if success else "error",
            "message": message,
            "timestamp": _now_str(),
        }
    )


@app.route("/alum_dosing/latest_result", methods=["GET"])
def get_latest_result_api():
    with result_lock:
        predict_latest = latest_predict_result
        optimize_latest = latest_optimize_result

    if predict_latest is None and optimize_latest is None:
        return jsonify(
            {
                "status": "no_result",
                "message": "暂无执行结果",
                "timestamp": _now_str(),
            }
        )

    return jsonify(
        {
            "status": "success",
            "result": {
                "predict": predict_latest,
                "optimize": optimize_latest,
            },
            "timestamp": _now_str(),
        }
    )


@app.route("/alum_dosing/latest_result/predict", methods=["GET"])
def get_latest_predict_result_api():
    with result_lock:
        result = latest_predict_result
    if result is None:
        return jsonify({"status": "no_result", "message": "暂无预测任务结果", "timestamp": _now_str()})
    return jsonify({"status": "success", "result": result, "timestamp": _now_str()})


@app.route("/alum_dosing/latest_result/optimize", methods=["GET"])
def get_latest_optimize_result_api():
    with result_lock:
        result = latest_optimize_result
    if result is None:
        return jsonify({"status": "no_result", "message": "暂无优化任务结果", "timestamp": _now_str()})
    return jsonify({"status": "success", "result": result, "timestamp": _now_str()})


def main():
    try:
        logger.info("=" * 60)
        logger.info("投药优化服务启动")
        logger.info("模式：触发式API + 定时任务")
        logger.info(f"启动时间：{_now_str()}")
        logger.info("=" * 60)

        settings = _refresh_scheduler_settings()
        if settings.get("enabled", True) and settings.get("auto_start", True):
            start_scheduler()
        else:
            logger.info("[调度器] 自动启动已关闭")

        flask_thread = run_flask_app_non_blocking()
        while True:
            if not flask_thread.is_alive():
                logger.error("Flask服务已停止")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止服务...")
    except Exception as exc:
        logger.error(f"服务启动失败: {exc}\n{traceback.format_exc()}")
    finally:
        stop_scheduler()


if __name__ == "__main__":
    main()
