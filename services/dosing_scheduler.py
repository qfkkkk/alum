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

import schedule

from .dosing_api import app, get_pipeline
from .io_adapter import read_data, upload_recommend_message
from .response import error_response, ok_response
from utils.config_loader import load_config
from utils.logger import Logger

time_format = "%Y-%m-%d %H:%M:%S"
logger = Logger()

SCHEDULER_TAG_PREDICT = "alum_dosing_scheduler_predict"
SCHEDULER_TAG_OPTIMIZE = "alum_dosing_scheduler_optimize"
DEFAULT_TASK_NAMES = ("predict", "optimize")
TASK_TAG_MAP = {
    "predict": SCHEDULER_TAG_PREDICT,
    "optimize": SCHEDULER_TAG_OPTIMIZE,
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


def _default_task_names() -> Tuple[str, ...]:
    return DEFAULT_TASK_NAMES


def _default_fallback_minute_by_task(task_names: Tuple[str, ...]) -> Dict[str, int]:
    minute_map = {}
    for task_name in task_names:
        if task_name == "predict":
            minute_map[task_name] = 0
        elif task_name == "optimize":
            minute_map[task_name] = 5
        else:
            minute_map[task_name] = 0
    return minute_map


def _default_frequency(default_minute: int) -> Dict[str, Any]:
    return {"type": "hourly", "interval_hours": 1, "minute": default_minute}


def _default_scheduler_settings(
    task_names: Tuple[str, ...], fallback_minute_by_task: Dict[str, int]
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "auto_start": True,
        "task_names": list(task_names),
        "fallback_minute_by_task": dict(fallback_minute_by_task),
        "tasks": {
            task_name: {
                "enabled": True,
                "frequency": _default_frequency(fallback_minute_by_task.get(task_name, 0)),
            }
            for task_name in task_names
        },
    }


def _sanitize_task_names(raw_task_names: Any) -> Tuple[str, ...]:
    if not isinstance(raw_task_names, list):
        logger.warning("[Scheduler] task_names 配置非法，使用默认任务列表")
        return _default_task_names()

    task_names = []
    for item in raw_task_names:
        if isinstance(item, str):
            name = item.strip()
            if name:
                task_names.append(name)

    if not task_names:
        logger.warning("[Scheduler] task_names 为空，使用默认任务列表")
        return _default_task_names()
    return tuple(task_names)


def _sanitize_fallback_minutes(
    raw_fallback: Any, task_names: Tuple[str, ...]
) -> Dict[str, int]:
    default_map = _default_fallback_minute_by_task(task_names)
    if not isinstance(raw_fallback, dict):
        logger.warning("[Scheduler] fallback_minute_by_task 配置非法，使用默认分钟回退")
        return default_map

    minute_map = dict(default_map)
    for task_name in task_names:
        try:
            minute = int(raw_fallback.get(task_name, minute_map[task_name]))
        except (TypeError, ValueError):
            minute = minute_map[task_name]
        if minute < 0 or minute > 59:
            logger.warning(f"[Scheduler:{task_name}] fallback minute 超出范围，使用默认值")
            minute = default_map[task_name]
        minute_map[task_name] = minute
    return minute_map


def _sanitize_frequency(
    raw_frequency: Any,
    warn_prefix: str,
    default_minute: int,
) -> Dict[str, Any]:
    frequency = dict(_default_frequency(default_minute))
    if not isinstance(raw_frequency, dict):
        logger.warning(f"{warn_prefix} frequency 配置非法，回退默认")
        return frequency

    frequency_type = str(raw_frequency.get("type", "hourly")).lower().strip()
    if frequency_type in ("seconds", "secondly"):
        try:
            interval_seconds = int(raw_frequency.get("interval_seconds", 10))
        except (TypeError, ValueError):
            interval_seconds = 10
        if interval_seconds < 1:
            logger.warning(f"{warn_prefix} interval_seconds 必须 >= 1，回退默认 10s")
            interval_seconds = 10
        return {"type": "seconds", "interval_seconds": interval_seconds}

    if frequency_type != "hourly":
        logger.warning(f"{warn_prefix} 仅支持 hourly/seconds，回退默认")
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

    minute_step = raw_frequency.get("minute_step")
    if minute_step is None:
        return {"type": "hourly", "interval_hours": interval_hours, "minute": minute}

    try:
        minute_step = int(minute_step)
    except (TypeError, ValueError):
        logger.warning(f"{warn_prefix} minute_step 非法，忽略该字段")
        return {"type": "hourly", "interval_hours": interval_hours, "minute": minute}

    if minute_step < 1 or minute_step > 59:
        logger.warning(f"{warn_prefix} minute_step 必须在 [1, 59]，忽略该字段")
        return {"type": "hourly", "interval_hours": interval_hours, "minute": minute}

    return {
        "type": "hourly",
        "interval_hours": interval_hours,
        "minute": minute,
        "minute_step": minute_step,
    }


def _sanitize_task_settings(
    task_name: str, raw_task: Any, fallback_minute_by_task: Dict[str, int]
) -> Dict[str, Any]:
    default_minute = fallback_minute_by_task.get(task_name, 0)
    task_settings = {
        "enabled": True,
        "frequency": _default_frequency(default_minute),
    }
    if not isinstance(raw_task, dict):
        logger.warning(f"[Scheduler:{task_name}] task 配置非法，使用默认")
        return task_settings

    task_settings["enabled"] = bool(raw_task.get("enabled", task_settings["enabled"]))
    task_settings["frequency"] = _sanitize_frequency(
        raw_task.get("frequency", task_settings["frequency"]),
        f"[Scheduler:{task_name}]",
        default_minute,
    )
    return task_settings


def _sanitize_scheduler_settings(raw_scheduler: Any) -> Dict[str, Any]:
    task_names = _default_task_names()
    fallback_minute_by_task = _default_fallback_minute_by_task(task_names)
    settings = _default_scheduler_settings(task_names, fallback_minute_by_task)
    if not isinstance(raw_scheduler, dict):
        logger.warning("[Scheduler] 配置类型非法，使用默认配置")
        return settings

    settings["enabled"] = bool(raw_scheduler.get("enabled", settings["enabled"]))
    settings["auto_start"] = bool(raw_scheduler.get("auto_start", settings["auto_start"]))
    task_names = _sanitize_task_names(raw_scheduler.get("task_names", list(task_names)))
    fallback_minute_by_task = _sanitize_fallback_minutes(
        raw_scheduler.get("fallback_minute_by_task", fallback_minute_by_task),
        task_names,
    )
    settings = _default_scheduler_settings(task_names, fallback_minute_by_task)
    settings["enabled"] = bool(raw_scheduler.get("enabled", settings["enabled"]))
    settings["auto_start"] = bool(raw_scheduler.get("auto_start", settings["auto_start"]))

    raw_tasks = raw_scheduler.get("tasks")
    if not isinstance(raw_tasks, dict):
        logger.warning("[Scheduler] 缺少 tasks 配置，使用默认任务配置")
        return settings

    for task_name in task_names:
        settings["tasks"][task_name] = _sanitize_task_settings(
            task_name, raw_tasks.get(task_name), fallback_minute_by_task
        )

    return settings


def _refresh_scheduler_settings() -> Dict[str, Any]:
    global scheduler_settings
    config = load_config()
    raw_scheduler = config.get("scheduler", {})
    scheduler_settings = _sanitize_scheduler_settings(raw_scheduler)
    return scheduler_settings


def _register_task_job(tag: str, task_name: str, frequency: Dict[str, Any], job_func) -> None:
    schedule.clear(tag)
    frequency_type = str(frequency.get("type", "hourly")).lower().strip()

    if frequency_type == "seconds":
        try:
            interval_seconds = int(frequency.get("interval_seconds", 10))
        except (TypeError, ValueError):
            interval_seconds = 10
        if interval_seconds < 1:
            interval_seconds = 10
        schedule.every(interval_seconds).seconds.do(job_func).tag(tag)
        logger.info(
            f"[调度器] 已注册任务: task={task_name} type=seconds interval_seconds={interval_seconds}"
        )
        return

    try:
        interval_hours = int(frequency.get("interval_hours", 1))
    except (TypeError, ValueError):
        interval_hours = 1
    try:
        minute = int(frequency.get("minute", 0))
    except (TypeError, ValueError):
        minute = 0
    if interval_hours < 1:
        interval_hours = 1
    if minute < 0 or minute > 59:
        minute = 0

    minute_step = frequency.get("minute_step")
    if minute_step is not None:
        try:
            minute_step = int(minute_step)
        except (TypeError, ValueError):
            minute_step = None

    if isinstance(minute_step, int) and 1 <= minute_step <= 59:
        minute_points = list(range(minute, 60, minute_step))
        if not minute_points:
            minute_points = [minute]
        for minute_point in minute_points:
            schedule.every(interval_hours).hours.at(f":{minute_point:02d}").do(job_func).tag(tag)
        logger.info(
            f"[调度器] 已注册任务: task={task_name} type=hourly interval_hours={interval_hours} "
            f"minute={minute} minute_step={minute_step} minute_points={minute_points}"
        )
        return

    schedule.every(interval_hours).hours.at(f":{minute:02d}").do(job_func).tag(tag)
    logger.info(
        f"[调度器] 已注册任务: task={task_name} type=hourly interval_hours={interval_hours} minute={minute}"
    )


def _register_scheduler_jobs() -> None:
    schedule.clear(SCHEDULER_TAG_PREDICT)
    schedule.clear(SCHEDULER_TAG_OPTIMIZE)

    tasks = scheduler_settings.get("tasks", {})
    task_names = scheduler_settings.get("task_names", list(DEFAULT_TASK_NAMES))
    job_func_by_task = {
        "predict": scheduled_predict_job,
        "optimize": scheduled_optimization_job,
    }

    enabled_count = 0
    for task_name in task_names:
        task_cfg = tasks.get(task_name, {})
        if not bool(task_cfg.get("enabled", True)):
            continue

        tag = TASK_TAG_MAP.get(task_name)
        job_func = job_func_by_task.get(task_name)
        if not tag or not job_func:
            logger.warning(f"[调度器] 未识别任务 {task_name}，跳过注册")
            continue

        _register_task_job(
            tag,
            task_name,
            task_cfg.get("frequency", _default_frequency(0)),
            job_func,
        )
        enabled_count += 1

    if enabled_count == 0:
        logger.warning("[调度器] 所有任务均被禁用或未识别")


def _get_next_run_at(tag: str) -> Optional[str]:
    jobs = schedule.get_jobs(tag)
    if not jobs:
        return None
    next_candidates = [job.next_run for job in jobs if job.next_run is not None]
    if not next_candidates:
        return None
    next_run = min(next_candidates)
    return next_run.strftime(time_format)


def _save_latest_result(task_name: str, payload: Dict[str, Any]) -> None:
    global latest_predict_result, latest_optimize_result
    with result_lock:
        if task_name == "predict":
            latest_predict_result = payload
        elif task_name == "optimize":
            latest_optimize_result = payload


def _get_latest_result_by_task(task_name: str) -> Optional[Dict[str, Any]]:
    if task_name == "predict":
        return latest_predict_result
    if task_name == "optimize":
        return latest_optimize_result
    return None


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
        upload_result = upload_recommend_message(
            predictions if isinstance(predictions, dict) else {},
            write_targets=("predict",),
        )
        upload_skipped = bool(upload_result.get("skipped", True))

        latest_payload = {
            "task": "predict",
            "status": "success",
            "executed_at": _now_str(),
            "duration_ms": duration_ms,
            "upload_skipped": upload_skipped,
            "pool_count": pool_count,
            "result": predictions,
        }
        _save_latest_result("predict", latest_payload)
        logger.info(
            f"[定时任务] 预测任务成功: duration_ms={duration_ms}, pool_count={pool_count}, upload_skipped={upload_skipped}"
        )
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

        upload_result = upload_recommend_message(
            result if isinstance(result, dict) else {},
            write_targets=("optimize",),
        )
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


def run_flask_app_non_blocking(host: str = "0.0.0.0", port: int = 5001):
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
        task_names = scheduler_settings.get("task_names", list(DEFAULT_TASK_NAMES))
        tasks_cfg = scheduler_settings.get("tasks", {})

    tasks_status = {}
    with result_lock:
        for task_name in task_names:
            latest = _get_latest_result_by_task(task_name)
            tag = TASK_TAG_MAP.get(task_name)
            tasks_status[task_name] = {
                "enabled": bool(tasks_cfg.get(task_name, {}).get("enabled", True)),
                "next_run_at": _get_next_run_at(tag) if tag else None,
                "has_latest_result": latest is not None,
                "last_executed_at": latest.get("executed_at") if latest else None,
            }

    return ok_response(
        data={
            "scheduler": {
                "running": running,
                "tasks": tasks_status,
            }
        }
    )


@app.route("/alum_dosing/scheduler/start", methods=["POST"])
def start_scheduler_api():
    success, reason = start_scheduler()
    if success:
        return ok_response(
            code="SCHEDULER_STARTED",
            message="调度器已启动",
            data={"scheduler_running": True, "state": "started"},
        )
    elif reason == "already_running":
        return ok_response(
            code="SCHEDULER_ALREADY_RUNNING",
            message="调度器已在运行中",
            data={"scheduler_running": True, "state": "already_running"},
        )
    elif reason == "disabled":
        return error_response(
            code="SCHEDULER_DISABLED",
            message="调度器已禁用（scheduler.enabled=false）",
            detail="scheduler.enabled=false",
            status_code=400,
        )
    return error_response(
        code="SCHEDULER_START_FAILED",
        message="调度器启动失败",
        detail=f"reason={reason}",
        status_code=500,
    )


@app.route("/alum_dosing/scheduler/stop", methods=["POST"])
def stop_scheduler_api():
    success, reason = stop_scheduler()
    if not success:
        return error_response(
            code="SCHEDULER_STOP_FAILED",
            message="调度器停止失败",
            detail=f"reason={reason}",
            status_code=500,
        )
    if reason == "already_stopped":
        return ok_response(
            code="SCHEDULER_ALREADY_STOPPED",
            message="调度器未运行（已是停止状态）",
            data={"scheduler_running": False, "state": "already_stopped"},
        )
    return ok_response(
        code="SCHEDULER_STOPPED",
        message="调度器已停止",
        data={"scheduler_running": False, "state": "stopped"},
    )


@app.route("/alum_dosing/latest_result", methods=["GET"])
def get_latest_result_api():
    with result_lock:
        predict_latest = latest_predict_result
        optimize_latest = latest_optimize_result

    if predict_latest is None and optimize_latest is None:
        return ok_response(
            code="NO_RESULT",
            message="暂无执行结果",
            data={"latest": None},
        )

    return ok_response(
        data={
            "latest": {
                "predict": predict_latest,
                "optimize": optimize_latest,
            }
        }
    )


@app.route("/alum_dosing/latest_result/predict", methods=["GET"])
def get_latest_predict_result_api():
    with result_lock:
        result = latest_predict_result
    if result is None:
        return ok_response(
            code="NO_RESULT",
            message="暂无预测任务结果",
            data={"task": "predict", "latest": None},
        )
    return ok_response(data={"task": "predict", "latest": result})


@app.route("/alum_dosing/latest_result/optimize", methods=["GET"])
def get_latest_optimize_result_api():
    with result_lock:
        result = latest_optimize_result
    if result is None:
        return ok_response(
            code="NO_RESULT",
            message="暂无优化任务结果",
            data={"task": "optimize", "latest": None},
        )
    return ok_response(data={"task": "optimize", "latest": result})


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
