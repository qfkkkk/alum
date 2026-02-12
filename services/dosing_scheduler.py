# -*- encoding: utf-8 -*-
"""
投药优化定时调度服务。

职责：
    1. 周期性执行 data_read -> pipeline.run
    2. 保存最近一次任务执行结果
    3. 提供调度器状态/启停/最新结果 API

说明：
    - upload_recommend_message 暂未实现，当前版本仅记录 upload_skipped 日志。
    - 调度频率从 configs/app.yaml 的 scheduler 段读取，非法值回退到默认值。
"""

import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from flask import jsonify

from .dosing_api import app, get_pipeline
from .io_adapter import read_data, upload_recommend_message
from utils.config_loader import load_config
from utils.logger import Logger

try:
    import schedule  # type: ignore
except ImportError:
    class _SimpleJob:
        def __init__(self, interval_hours: int, minute: int, job_func):
            self.interval_hours = interval_hours
            self.minute = minute
            self.job_func = job_func
            self.tags = set()
            self.next_run = self._compute_next_run(datetime.now())

        def _compute_next_run(self, now: datetime) -> datetime:
            candidate = now.replace(minute=self.minute, second=0, microsecond=0)
            if candidate <= now:
                candidate += timedelta(hours=1)

            if self.interval_hours > 1:
                mod = candidate.hour % self.interval_hours
                if mod != 0:
                    candidate += timedelta(hours=(self.interval_hours - mod))
            return candidate

        def should_run(self, now: datetime) -> bool:
            return now >= self.next_run

        def run(self):
            self.job_func()
            self.next_run = self.next_run + timedelta(hours=self.interval_hours)

        def tag(self, *tags):
            self.tags.update(tags)
            return self

    class _SimpleEvery:
        def __init__(self, scheduler, interval):
            self._scheduler = scheduler
            self._interval = interval
            self._minute = 0

        @property
        def hours(self):
            return self

        def at(self, minute_str: str):
            try:
                self._minute = int(str(minute_str).strip().replace(":", ""))
            except ValueError:
                self._minute = 0
            return self

        def do(self, job_func):
            job = _SimpleJob(self._interval, self._minute, job_func)
            self._scheduler.jobs.append(job)
            return job

    class _SimpleSchedule:
        def __init__(self):
            self.jobs = []

        def every(self, interval=1):
            return _SimpleEvery(self, interval)

        def clear(self, tag=None):
            if tag is None:
                self.jobs = []
            else:
                self.jobs = [job for job in self.jobs if tag not in job.tags]

        def get_jobs(self, tag=None):
            if tag is None:
                return list(self.jobs)
            return [job for job in self.jobs if tag in job.tags]

        def run_pending(self):
            now = datetime.now()
            for job in list(self.jobs):
                if job.should_run(now):
                    job.run()

    schedule = _SimpleSchedule()

# 全局配置
time_format = "%Y-%m-%d %H:%M:%S"
logger = Logger()

SCHEDULER_TAG = "alum_dosing_scheduler"

DEFAULT_SCHEDULER_SETTINGS = {
    "enabled": True,
    "auto_start": True,
    "frequency": {
        "type": "hourly",
        "interval_hours": 1,
        "minute": 5,
    },
}

# 调度器状态
state_lock = threading.Lock()
result_lock = threading.Lock()
scheduler_running = False
scheduler_thread: Optional[threading.Thread] = None
latest_result: Optional[Dict[str, Any]] = None
scheduler_settings: Dict[str, Any] = DEFAULT_SCHEDULER_SETTINGS.copy()


def _now_str() -> str:
    return datetime.now().strftime(time_format)


def _sanitize_scheduler_settings(raw_scheduler: Dict[str, Any]) -> Dict[str, Any]:
    """校验并规范 scheduler 配置，非法值回退默认值。"""
    settings = {
        "enabled": DEFAULT_SCHEDULER_SETTINGS["enabled"],
        "auto_start": DEFAULT_SCHEDULER_SETTINGS["auto_start"],
        "frequency": DEFAULT_SCHEDULER_SETTINGS["frequency"].copy(),
    }

    if not isinstance(raw_scheduler, dict):
        logger.warning("[Scheduler] 配置类型非法，使用默认配置")
        return settings

    settings["enabled"] = bool(raw_scheduler.get("enabled", settings["enabled"]))
    settings["auto_start"] = bool(raw_scheduler.get("auto_start", settings["auto_start"]))

    raw_frequency = raw_scheduler.get("frequency", {})
    if not isinstance(raw_frequency, dict):
        logger.warning("[Scheduler] frequency 配置非法，回退默认 1h@05")
        return settings

    frequency_type = str(raw_frequency.get("type", "hourly")).lower()
    if frequency_type != "hourly":
        logger.warning("[Scheduler] 仅支持 hourly，回退默认 1h@05")
        return settings

    try:
        interval_hours = int(raw_frequency.get("interval_hours", 1))
    except (TypeError, ValueError):
        interval_hours = 1
    if interval_hours < 1:
        logger.warning("[Scheduler] interval_hours 必须 >= 1，回退默认 1")
        interval_hours = 1

    try:
        minute = int(raw_frequency.get("minute", 5))
    except (TypeError, ValueError):
        minute = 5
    if minute < 0 or minute > 59:
        logger.warning("[Scheduler] minute 必须在 [0, 59]，回退默认 5")
        minute = 5

    settings["frequency"] = {
        "type": "hourly",
        "interval_hours": interval_hours,
        "minute": minute,
    }
    return settings


def _refresh_scheduler_settings() -> Dict[str, Any]:
    """从 app.yaml 读取 scheduler 配置并缓存。"""
    global scheduler_settings

    config = load_config()
    raw_scheduler = config.get("scheduler", {})
    scheduler_settings = _sanitize_scheduler_settings(raw_scheduler)
    return scheduler_settings


def _register_scheduler_job() -> None:
    """注册定时任务（按 tag 清理，避免误清理其他调度任务）。"""
    settings = scheduler_settings
    frequency = settings["frequency"]
    interval_hours = frequency["interval_hours"]
    minute = frequency["minute"]

    schedule.clear(SCHEDULER_TAG)
    schedule.every(interval_hours).hours.at(f":{minute:02d}").do(scheduled_optimization_job).tag(SCHEDULER_TAG)
    logger.info(
        f"[调度器] 已注册任务: type=hourly interval_hours={interval_hours} minute={minute}"
    )


def _get_next_run_at() -> Optional[str]:
    jobs = schedule.get_jobs(SCHEDULER_TAG)
    if not jobs:
        return None
    next_run = jobs[0].next_run
    if next_run is None:
        return None
    return next_run.strftime(time_format)


def _save_latest_result(payload: Dict[str, Any]) -> None:
    global latest_result
    with result_lock:
        latest_result = payload


def scheduled_optimization_job():
    """
    执行一次调度任务。

    流程：
        1. data_read() 读取输入
        2. pipeline.run(data_dict, last_dt)
        3. upload 步骤占位（仅日志）
        4. 更新 latest_result
    """
    start_ts = time.time()
    logger.info(f"[定时任务] 开始执行 - {_now_str()}")

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
            "status": "success",
            "executed_at": _now_str(),
            "duration_ms": duration_ms,
            "upload_skipped": upload_skipped,
            "pool_count": pool_count,
            "result": result,
        }
        _save_latest_result(latest_payload)

        logger.info(
            f"[定时任务] 执行成功: duration_ms={duration_ms}, pool_count={pool_count}, upload_skipped={upload_skipped}"
        )
    except Exception as exc:
        duration_ms = int((time.time() - start_ts) * 1000)
        error_traceback = traceback.format_exc()
        latest_payload = {
            "status": "error",
            "executed_at": _now_str(),
            "duration_ms": duration_ms,
            "upload_skipped": True,
            "pool_count": 0,
            "error": str(exc),
            "traceback": error_traceback,
        }
        _save_latest_result(latest_payload)
        logger.error(f"[定时任务] 执行失败: {exc}\n{error_traceback}")


def _run_scheduler_loop():
    """调度器后台循环。"""
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
    """启动调度器。返回 (success, reason)。"""
    global scheduler_running, scheduler_thread

    if interval_minutes is not None:
        logger.warning("[调度器] interval_minutes 参数已废弃，当前由配置文件 scheduler.frequency 控制")

    with state_lock:
        if scheduler_running:
            return False, "already_running"

        settings = _refresh_scheduler_settings()
        if not settings.get("enabled", True):
            logger.warning("[调度器] 已禁用（scheduler.enabled=false）")
            return False, "disabled"

        _register_scheduler_job()

        scheduler_running = True
        scheduler_thread = threading.Thread(target=_run_scheduler_loop, daemon=True)
        scheduler_thread.start()

    logger.info("[调度器] 已启动")
    return True, "started"


def stop_scheduler() -> Tuple[bool, str]:
    """停止调度器（幂等）。返回 (success, reason)。"""
    global scheduler_running, scheduler_thread

    with state_lock:
        if not scheduler_running:
            schedule.clear(SCHEDULER_TAG)
            return True, "already_stopped"

        scheduler_running = False
        local_thread = scheduler_thread
        scheduler_thread = None
        schedule.clear(SCHEDULER_TAG)

    if local_thread is not None and local_thread.is_alive():
        local_thread.join(timeout=2)

    logger.info("[调度器] 已停止")
    return True, "stopped"


def run_flask_app_non_blocking(host: str = "0.0.0.0", port: int = 5002):
    """在后台线程启动 Flask API。"""

    def flask_thread_func():
        app.run(host=host, port=port, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=flask_thread_func, daemon=True)
    flask_thread.start()
    logger.info(f"[API] Flask服务线程已启动 @ {host}:{port}")
    return flask_thread


@app.route("/alum_dosing/scheduler/status", methods=["GET"])
def scheduler_status_api():
    """获取调度器状态。"""
    with state_lock:
        running = scheduler_running

    status = {
        "scheduler_running": running,
        "timestamp": _now_str(),
        "next_run_at": _get_next_run_at(),
    }

    with result_lock:
        if latest_result is not None:
            status["has_latest_result"] = True
            status["last_executed_at"] = latest_result.get("executed_at")
        else:
            status["has_latest_result"] = False

    return jsonify(status)


@app.route("/alum_dosing/scheduler/start", methods=["POST"])
def start_scheduler_api():
    """启动调度器。"""
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
    """停止调度器（幂等）。"""
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
    """获取最近一次调度结果。"""
    with result_lock:
        if latest_result is None:
            return jsonify(
                {
                    "status": "no_result",
                    "message": "暂无执行结果",
                    "timestamp": _now_str(),
                }
            )
        result = latest_result

    return jsonify({"status": "success", "result": result, "timestamp": _now_str()})


def main():
    """启动调度服务（API + Scheduler）。"""
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
