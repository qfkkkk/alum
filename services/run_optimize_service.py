# -*- encoding: utf-8 -*-
"""
优化服务启动脚本（仅调度 optimize 任务）。

运行模式：
    - api-only: 仅启动 API
    - scheduler-only: 仅启动定时任务
    - full: 同时启动 API + 定时任务
    - status: 查看调度器状态
"""

import argparse
import time
import urllib.error
import urllib.request

from services.dosing_api import run_flask_app
from services.dosing_scheduler import (
    run_flask_app_non_blocking,
    set_scheduler_task_scope,
    start_scheduler,
    stop_scheduler,
)
from utils.logger import Logger

logger = Logger()

TASK_SCOPE = ("optimize",)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5002


def run_api_only(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    logger.info("启动模式：api-only（优化服务，仅API）")
    set_scheduler_task_scope(TASK_SCOPE)
    run_flask_app(host=host, port=port)


def run_scheduler_only():
    logger.info("启动模式：scheduler-only（优化服务，仅定时任务）")
    set_scheduler_task_scope(TASK_SCOPE)
    success, reason = start_scheduler(task_names_override=TASK_SCOPE)
    if not success and reason != "already_running":
        logger.error(f"调度器启动失败，reason={reason}")
        return

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止调度器...")
    finally:
        stop_scheduler()


def run_full_service(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    logger.info("启动模式：full（优化服务，API + 定时任务）")
    set_scheduler_task_scope(TASK_SCOPE)
    success, reason = start_scheduler(task_names_override=TASK_SCOPE)
    if not success and reason not in ("already_running", "disabled"):
        logger.error(f"调度器启动失败，reason={reason}")
        return
    if reason == "disabled":
        logger.warning("调度器已禁用，仅启动 API")

    flask_thread = run_flask_app_non_blocking(host=host, port=port)
    try:
        while True:
            if not flask_thread.is_alive():
                logger.error("Flask服务已停止")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止服务...")
    finally:
        stop_scheduler()


def show_status(port: int = DEFAULT_PORT):
    url = f"http://localhost:{port}/alum_dosing/scheduler/status"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = response.read().decode("utf-8")
            print(payload)
    except urllib.error.URLError as exc:
        print(f"获取状态失败: {exc}")


def main():
    parser = argparse.ArgumentParser(description="优化服务管理")
    parser.add_argument(
        "mode",
        choices=["api-only", "scheduler-only", "full", "status"],
        nargs="?",
        default="full",
        help="运行模式",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="API 监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="API 监听端口")
    args = parser.parse_args()

    if args.mode == "api-only":
        run_api_only(host=args.host, port=args.port)
    elif args.mode == "scheduler-only":
        run_scheduler_only()
    elif args.mode == "full":
        run_full_service(host=args.host, port=args.port)
    elif args.mode == "status":
        show_status(port=args.port)


if __name__ == "__main__":
    main()
