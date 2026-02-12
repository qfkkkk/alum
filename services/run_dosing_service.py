# -*- encoding: utf-8 -*-
"""
投药优化服务启动脚本。

运行模式：
    - api-only: 仅启动触发式 API
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
    main as run_full_service,
    start_scheduler,
    stop_scheduler,
)
from utils.logger import Logger

logger = Logger()


def run_api_only():
    logger.info("启动模式：api-only（仅触发式API）")
    run_flask_app()


def run_scheduler_only():
    logger.info("启动模式：scheduler-only（仅定时任务）")
    success, reason = start_scheduler()
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


def show_status():
    url = "http://localhost:5002/alum_dosing/scheduler/status"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = response.read().decode("utf-8")
            print(payload)
    except urllib.error.URLError as exc:
        print(f"获取状态失败: {exc}")


def main():
    parser = argparse.ArgumentParser(description="投药优化服务管理")
    parser.add_argument(
        "mode",
        choices=["api-only", "scheduler-only", "full", "status"],
        nargs="?",
        default="full",
        help="运行模式",
    )
    args = parser.parse_args()

    if args.mode == "api-only":
        run_api_only()
    elif args.mode == "scheduler-only":
        run_scheduler_only()
    elif args.mode == "full":
        logger.info("启动模式：full（API + 定时任务）")
        run_full_service()
    elif args.mode == "status":
        show_status()


if __name__ == "__main__":
    main()
