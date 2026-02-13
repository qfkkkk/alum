"""
压力预测服务启动脚本
提供多种运行模式选择
"""
import sys
import os
import argparse
from pathlib import Path


from pressure_prediction_scheduler import start_scheduler, stop_scheduler
from utils.logger import Logger

logger = Logger()

def run_api_only():
    """仅运行API服务（不启动定时任务）"""
    import threading
    from pressure_prediction_scheduler import app
    
    logger.info("启动模式：仅API服务（无定时任务）")
    logger.info("API端点：http://0.0.0.0:5001/pump_station/pressure_prediction")
    
    def run_flask():
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    logger.info("API服务已启动，按 Ctrl+C 停止服务")
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("正在停止API服务...")

def run_scheduler_only():
    """仅运行定时任务（不启动API服务）"""
    logger.info("启动模式：仅定时任务")
    logger.info("定时任务：每小时执行一次压力预测")
    
    start_scheduler()
    
    logger.info("定时调度器已启动，按 Ctrl+C 停止调度器")
    
    try:
        while True:
            import time
            time.sleep(10)
    except KeyboardInterrupt:
        stop_scheduler()
        logger.info("定时调度器已停止")

def run_full_service():
    """运行完整服务（API + 定时任务）"""
    logger.info("启动模式：完整服务（API + 定时任务）")
    from pressure_prediction_scheduler import main as scheduler_main
    scheduler_main()

def show_status():
    """显示服务状态"""
    import requests
    try:
        response = requests.get('http://localhost:5001/pump_station/scheduler/status', timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("服务状态：")
            print(f"  调度器运行状态: {'运行中' if status['scheduler_running'] else '已停止'}")
            print(f"  最新预测结果: {'有' if status['has_latest_prediction'] else '无'}")
            if status['has_latest_prediction']:
                print(f"  预测结果形状: {status['prediction_shape']}")
                print(f"  最新时间戳: {status['latest_timestamp']}")
            print(f"  状态更新时间: {status['last_update']}")
        else:
            print("无法获取服务状态，服务可能未启动")
    except Exception as e:
        print(f"连接服务失败: {e}")
        print("请确保服务正在运行")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='压力预测服务管理')
    parser.add_argument('mode', choices=['api-only', 'scheduler-only', 'full', 'status'], 
                       nargs='?', default='full',
                       help='运行模式: api-only(仅API), scheduler-only(仅定时任务), full(完整服务), status(状态检查)')
    
    args = parser.parse_args()
    
    # 只在启动时打印一次标题
    if not hasattr(main, 'title_printed'):
        print("=" * 60)
        print("压力预测服务管理系统")
        print("=" * 60)
        main.title_printed = True
    
    if args.mode == 'api-only':
        run_api_only()
    elif args.mode == 'scheduler-only':
        run_scheduler_only()
    elif args.mode == 'full':
        run_full_service()
    elif args.mode == 'status':
        show_status()

if __name__ == "__main__":
    main()