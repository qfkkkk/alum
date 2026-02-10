# -*- encoding: utf-8 -*-
"""
投药优化定时调度服务
功能：同时提供API服务和=

特性：
    - 每小时自动执行一次完整优化流程
    - 自动将优化结果写回IoT平台
    - 同时提供API供手动触发
    - 支持调度器启停控制

路由列表（除dosing_api中的路由外，额外增加）：
    GET  /alum_dosing/scheduler/status  - 调度器状态
    POST /alum_dosing/scheduler/start   - 启动调度器
    POST /alum_dosing/scheduler/stop    - 停止调度器
    GET  /alum_dosing/latest_result     - 最新结果
"""
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import schedule
from flask import Flask, jsonify, request

from .dosing_pipeline import DosingPipeline
from .dosing_api import app, data_read, get_pipeline

# 复用data模块
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from data.data_factory import upload_recommend_message
from utils.logger import Logger

# 全局配置
time_format = '%Y-%m-%d %H:%M:%S'
logger = Logger()

# 调度器状态
scheduler_running = False
scheduler_thread = None
latest_result = None
result_lock = threading.Lock()


def scheduled_optimization_job():
    """
    定时优化任务
    
    功能：
        1. 读取最新数据
        2. 预测未来5个时间点的出水浊度
        3. 计算最优投药量
        4. 将结果写回IoT平台
        5. 保存结果到内存（供API查询）
    
    触发方式：
        由schedule库每小时自动调用
    """
    global latest_result
    
    try:
        logger.info(f"[定时任务] 执行投药优化 - {datetime.now().strftime(time_format)}")
        
        # 1. 读取数据
        data = data_read()
        if data is None or data.empty:
            logger.error("[定时任务] 数据读取失败")
            return
        
        # 2. 获取当前流量（从数据中提取最新值）
        # current_flow = data['flow'].iloc[-1]
        current_flow = 1000  # 占位
        
        # 3. 执行完整流程
        pipeline = get_pipeline()
        result = pipeline.run(data, current_flow=current_flow)
        
        # 4. 写回平台
        if result.get('status') == 'success':
            upload_recommend_message({
                "WSPS_ALG_OUT": {
                    "Optimal_Dosing": str(result['dosing_result']['optimal_dosing']),
                    "Turbidity_Forecast": str(result['turbidity_predictions'].to_dict())
                }
            })
            logger.info(f"[定时任务] 结果已写回平台")
        
        # 5. 保存到内存
        with result_lock:
            latest_result = {
                **result,
                'executed_at': datetime.now().strftime(time_format)
            }
        
        logger.info("[定时任务] 优化任务执行成功")
        
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"[定时任务] 执行失败: {str(e)}\n{error_msg}")


def start_scheduler(interval_minutes: int = 60):
    """
    启动定时调度器
    
    参数：
        interval_minutes: 执行间隔（分钟），默认60分钟
    
    说明：
        在后台线程中运行调度循环
    """
    global scheduler_running, scheduler_thread
    
    if scheduler_running:
        logger.warning("[调度器] 已在运行中")
        return False
    
    # 配置定时任务
    schedule.clear()
    schedule.every().hour.at(":05").do(scheduled_optimization_job)
    
    # 启动调度器线程
    scheduler_running = True
    scheduler_thread = threading.Thread(target=_run_scheduler_loop, daemon=True)
    scheduler_thread.start()
    
    logger.info(f"[调度器] 已启动，间隔: {interval_minutes}分钟")
    return True


def _run_scheduler_loop():
    """调度器主循环"""
    global scheduler_running
    while scheduler_running:
        schedule.run_pending()
        time.sleep(10)


def stop_scheduler():
    """停止定时调度器"""
    global scheduler_running
    scheduler_running = False
    schedule.clear()
    logger.info("[调度器] 已停止")
    return True


def run_flask_app_non_blocking(host: str = '0.0.0.0', port: int = 5002):
    """
    非阻塞模式启动Flask
    
    参数：
        host: 监听地址
        port: 监听端口
    
    返回：
        threading.Thread: Flask服务线程
    """
    def flask_thread_func():
        app.run(host=host, port=port, debug=False, use_reloader=False)
    
    flask_thread = threading.Thread(target=flask_thread_func, daemon=True)
    flask_thread.start()
    logger.info(f"[API] Flask服务线程已启动 @ {host}:{port}")
    return flask_thread


# ==================== 调度器管理API路由 ====================

@app.route('/alum_dosing/scheduler/status', methods=['GET'])
def scheduler_status_api():
    """
    获取调度器状态
    
    返回：
        {
            "scheduler_running": true/false,
            "has_latest_result": true/false,
            "last_executed_at": "...",
            "timestamp": "..."
        }
    """
    status = {
        'scheduler_running': scheduler_running,
        'timestamp': datetime.now().strftime(time_format)
    }
    
    with result_lock:
        if latest_result is not None:
            status['has_latest_result'] = True
            status['last_executed_at'] = latest_result.get('executed_at')
        else:
            status['has_latest_result'] = False
    
    return jsonify(status)


@app.route('/alum_dosing/scheduler/start', methods=['POST'])
def start_scheduler_api():
    """启动调度器"""
    success = start_scheduler()
    return jsonify({
        'status': 'success' if success else 'already_running',
        'message': '调度器已启动' if success else '调度器已在运行中',
        'timestamp': datetime.now().strftime(time_format)
    })


@app.route('/alum_dosing/scheduler/stop', methods=['POST'])
def stop_scheduler_api():
    """停止调度器"""
    success = stop_scheduler()
    return jsonify({
        'status': 'success',
        'message': '调度器已停止',
        'timestamp': datetime.now().strftime(time_format)
    })


@app.route('/alum_dosing/latest_result', methods=['GET'])
def get_latest_result_api():
    """
    获取最新优化结果
    
    返回：
        定时任务最近一次执行的完整结果
    """
    with result_lock:
        if latest_result is None:
            return jsonify({
                'status': 'no_result',
                'message': '暂无执行结果',
                'timestamp': datetime.now().strftime(time_format)
            })
        
        return jsonify({
            'status': 'success',
            'result': latest_result,
            'timestamp': datetime.now().strftime(time_format)
        })


# ==================== 主函数 ====================

def main():
    """
    主函数：同时启动API服务和定时调度器
    
    运行模式：
        1. 启动定时调度器（后台线程）
        2. 启动Flask API服务（后台线程）
        3. 主线程保持运行
    
    使用方式：
        python -m modules.alum_dosing.services.dosing_scheduler
    """
    try:
        logger.info("=" * 60)
        logger.info("投药优化服务启动")
        logger.info("模式：触发式API + 定时任务")
        logger.info(f"启动时间：{datetime.now().strftime(time_format)}")
        logger.info("=" * 60)
        
        # 启动调度器
        start_scheduler()
        
        # 启动API服务（非阻塞）
        flask_thread = run_flask_app_non_blocking()
        
        # 主线程保持运行
        try:
            while True:
                if not flask_thread.is_alive():
                    logger.error("Flask服务已停止")
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止服务...")
            stop_scheduler()
            
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        stop_scheduler()


if __name__ == "__main__":
    main()
