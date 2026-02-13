"""
压力预测API服务与定时调度器
同时提供触发式API服务和定时预测功能
"""
import os
import joblib
import numpy as np
import pandas as pd
import torch
import warnings
import traceback
import threading
import time
from typing import Dict, List, Union, Optional
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request
from water_pressure_prediction import PressPredictor
from data.data_factory import load_agg_data,upload_recommend_message
from utils.logger import Logger
from utils.read_config import get_config
import schedule

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore")

# 全局变量
time_format = '%Y-%m-%d %H:%M:%S'
logger = Logger()
app = Flask(__name__)

# 定时任务相关变量
scheduler_running = False
scheduler_thread = None
latest_prediction = None
prediction_lock = threading.Lock()

def run_flask_app():
    """运行Flask API服务"""
    try:
        logger.info("\n\n\n启动压力预测API服务\n\n\n")
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"Flask服务运行异常：\n{error_msg}")

def run_flask_app_non_blocking():
    """在单独的线程中运行Flask API服务（非阻塞模式）"""
    def flask_thread_func():
        try:
            logger.info("启动Flask API服务线程")
            app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"Flask服务线程异常：{str(e)}")
    
    flask_thread = threading.Thread(target=flask_thread_func, daemon=True)
    flask_thread.start()
    logger.info("Flask API服务线程已启动")
    return flask_thread

def data_read():
    """读取数据点位"""
    try:
        config,rename_dict = get_config(type='real', model_name='pressure_predict')
        
        attributes = []
        for station_name, point_names in config.items():
            for point_name in point_names:
                attributes.append({
                    "AssetCode": station_name,
                    "AttributeCode": point_name
                })
        
        data1 = load_agg_data(attributes, minutes=60 * 24 * 30, interval="1H")
        
        column_names = ['datetime']
        for attribute in attributes:
            station_name = attribute['AssetCode']
            point_name = attribute['AttributeCode']
            column_names.append(f"{station_name}_{point_name}")
        
        if len(data1.columns) != len(column_names):
            logger.warning(f"数据列数({len(data1.columns)})与属性数({len(column_names)})不匹配")
        else:
            data1.columns = column_names
        data1.rename(columns = rename_dict,inplace=True)

        return data1

    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"数据读取异常：\n{error_msg}")
        return None

def scheduled_prediction_job():
    """定时预测任务"""
    global latest_prediction
    
    try:
        logger.info(f"[定时任务] 执行定时压力预测 - {datetime.now().strftime(time_format)}")
        
        # 读取数据
        data = data_read()
        if data is None or data.empty:
            logger.error("[定时任务] 数据读取失败，跳过本次预测")
            return
        
        logger.info(f"[定时任务] 成功读取数据，数据形状: {data.shape}")
        
        # 加载预测器
        path = Path(__file__)
        model_path = path.parent / 'model_save' / 'pytorch_pressure_model.pth'
        model_info_path = path.parent / 'model_save' / 'pytorch_pressure_model_info.pth'
        
        predictor = PressPredictor(
            model_path=model_path,
            info_path=model_info_path
        )

        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime',inplace=True)
        cols = ['SSBF_ZO_Press_P_102A_Press', 'SSBF_ZO_Press_P_102B_Press', 'SSBF_ZO_Press_P_102C_Press']
        data['pressure'] = data[cols].replace(0, np.nan).mean(axis=1)
        data['pressure'] = data['pressure'].fillna(0)
        # 执行预测
        predictions = predictor.predict(data=data)
        logger.info(f"[定时任务] 预测完成，结果形状: {predictions.shape}")
        logger.info(f"[定时任务] 预测结果预览: {predictions}")
        
        upload_recommend_message(
            {
                "WSPS_ALG_OUT":{
                    "Pressure_Forecast": '{}'.format(predictions.to_string())
            }
            }
        )

        # 保存最新预测结果
        with prediction_lock:
            latest_prediction = predictions
        
        # 保存预测结果到文件
        save_prediction_to_file(predictions)
        
        logger.info("[定时任务] 预测任务执行成功")
        
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"[定时任务] 执行失败: {str(e)}\n{error_msg}")

def save_prediction_to_file(predictions: pd.DataFrame):
    """保存预测结果到文件"""
    try:
        predictions_dir = Path(__file__).parent / 'pressure_predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = predictions_dir / f"pressure_prediction_{timestamp}.csv"
        
        predictions.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"[定时任务] 预测结果已保存到: {filename}")
        
    except Exception as e:
        logger.error(f"[定时任务] 保存预测结果失败: {str(e)}")

def start_scheduler():
    """启动定时调度器"""
    global scheduler_running, scheduler_thread
    
    if scheduler_running:
        logger.warning("[定时调度器] 调度器已在运行中")
        return
    
    try:
        # 配置定时任务
        schedule.every().hour.at(":10").do(scheduled_prediction_job)
        logger.info("[定时调度器] 配置定时任务：每小时执行一次压力预测")
        
        # 启动调度器线程
        scheduler_running = True
        scheduler_thread = threading.Thread(target=run_scheduler_loop, daemon=True)
        scheduler_thread.start()
        logger.info("[定时调度器] 定时调度器已启动")
        
    except Exception as e:
        logger.error(f"[定时调度器] 启动失败: {str(e)}")

def run_scheduler_loop():
    """调度器主循环"""
    global scheduler_running
    
    logger.info("[定时调度器] 开始调度循环")
    
    while scheduler_running:
        try:
            schedule.run_pending()
            time.sleep(10)
        except Exception as e:
            logger.error(f"[定时调度器] 调度循环异常: {str(e)}")
            time.sleep(30)

def stop_scheduler():
    """停止定时调度器"""
    global scheduler_running, scheduler_thread
    
    if not scheduler_running:
        logger.warning("[定时调度器] 调度器未在运行")
        return
    
    scheduler_running = False
    if scheduler_thread and scheduler_thread.is_alive():
        scheduler_thread.join(timeout=10)
    
    logger.info("[定时调度器] 定时调度器已停止")

@app.route('/pump_station/pressure_prediction', methods=['POST'])
def pressure_prediction_api():
    """压力预测API（触发式）"""
    try:
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({'error': '请求数据不能为空'}), 400

        required_fields = ['mode', 'data']
        missing_fields = [field for field in required_fields if field not in request_data]
        if missing_fields:
            return jsonify({'error': f'缺少必要参数: {", ".join(missing_fields)}'}), 400
        
        mode = request_data['mode']
        logger.info(f"[API] 收到预测请求，模式: {mode}")

        if mode == 'mulitsim':
            data = request_data['data']
            logger.info(f"API请求数据: mode: {mode}, data: {data}")

        if mode == 'agent':
            data = request_data['data']
            logger.info(f"API请求数据: mode: {mode}, data: {data}")

        if mode == 'online':
            logger.info(f"API请求数据: mode: {mode}")
            data = data_read()

        path = Path(__file__)
        model_path = path.parent / 'model_save' / 'pytorch_pressure_model.pth'
        model_info_path = path.parent / 'model_save' / 'pytorch_pressure_model_info.pth'
        
        predictor = PressPredictor(
            model_path=model_path,
            info_path=model_info_path
        )

        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime',inplace=True)
        cols = ['SSBF_ZO_Press_P_102A_Press', 'SSBF_ZO_Press_P_102B_Press', 'SSBF_ZO_Press_P_102C_Press']
        data['pressure'] = data[cols].replace(0, np.nan).mean(axis=1)
        data['pressure'] = data['pressure'].fillna(0)
        predictions = predictor.predict(data=data)
        logger.info(f"[API] 预测完成，返回结果")
        
        return jsonify({
            'status': 'success',
            'mode': mode,
            'timestamp': datetime.now().strftime(time_format),
            'data': predictions.to_json(orient="records", force_ascii=False)
        })
    
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"API请求异常：\n{error_msg}")
        return jsonify({'error': str(error_msg)}), 500

@app.route('/pump_station/scheduler/start', methods=['POST'])
def start_scheduler_api():
    """启动定时调度器API"""
    try:
        start_scheduler()
        return jsonify({
            'status': 'success',
            'message': '定时调度器已启动',
            'timestamp': datetime.now().strftime(time_format)
        })
    except Exception as e:
        return jsonify({'error': f'启动调度器失败: {str(e)}'}), 500


def main():
    """主函数：启动API服务和定时调度器"""
    try:
        logger.info("=" * 60)
        logger.info("压力预测服务启动")
        logger.info("服务模式：触发式API + 定时任务")
        logger.info(f"启动时间：{datetime.now().strftime(time_format)}")
        logger.info("=" * 60)
        
        # 启动定时调度器
        logger.info("正在启动定时调度器...")
        start_scheduler()
        
        # 启动Flask API服务（非阻塞模式）
        logger.info("正在启动API服务...")
        flask_thread = run_flask_app_non_blocking()
        
        # 主线程保持运行，等待中断信号
        try:
            while True:
                # 检查Flask线程是否还活着
                if not flask_thread.is_alive():
                    logger.error("Flask API服务线程已停止，准备退出")
                    break
                time.sleep(5)  # 每5秒检查一次
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止服务...")
            stop_scheduler()
            if flask_thread.is_alive():
                logger.info("等待Flask线程结束...")
                flask_thread.join(timeout=10)
            logger.info("服务已停止")
        
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        stop_scheduler()

if __name__ == "__main__":
    main()