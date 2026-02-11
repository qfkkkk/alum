
import unittest
import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# 添加 predictors 目录到 sys.path 以支持 predict.py 的绝对导入 (from dataset import ...)
sys.path.insert(0, os.path.join(project_root, 'predictors'))

from models.xPatch import predict as script_predict
from models.xPatch import dataset
from predictors.predictor_manager import TurbidityPredictorManager
from utils.config_loader import load_config

class TestConsistency(unittest.TestCase):
    
    def setUp(self):
        self.pool_id = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 路径设置
        self.test_data_path = os.path.join(project_root, 'checkpoints/xPatch/test_data/test_data.csv')
        self.checkpoint_base = os.path.join(project_root, 'checkpoints/xPatch')
        
        # 确保测试数据存在
        if not os.path.exists(self.test_data_path):
            self.skipTest(f"测试数据不存在: {self.test_data_path}")

    def test_prediction_consistency(self):
        """对比 predict.py 脚本与 Manager 封装后的预测结果一致性"""
        print("\n=== 一致性测试: Script vs Manager ===")
        
        # 1. 准备输入数据 (使用 dataset.py 的预处理逻辑)
        print(f"读取测试数据: {self.test_data_path}")
        df_pool, feature_names = dataset.load_and_preprocess(self.test_data_path, self.pool_id)
        data_values = df_pool.values.astype(np.float32)
        
        # 取最后 60 行
        seq_len = 60
        if len(data_values) < seq_len:
            self.fail(f"数据长度不足: {len(data_values)} < {seq_len}")
            
        input_data = data_values[-seq_len:]
        print(f"输入数据形状: {input_data.shape}")
        
        # ---------------------------------------------------------
        # 2. 方式 A: 使用 predict.py 原始逻辑
        # ---------------------------------------------------------
        print("运行 predict.py 原生逻辑...")
        # 注意: predict.load_model 期望 output_dir 是包含 pool_x 的父目录
        # 我们的结构是 checkpoints/xPatch/pool_1
        # 所以 output_dir 应该是 checkpoints/xPatch
        model, scaler, model_cfg, use_diff = script_predict.load_model(
            self.pool_id, 
            output_dir=self.checkpoint_base, 
            device=self.device
        )
        
        pred_script = script_predict.predict(
            model, scaler, model_cfg, use_diff, input_data, device=self.device
        )
        print(f"Script 预测结果: {pred_script}")
        
        # ---------------------------------------------------------
        # 3. 方式 B: 使用 TurbidityPredictorManager
        # ---------------------------------------------------------
        print("运行 TurbidityPredictorManager...")
        manager = TurbidityPredictorManager() # 自动加载 configs/app.yaml
        
        # 构造输入
        last_dt = datetime.now() # 时间戳不影响数值一致性对比
        
        # Manager 内部会调用 predict_single -> predict_with_timestamps -> predict
        # 我们直接调用内部 predictor 的 predict 方法来获取纯数值数组，避免时间戳解析误差
        if f'pool_{self.pool_id}' not in manager.predictors:
             self.skipTest(f"pool_{self.pool_id} 未在 Manager 中启用")
             
        predictor = manager.predictors[f'pool_{self.pool_id}']
        pred_manager = predictor.predict(input_data)
        print(f"Manager 预测结果: {pred_manager}")
        
        # ---------------------------------------------------------
        # 4. 对比
        # ---------------------------------------------------------
        try:
            np.testing.assert_allclose(pred_script, pred_manager, rtol=1e-5, atol=1e-6)
            print("✅ 结果完全一致")
        except AssertionError as e:
            print("❌ 结果不一致")
            diff = np.abs(pred_script - pred_manager)
            print(f"最大差异: {np.max(diff)}")
            raise e

if __name__ == '__main__':
    unittest.main()
