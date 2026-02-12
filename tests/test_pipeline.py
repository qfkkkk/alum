
import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.dosing_pipeline import DosingPipeline
from utils.config_loader import load_config

class TestDosingPipeline(unittest.TestCase):
    
    def setUp(self):
        self.last_dt = datetime(2026, 2, 11, 12, 0, 0)
        self.config = load_config()
        self.seq_len = self.config.get('seq_len', 60)
        self.n_features = len(self.config.get('features', []))
        
        # 构造随机输入数据
        self.input_data = np.random.rand(self.seq_len, self.n_features).astype(np.float32)

    def test_pipeline_run(self):
        """测试完整管道流程: 预测(xPatch) -> 优化(PSO)"""
        print("\n=== 测试投药优化管道 (Integration) ===")
        
        # 初始化管道
        # 使用默认配置 (加载 app.yaml, 创建 xPatch 预测器, 创建 PSO 优化器)
        pipeline = DosingPipeline(optimizer_type='pso')
        
        # 检查是否启用了池子
        enabled_pools = pipeline.predictor_manager.enabled_pools
        if not enabled_pools:
            print("没有启用的池子，跳过测试")
            return
            
        print(f"启用池子: {enabled_pools}")
        
        # 构造输入数据
        data_dict = {
            pool_name: self.input_data 
            for pool_name in enabled_pools
        }
        
        # 执行管道
        try:
            results = pipeline.run(
                data_dict, 
                self.last_dt
            )
            print(results)
            # 验证结果结构
            for pool_name, res in results.items():
                print(f"\n检查 {pool_name}:")
                
                # 验证预测结果 (6个点)
                preds = res['predictions']
                self.assertEqual(len(preds), 6)
                
                # 验证推荐结果 (5个点)
                recs = res['recommendations']
                self.assertEqual(len(recs), 5)
                
                # 验证时间戳匹配
                # 推荐的时间戳应该是预测的前5个
                pred_times = sorted(preds.keys())
                rec_times = sorted(recs.keys())
                
                self.assertEqual(rec_times, pred_times[:5])
                
                # 验证返回值类型
                first_rec = recs[rec_times[0]]
                print(f"  首个推荐投药量: {first_rec}")
                self.assertIsInstance(first_rec, float)
            
            print("管道 Run 流程验证通过！")
            # --- 新增测试: 单独调用 predict_only ---
            print("\n=== 测试 predict_only ===")
            preds = pipeline.predict_only(data_dict, self.last_dt)
            self.assertEqual(len(preds), len(enabled_pools))
            print(f"预测只有 {len(preds)} 个池子 - OK")
            
            pool_1_preds = preds['pool_1']
            self.assertEqual(len(pool_1_preds), 6)
            print("Pool 1 预测 6 个点 - OK")
            
            # --- 新增测试: 单独调用 optimize_only ---
            print("\n=== 测试 optimize_only ===")
            
            # 手动构造特征数据传给 optimize_only
            # (在 run() 模式下这是自动提取的，这里我们需要手动构造以测试接口)
            mock_features = {}
            for idx, pool_name in enumerate(enabled_pools):
                mock_features[pool_name] = {
                    'current_dose': float(15 + idx),
                    'ph': float(7.0 + 0.1 * idx),
                    'flow': float(2000 + 100 * idx),
                }
            
            # 使用刚才的预测结果作为输入
            recs = pipeline.optimize_only(
                preds, 
                current_features=mock_features
            )
            self.assertEqual(len(recs), len(enabled_pools))
            
            pool_1_recs = recs['pool_1']
            print(f"Pool 1 推荐: {pool_1_recs}")
            self.assertEqual(len(pool_1_recs), 5)
            print("Pool 1 推荐 5 个点 - OK")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"管道执行失败: {e}")

if __name__ == '__main__':
    unittest.main()
