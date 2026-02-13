
import unittest
import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.dosing_pipeline import DosingPipeline
from services.io_adapter import read_data

class TestDosingPipeline(unittest.TestCase):
    
    def setUp(self):
        self.config = None

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

        # 读取真实输入（本地模式：由 dataio 生成假数据；远程模式：读取真实源）
        data_dict, last_dt = read_data(pipeline.predictor_manager.config)
        
        # 执行管道
        try:
            results = pipeline.run(
                data_dict, 
                last_dt
            )
            print(results)
            # 验证结果结构
            for pool_name, res in results.items():
                print(f"\n检查 {pool_name}:")

                # 验证推荐结果 (5个点)
                recs = res['recommendations']
                self.assertEqual(len(recs), 5)

                # 验证返回值类型
                rec_times = sorted(recs.keys())
                first_rec = recs[rec_times[0]]
                print(f"  首个推荐投药量: {first_rec}")
                self.assertIsInstance(first_rec, float)

                # run 仅返回优化结果，不返回预测结果
                self.assertNotIn('predictions', res)
                self.assertNotIn('generated_at', res)
            
            print("管道 Run 流程验证通过！")
            # --- 新增测试: 单独调用 predict_only ---
            print("\n=== 测试 predict_only ===")
            preds = pipeline.predict_only(data_dict, last_dt)
            self.assertEqual(len(preds), len(enabled_pools))
            print(f"预测只有 {len(preds)} 个池子 - OK")
            
            pool_1_preds = preds['pool_1']
            self.assertEqual(len(pool_1_preds), 6)
            print("Pool 1 预测 6 个点 - OK")
            
            # --- 新增测试: 单独调用 optimize_only ---
            print("\n=== 测试 optimize_only ===")

            # 使用真实输入提取当前特征，避免手工 mock
            current_features = pipeline._extract_last_features(data_dict)
            
            # 使用刚才的预测结果作为输入
            recs = pipeline.optimize_only(
                preds, 
                current_features=current_features
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

    def test_predict_turbidity(self):
        """测试浊度预测结果结构与数据类型"""
        print("\n=== 测试浊度预测 ===")

        pipeline = DosingPipeline(optimizer_type='pso')
        enabled_pools = pipeline.predictor_manager.enabled_pools
        if not enabled_pools:
            print("没有启用的池子，跳过测试")
            return

        data_dict, last_dt = read_data(pipeline.predictor_manager.config)
        preds = pipeline.predict_only(data_dict, last_dt)
        print("pipeline.predict_only preds:", preds)
        pred_len = int(pipeline.predictor_manager.config.get('pred_len', 6))
        self.assertEqual(len(preds), len(enabled_pools))

        for pool_name in enabled_pools:
            self.assertIn(pool_name, preds)
            pool_preds = preds[pool_name]
            self.assertEqual(len(pool_preds), pred_len)

            first_ts = sorted(pool_preds.keys())[0]
            first_val = pool_preds[first_ts]
            self.assertIsInstance(first_ts, str)
            self.assertIsInstance(float(first_val), float)

if __name__ == '__main__':
    unittest.main()
