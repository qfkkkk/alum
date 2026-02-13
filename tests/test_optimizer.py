# -*- encoding: utf-8 -*-
"""
投药优化器测试脚本
功能：使用真实CSV数据对DosingOptimizer进行滚动测试
数据源：test_data.csv（离线真实数据）
"""
import sys
from pathlib import Path

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

from predictors.turbidity_predictor import TurbidityPredictor
from optimizers.dosing_optimizer import create_dosing_optimizer
from utils.config_loader import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class OptimizerRollingTester:
    """
    优化器滚动测试器
    
    功能说明：
        - 使用真实CSV数据进行滚动测试
        - 对每个池子独立创建预测器和优化器
        - 按5分钟步长滚动，每次预测6步，输出5步控制序列
        - 打印预测器输出和优化器结果
    """
    
    def __init__(self, csv_path: str, config_path: str = None):
        """
        初始化测试器
        
        参数：
            csv_path: 测试数据CSV文件路径
            config_path: 配置文件路径（默认使用 configs/app.yaml）
        """
        self.csv_path = csv_path
        self.config = load_config(config_path)
        
        # 从配置中读取参数
        self.seq_len = self.config.get('seq_len', 60)
        self.pred_len = self.config.get('pred_len', 6)
        self.control_horizon = self.config.get('control_horizon', 5)
        self.time_interval = self.config.get('time_interval_minutes', 5)
        self.features = self.config.get('features', [])
        
        # 池子列表
        self.pool_ids = ['pool_1', 'pool_2', 'pool_3', 'pool_4']
        
        # 数据
        self.df = None
        self.predictors = {}
        self.optimizers = {}
        
        logger.info("=" * 80)
        logger.info("投药优化器滚动测试器初始化完成")
        logger.info(f"  - 数据源: {csv_path}")
        logger.info(f"  - 输入步长: {self.seq_len} 步 (历史数据)")
        logger.info(f"  - 预测步长: {self.pred_len} 步 (未来30分钟)")
        logger.info(f"  - 控制步长: {self.control_horizon} 步 (控制序列)")
        logger.info(f"  - 时间间隔: {self.time_interval} 分钟")
        logger.info("=" * 80)
    
    def load_data(self):
        """加载CSV数据并进行预处理"""
        logger.info("\n[1] 加载数据...")
        
        self.df = pd.read_csv(self.csv_path)
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df = self.df.sort_values('DateTime').reset_index(drop=True)
        
        logger.info(f"  ✓ 数据加载完成: {len(self.df)} 行")
        logger.info(f"  - 时间范围: {self.df['DateTime'].min()} 至 {self.df['DateTime'].max()}")
        logger.info(f"  - 列名: {list(self.df.columns)}")
    
    def initialize_predictors(self):
        """为每个池子初始化预测器"""
        logger.info("\n[2] 初始化预测器...")
        
        for pool_id in self.pool_ids:
            pool_num = int(pool_id.split('_')[1])
            try:
                predictor = TurbidityPredictor(pool_id=pool_num, config=self.config)
                self.predictors[pool_id] = predictor
                logger.info(f"  ✓ {pool_id} 预测器初始化完成")
            except Exception as e:
                logger.error(f"  ✗ {pool_id} 预测器初始化失败: {e}")
                raise
    
    def initialize_optimizers(self):
        """为每个池子初始化优化器"""
        logger.info("\n[3] 初始化优化器...")
        
        for pool_id in self.pool_ids:
            try:
                optimizer = create_dosing_optimizer(pool_id=pool_id)
                self.optimizers[pool_id] = optimizer
                logger.info(f"  ✓ {pool_id} 优化器初始化完成")
            except Exception as e:
                logger.error(f"  ✗ {pool_id} 优化器初始化失败: {e}")
                raise
    
    def prepare_pool_input(self, pool_id: str, start_idx: int) -> np.ndarray:
        """
        准备单个池子的输入数据
        
        参数：
            pool_id: 池子ID（如 'pool_1'）
            start_idx: 起始索引
            
        返回：
            shape [seq_len, n_features] 的数组
        """
        pool_num = int(pool_id.split('_')[1])
        
        # CSV列名映射（根据实际数据文件）
        column_map = {
            'dose': f'dose_{pool_num}',
            'turb_chushui': f'turb_chushui_{pool_num}',
            'turb_jinshui': f'turb_jinshui_{pool_num}',
            'flow': f'flow_{pool_num}',
            'pH': 'pH',
            'temp_shuimian': 'temp_shuimian'
        }
        
        # 提取数据
        data_slice = self.df.iloc[start_idx:start_idx + self.seq_len]
        
        # 按照 features 顺序构建输入
        input_data = []
        for feat in self.features:
            col_name = column_map[feat]
            input_data.append(data_slice[col_name].values)
        
        # 转置为 [seq_len, n_features]
        input_array = np.array(input_data).T
        
        return input_array
    
    def test_single_pool(self, pool_id: str, start_idx: int, current_time: datetime) -> Dict:
        """
        测试单个池子的预测和优化
        
        参数：
            pool_id: 池子ID
            start_idx: 数据起始索引
            current_time: 当前时间点
            
        返回：
            包含预测结果和优化结果的字典
        """
        logger.info(f"\n    --- {pool_id.upper()} ---")
        
        # 1. 准备输入数据
        input_data = self.prepare_pool_input(pool_id, start_idx)
        
        # 获取当前投矾量（输入序列的最后一个值）
        pool_num = int(pool_id.split('_')[1])
        current_dose = float(self.df.iloc[start_idx + self.seq_len - 1][f'dose_{pool_num}'])
        
        logger.info(f"    输入数据: shape={input_data.shape}, 当前投矾={current_dose:.2f} mg/L")
        
        # 2. 调用预测器
        predictor = self.predictors[pool_id]
        
        try:
            predictions_dict = predictor.predict_with_timestamps(
                input_data=input_data,
                last_datetime=current_time,
                time_interval_minutes=self.time_interval
            )
            
            logger.info(f"    【预测器输出】:")
            for time_str, pred_value in predictions_dict.items():
                logger.info(f"      {time_str}: {pred_value:.4f} NTU")
        
        except Exception as e:
            logger.error(f"    ✗ 预测失败: {e}")
            raise
        
        # 3. 调用优化器
        optimizer = self.optimizers[pool_id]
        
        try:
            # 构建优化器输入格式
            predictions_input = {pool_id: predictions_dict}
            current_features = {pool_id: {'current_dose': current_dose}}
            
            # 执行优化
            result = optimizer.optimize(
                predictions=predictions_input,
                current_features=current_features,
                last_datetime=current_time
            )
            
            # 优化器返回的是字典 {'时间': 投矾量}
            dose_dict = result[pool_id]
            
            logger.info(f"    【优化器输出】控制序列 (共{len(dose_dict)}步):")
            
            # 打印带时间戳的控制序列
            for time_str, dose in dose_dict.items():
                logger.info(f"      {time_str}: {dose:.2f} mg/L")
            
            # 提取投矾量列表用于后续统计
            dose_sequence = list(dose_dict.values())
            
            return {
                'predictions': predictions_dict,
                'control_sequence': dose_sequence,
                'current_dose': current_dose
            }
        
        except Exception as e:
            logger.error(f"    ✗ 优化失败: {e}")
            raise
    
    def run_rolling_test(self, num_iterations: int = 10, start_offset: int = 0):
        """
        执行滚动测试
        
        参数：
            num_iterations: 滚动迭代次数（默认10次，即50分钟）
            start_offset: 起始偏移量（从第几个数据点开始）
        """
        logger.info("\n" + "=" * 80)
        logger.info("[4] 开始滚动测试")
        logger.info("=" * 80)
        
        # 确保有足够的数据
        total_needed = start_offset + self.seq_len + num_iterations
        if total_needed > len(self.df):
            logger.warning(f"数据不足，最多可执行 {len(self.df) - start_offset - self.seq_len} 次迭代")
            num_iterations = len(self.df) - start_offset - self.seq_len
        
        results_all = {pool_id: [] for pool_id in self.pool_ids}
        
        # 滚动测试
        for iter_idx in range(num_iterations):
            current_idx = start_offset + iter_idx
            current_time = self.df.iloc[current_idx + self.seq_len - 1]['DateTime']
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"【迭代 {iter_idx + 1}/{num_iterations}】当前时间: {current_time}")
            logger.info(f"{'=' * 80}")
            
            # 对每个池子进行测试
            for pool_id in self.pool_ids:
                try:
                    result = self.test_single_pool(pool_id, current_idx, current_time)
                    results_all[pool_id].append(result)
                except Exception as e:
                    logger.error(f"  ✗ {pool_id} 测试失败: {e}")
                    continue
        
        logger.info("\n" + "=" * 80)
        logger.info("滚动测试完成！")
        logger.info("=" * 80)
        
        # 打印汇总统计
        self._print_summary(results_all, num_iterations)
        
        return results_all
    
    def _print_summary(self, results_all: Dict, num_iterations: int):
        """打印测试结果汇总"""
        logger.info("\n" + "=" * 80)
        logger.info("【测试结果汇总】")
        logger.info("=" * 80)
        
        for pool_id in self.pool_ids:
            results = results_all[pool_id]
            if not results:
                logger.info(f"\n{pool_id.upper()}: 无有效结果")
                continue
            
            logger.info(f"\n{pool_id.upper()}:")
            logger.info(f"  - 成功迭代次数: {len(results)}/{num_iterations}")
            
            # 统计预测值范围
            all_predictions = []
            for res in results:
                all_predictions.extend(res['predictions'].values())
            
            if all_predictions:
                logger.info(f"  - 预测浊度范围: {min(all_predictions):.4f} ~ {max(all_predictions):.4f} NTU")
                logger.info(f"  - 预测浊度均值: {np.mean(all_predictions):.4f} NTU")
            
            # 统计控制序列范围
            all_doses = []
            for res in results:
                all_doses.extend(res['control_sequence'])
            
            if all_doses:
                logger.info(f"  - 控制投矾范围: {min(all_doses):.2f} ~ {max(all_doses):.2f} mg/L")
                logger.info(f"  - 控制投矾均值: {np.mean(all_doses):.2f} mg/L")


def main():
    """主函数"""
    # 配置路径
    csv_path = project_root / "test_data.csv"
    
    # 创建测试器
    tester = OptimizerRollingTester(csv_path=str(csv_path))
    
    # 加载数据
    tester.load_data()
    
    # 初始化预测器和优化器
    tester.initialize_predictors()
    tester.initialize_optimizers()
    
    results = tester.run_rolling_test(
        num_iterations=10,  # 测试10次滚动优化
        start_offset=60     # 从第60个数据点开始（确保有60步历史）
    )
    
    logger.info("\n测试完成！")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"\n测试过程中发生错误: {e}", exc_info=True)
        sys.exit(1)
