# -*- encoding: utf-8 -*-
"""
DosingOptimizer 测试脚本
测试 MPC 投药优化器的功能
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from datetime import datetime
from optimizers.dosing_optimizer import create_dosing_optimizer


def test_basic_optimization():
    """测试基本的优化功能"""
    print("=" * 80)
    print("测试 1: 基本优化功能")
    print("=" * 80)
    
    # 1. 创建优化器
    print("\n[步骤 1] 创建优化器实例...")
    optimizer = create_dosing_optimizer('pool_1')
    
    # 2. 准备输入数据
    print("\n[步骤 2] 准备输入数据...")
    
    # 模拟预测器输出（6个预测值，对应预测时域 Np=6）
    predictions = {
        'pool_1': {
            '2024-01-01 12:05:00': 0.95,  # t+1
            '2024-01-01 12:10:00': 0.98,  # t+2
            '2024-01-01 12:15:00': 1.02,  # t+3
            '2024-01-01 12:20:00': 1.05,  # t+4
            '2024-01-01 12:25:00': 1.08,  # t+5
            '2024-01-01 12:30:00': 1.10,  # t+6
        }
    }
    
    # 当前特征（包含当前投矾量）
    current_features = {
        'pool_1': {
            'current_dose': 15.0,  # 当前投矾量 15 mg/L
            'ph': 7.2,
            'flow': 2000.0,
        }
    }
    
    # 最后一个数据点的时间
    last_datetime = datetime(2024, 1, 1, 12, 0, 0)
    
    # 3. 执行优化
    print("\n[步骤 3] 执行 MPC 优化...")
    result = optimizer.optimize(
        predictions=predictions,
        current_features=current_features,
        last_datetime=last_datetime
    )
    
    # 4. 显示结果
    print("\n[步骤 4] 优化结果:")
    print("-" * 80)
    for pool_id, dose_schedule in result.items():
        print(f"\n池子 {pool_id} 的控制序列（5步，对应控制时域 Nc=5）:")
        for time_str, dose_value in dose_schedule.items():
            print(f"  {time_str}: {dose_value:.2f} mg/L")
    
    print("\n[OK] 测试 1 完成")
    return result


def test_multiple_pools():
    """测试多个池子的优化"""
    print("\n" + "=" * 80)
    print("测试 2: 多池子优化")
    print("=" * 80)
    
    from optimizers.dosing_optimizer import create_multi_pool_optimizers
    
    # 1. 创建多个优化器
    print("\n[步骤 1] 创建多个优化器实例...")
    pool_ids = ['pool_1', 'pool_2']
    optimizers = create_multi_pool_optimizers(pool_ids)
    
    # 2. 准备输入数据
    print("\n[步骤 2] 准备输入数据...")
    
    predictions = {
        'pool_1': {
            '2024-01-01 12:05:00': 0.95,
            '2024-01-01 12:10:00': 0.98,
            '2024-01-01 12:15:00': 1.02,
            '2024-01-01 12:20:00': 1.05,
            '2024-01-01 12:25:00': 1.08,
            '2024-01-01 12:30:00': 1.10,
        },
        'pool_2': {
            '2024-01-01 12:05:00': 1.15,
            '2024-01-01 12:10:00': 1.18,
            '2024-01-01 12:15:00': 1.22,
            '2024-01-01 12:20:00': 1.25,
            '2024-01-01 12:25:00': 1.28,
            '2024-01-01 12:30:00': 1.30,
        }
    }
    
    current_features = {
        'pool_1': {'current_dose': 15.0},
        'pool_2': {'current_dose': 18.0},
    }
    
    last_datetime = datetime(2024, 1, 1, 12, 0, 0)
    
    # 3. 执行优化
    print("\n[步骤 3] 执行优化...")
    results = {}
    for pool_id, optimizer in optimizers.items():
        print(f"\n优化 {pool_id}...")
        result = optimizer.optimize(
            predictions={pool_id: predictions[pool_id]},
            current_features={pool_id: current_features[pool_id]},
            last_datetime=last_datetime
        )
        results.update(result)
    
    # 4. 显示结果
    print("\n[步骤 4] 优化结果:")
    print("-" * 80)
    for pool_id in pool_ids:
        print(f"\n池子 {pool_id} 的控制序列:")
        for time_str, dose_value in results[pool_id].items():
            print(f"  {time_str}: {dose_value:.2f} mg/L")
    
    print("\n[OK] 测试 2 完成")
    return results



if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DosingOptimizer 测试套件" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # 运行所有测试
        test_basic_optimization()
        test_multiple_pools()
        
        print("\n" + "=" * 80)
        print("所有测试完成！✓")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"测试失败: {e}")
        print("=" * 80 + "\n")
        import traceback
        traceback.print_exc()
