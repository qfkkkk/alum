"""
MPC 控制器闭环仿真测试 - 正确的滚动优化实现

测试数据：
- 使用2025年6月13日的数据进行测试（共288条记录）
- 覆盖一整天不同时段的运行情况

关键改进：
1. 使用MPC计算的投矾量（而非真实值）更新历史数据
2. 只使用真实的浊度、流量、pH等测量值作为反馈
3. 真正模拟"如果使用MPC控制，系统会如何响应"

CSV输出包含：
- MPC优化的投加药量 (MPC_Dose)
- 预测的浊度 (MPC_Turbidity)
- 真实的投加药量和浊度
- 预测模型的输入特征 (Input_Dose, Input_Turb_Chushui, Input_Turb_Jinshui, Input_Flow, Input_pH, Input_Temp)

性能优化（用于快速验证）：
- 仿真步数: 10步
- 验证样本: 3个
- 测试池子: 2个（池1和池2）
- 预测时域: 6步
- 控制时域: 5步
- PSO迭代次数: 30
- PSO种群大小: 20

注意：如需完整测试，可调整上述参数。
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimizers.config import MPCConfig, PoolState
from optimizers.controller import create_mpc_controller


def test_mpc_rolling_optimization():
    """
    MPC 滚动优化闭环仿真
    
    核心思想：
    - 每一步使用MPC优化的投矾量作为"实际应用"的控制量
    - 使用预测模型估算该控制量导致的浊度变化
    - 用真实数据的其他变量（流量、pH等）作为扰动
    """
    print("\n" + "="*80)
    print("MPC 滚动优化闭环仿真（正确实现）")
    print("="*80)
    
    # 加载真实数据
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_data.csv')
    print(f"\n[1] 加载数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"    数据总量: {len(df)} 条记录")
    
    # 使用2025年6月13日作为测试数据
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    target_date = pd.to_datetime('2025-06-13').date()
    test_df = df[df['DateTime'].dt.date == target_date].reset_index(drop=True)
    print(f"    测试日期: 2025-06-13")
    print(f"    测试数据量: {len(test_df)} 条记录")
    
    # 创建MPC配置（已优化以缩短运行时间）
    config = MPCConfig(
        prediction_horizon=6,  # 从6减少到4
        control_horizon=5,     # 从5减少到3
        time_step=5.0,
        target_turbidity=0.5,
        lambda_du=0.0005,       # 从0.005减少到0.001，允许更大的投加量变化
        optimizer_type='pso',
        max_iterations=30,     # 从50减少到30
        population_size=20,    # 从30减少到20
        models_dir='../models'
    )
    
    print(f"\n[2] MPC 配置:")
    print(f"    预测时域: {config.prediction_horizon} 步")
    print(f"    控制时域: {config.control_horizon} 步")
    print(f"    目标浊度: {config.target_turbidity} NTU")
    
    # 创建MPC控制器
    print(f"\n[3] 初始化 MPC 控制器...")
    controller = create_mpc_controller(config)
    
    # 仿真参数（针对单日数据调整）
    seq_len = 60
    n_simulation_steps = 10  # 仿真步数
    n_samples = 3            # 增加样本数以更好地覆盖一天的不同时段
    
    # 筛选有效的起始时刻（确保投加量不为0）
    print(f"\n[4] 筛选有效的仿真起始时刻（投加量不为0）...")
    available_start = seq_len
    available_end = len(test_df) - n_simulation_steps - config.prediction_horizon
    
    print(f"    测试数据范围: {available_start} - {available_end} (共 {available_end - available_start} 个可能起点)")
    
    # 找出所有投加量不为0的时间段
    valid_indices = []
    for idx in range(available_start, available_end):
        # 检查该时刻及后续n_simulation_steps步内，至少有一个池子的投加量不为0
        is_valid = False
        for pool_id in [1, 2]:  # 只检查要测试的池子
            doses = test_df.loc[idx:idx+n_simulation_steps, f'dose_{pool_id}'].values
            if np.any(doses > 0):  # 至少有一个投加量不为0
                is_valid = True
                break
        if is_valid:
            valid_indices.append(idx)
    
    print(f"    可用时间段: {len(valid_indices)} 个")
    
    if len(valid_indices) < n_samples:
        print(f"    [警告] 有效样本数量 {len(valid_indices)} < 请求样本数 {n_samples}")
        n_samples = len(valid_indices)
    
    # 从有效时间段中均匀选择（覆盖早、中、晚不同时段）
    if len(valid_indices) > 0:
        selected_positions = np.linspace(0, len(valid_indices)-1, n_samples, dtype=int)
        start_indices = [valid_indices[i] for i in selected_positions]
    else:
        raise ValueError("没有找到有效的仿真起始时刻（所有时间段投加量都为0）")
    
    print(f"\n[5] 仿真设置:")
    print(f"    历史序列: {seq_len} 步")
    print(f"    仿真步数: {n_simulation_steps} 步")
    print(f"    验证样本: {n_samples} 个")
    print(f"    起始时刻: {[test_df.loc[idx, 'DateTime'] for idx in start_indices]}")
    
    # 存储结果
    all_results = []
    
    # 对每个起始时刻进行仿真
    for sample_id, start_idx in enumerate(start_indices):
        print(f"\n{'='*80}")
        print(f"仿真样本 {sample_id + 1}/{n_samples}")
        print(f"起始时刻: {test_df.loc[start_idx, 'DateTime']}")
        # 显示该时刻的真实投加量
        print(f"真实投加量: ", end="")
        for pool_id in [1, 2]:
            dose = test_df.loc[start_idx, f'dose_{pool_id}']
            print(f"池{pool_id}={dose:.2f} mg/L  ", end="")
        print()
        print('='*80)
        
        simulation_result = {
            'sample_id': sample_id + 1,
            'start_time': test_df.loc[start_idx, 'DateTime'],
            'start_index': start_idx,
            'pools': {}
        }
        
        # 对每个池子进行仿真（仅测试池子1和2以缩短运行时间）
        for pool_id in [1, 2]:
            print(f"\n池子 {pool_id} - MPC滚动优化仿真:")
            print('-' * 60)
            
            pool_trajectory = {
                'time_steps': [],
                'datetime': [],
                'mpc_doses': [],
                'real_doses': [],
                'mpc_turbidity': [],  # MPC闭环下的浊度（由预测模型给出）
                'real_turbidity': [],
                'cost_values': [],
                'control_sequences': [],  # 完整的控制序列
                # 预测模型的输入特征
                'input_dose': [],
                'input_turb_chushui': [],
                'input_turb_jinshui': [],
                'input_flow': [],
                'input_pH': [],
                'input_temp': []
            }
            
            # 初始化历史数据
            history_start = start_idx - seq_len
            history_end = start_idx
            
            # 提取初始历史数据（使用真实历史）
            history_data = np.zeros((seq_len, 6))
            history_data[:, 0] = test_df.loc[history_start:history_end-1, f'dose_{pool_id}'].values
            history_data[:, 1] = test_df.loc[history_start:history_end-1, f'turb_chushui_{pool_id}'].values
            history_data[:, 2] = test_df.loc[history_start:history_end-1, f'turb_jinshui_{pool_id}'].values
            history_data[:, 3] = test_df.loc[history_start:history_end-1, f'flow_{pool_id}'].values
            history_data[:, 4] = test_df.loc[history_start:history_end-1, 'pH'].values
            history_data[:, 5] = test_df.loc[history_start:history_end-1, 'temp_shuimian'].values
            
            # 当前投矾量（初始值使用真实历史的最后一个）
            current_mpc_dose = test_df.loc[start_idx-1, f'dose_{pool_id}']
            
            # 获取预测器以便使用模型预测浊度
            predictor = controller.predictors[pool_id]
            
            # 开始滚动优化仿真
            for step in range(n_simulation_steps):
                current_idx = start_idx + step
                
                # 1. 创建池子状态（基于当前历史数据和MPC的投矾量）
                pool_state = PoolState(
                    pool_id=pool_id,
                    current_dose=current_mpc_dose,  # 使用MPC的投矾量
                    history_data=history_data
                )
                
                # 2. MPC优化，计算最优控制序列
                try:
                    result = controller.control_step(pool_state, verbose=False)
                    
                    # 3. 获取MPC优化的下一步投矾量
                    mpc_dose = result['next_dose']
                    control_sequence = result['optimal_dose_sequence']
                    
                    # 4. 使用预测模型估算MPC控制下的浊度
                    # 注意：这里使用MPC的投矾量进行预测
                    future_doses_mpc = np.array([mpc_dose])  # 只预测下一步
                    try:
                        mpc_turb_pred = predictor.predict(
                            history_data, 
                            np.repeat(mpc_dose, predictor.pred_len)
                        )[0]  # 取第一步的预测
                    except:
                        mpc_turb_pred = result['predicted_turbidity'][0]
                    
                    # 5. 获取真实数据（用于对比）
                    real_dose = test_df.loc[current_idx, f'dose_{pool_id}']
                    real_turbidity = test_df.loc[current_idx, f'turb_chushui_{pool_id}']
                    
                    # 6. 记录结果和输入特征
                    pool_trajectory['time_steps'].append(step)
                    pool_trajectory['datetime'].append(test_df.loc[current_idx, 'DateTime'])
                    pool_trajectory['mpc_doses'].append(mpc_dose)
                    pool_trajectory['real_doses'].append(real_dose)
                    pool_trajectory['mpc_turbidity'].append(mpc_turb_pred)
                    pool_trajectory['real_turbidity'].append(real_turbidity)
                    pool_trajectory['cost_values'].append(result['optimal_cost'])
                    pool_trajectory['control_sequences'].append(control_sequence.tolist())
                    
                    # 记录当前时刻的输入特征（历史数据的最后一行）
                    pool_trajectory['input_dose'].append(history_data[-1, 0])
                    pool_trajectory['input_turb_chushui'].append(history_data[-1, 1])
                    pool_trajectory['input_turb_jinshui'].append(history_data[-1, 2])
                    pool_trajectory['input_flow'].append(history_data[-1, 3])
                    pool_trajectory['input_pH'].append(history_data[-1, 4])
                    pool_trajectory['input_temp'].append(history_data[-1, 5])
                    
                    # 7. 打印进度
                    if step % 5 == 0 or step == 0:
                        print(f"  步骤 {step+1:2d} | "
                              f"MPC投矾: {mpc_dose:6.2f} mg/L | 真实投矾: {real_dose:6.2f} mg/L | "
                              f"差值: {mpc_dose-real_dose:+6.2f} mg/L")
                        print(f"           | "
                              f"MPC浊度: {mpc_turb_pred:.4f} NTU | 真实浊度: {real_turbidity:.4f} NTU | "
                              f"Cost: {result['optimal_cost']:.4f}")
                    
                    # 8. 【关键】滚动更新历史数据
                    # 方案A: 使用MPC的投矾量 + 真实的其他测量值
                    new_row_mpc = np.array([
                        mpc_dose,  # 使用MPC优化的投矾量
                        mpc_turb_pred,  # 使用模型预测的浊度
                        test_df.loc[current_idx, f'turb_jinshui_{pool_id}'],  # 真实进水浊度
                        test_df.loc[current_idx, f'flow_{pool_id}'],  # 真实流量
                        test_df.loc[current_idx, 'pH'],  # 真实pH
                        test_df.loc[current_idx, 'temp_shuimian']  # 真实温度
                    ])
                    
                    # 滚动历史窗口
                    history_data = np.vstack([history_data[1:], new_row_mpc.reshape(1, -1)])
                    
                    # 更新当前MPC投矾量（用于下一步优化）
                    current_mpc_dose = mpc_dose
                    
                except Exception as e:
                    print(f"  [错误] 步骤 {step+1} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # 保存该池子的轨迹
            simulation_result['pools'][pool_id] = pool_trajectory
            
            # 计算统计指标
            if len(pool_trajectory['mpc_doses']) > 0:
                mpc_doses = np.array(pool_trajectory['mpc_doses'])
                real_doses = np.array(pool_trajectory['real_doses'])
                mpc_turb = np.array(pool_trajectory['mpc_turbidity'])
                real_turb = np.array(pool_trajectory['real_turbidity'])
                
                # 投矾量对比
                dose_diff_mean = np.mean(mpc_doses - real_doses)
                dose_diff_std = np.std(mpc_doses - real_doses)
                dose_mae = np.mean(np.abs(mpc_doses - real_doses))
                
                # MPC控制下的平均浊度
                mpc_turb_mean = np.mean(mpc_turb)
                real_turb_mean = np.mean(real_turb)
                
                # MPC达标率
                target_turb = config.target_turbidity
                mpc_达标率 = np.sum(mpc_turb <= target_turb) / len(mpc_turb) * 100
                real_达标率 = np.sum(real_turb <= target_turb) / len(real_turb) * 100
                
                print(f"\n  池子 {pool_id} 统计指标:")
                print(f"    投矾量差异:")
                print(f"      平均差值: {dose_diff_mean:+.4f} mg/L")
                print(f"      标准差:   {dose_diff_std:.4f} mg/L")
                print(f"      MAE:      {dose_mae:.4f} mg/L")
                print(f"    浊度对比:")
                print(f"      MPC平均浊度:  {mpc_turb_mean:.4f} NTU")
                print(f"      真实平均浊度: {real_turb_mean:.4f} NTU")
                print(f"    达标率:")
                print(f"      MPC达标率:   {mpc_达标率:.1f}%")
                print(f"      真实达标率:  {real_达标率:.1f}%")
                
                # 如果MPC平均投矾量更少且达标率更高，说明优化有效
                if dose_diff_mean < 0 and mpc_达标率 >= real_达标率:
                    print(f"    [优化有效] MPC节约投矾 {abs(dose_diff_mean):.2f} mg/L，同时保证达标")
                elif dose_diff_mean < 0:
                    print(f"    [注意] MPC节约投矾但达标率下降")
                elif mpc_达标率 > real_达标率:
                    print(f"    [优化有效] MPC提高达标率")
        
        all_results.append(simulation_result)
    
    # 保存结果
    print(f"\n{'='*80}")
    print("[6] 保存仿真结果")
    print('='*80)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for result in all_results:
        for pool_id in [1, 2]:  # 仅保存测试的池子
            if pool_id in result['pools']:
                traj = result['pools'][pool_id]
                
                # 创建DataFrame
                df_result = pd.DataFrame({
                    'DateTime': traj['datetime'],
                    'TimeStep': traj['time_steps'],
                    'MPC_Dose': traj['mpc_doses'],
                    'Real_Dose': traj['real_doses'],
                    'Dose_Difference': np.array(traj['mpc_doses']) - np.array(traj['real_doses']),
                    'MPC_Turbidity': traj['mpc_turbidity'],
                    'Real_Turbidity': traj['real_turbidity'],
                    'Turbidity_Difference': np.array(traj['mpc_turbidity']) - np.array(traj['real_turbidity']),
                    'Cost_Value': traj['cost_values'],
                    # 预测模型的输入特征
                    'Input_Dose': traj['input_dose'],
                    'Input_Turb_Chushui': traj['input_turb_chushui'],
                    'Input_Turb_Jinshui': traj['input_turb_jinshui'],
                    'Input_Flow': traj['input_flow'],
                    'Input_pH': traj['input_pH'],
                    'Input_Temp': traj['input_temp']
                })
                
                # 保存文件
                filename = f'mpc_rolling_sample{result["sample_id"]}_pool{pool_id}_{timestamp}.csv'
                filepath = os.path.join(output_dir, filename)
                df_result.to_csv(filepath, index=False, encoding='utf-8-sig')
                print(f"    已保存: {filename}")
    
    # 总结
    print(f"\n{'='*80}")
    print("[7] MPC滚动优化总结")
    print('='*80)
    
    for result in all_results:
        print(f"\n样本 {result['sample_id']}: {result['start_time']}")
        print(f"{'池子':<6} {'投矾差值':<12} {'投矾MAE':<12} {'MPC浊度':<12} {'真实浊度':<12} {'MPC达标率':<12} {'真实达标率':<12}")
        print('-' * 90)
        
        for pool_id in [1, 2]:  # 仅总结测试的池子
            if pool_id in result['pools']:
                traj = result['pools'][pool_id]
                if len(traj['mpc_doses']) > 0:
                    mpc_doses = np.array(traj['mpc_doses'])
                    real_doses = np.array(traj['real_doses'])
                    mpc_turb = np.array(traj['mpc_turbidity'])
                    real_turb = np.array(traj['real_turbidity'])
                    
                    dose_diff = np.mean(mpc_doses - real_doses)
                    dose_mae = np.mean(np.abs(mpc_doses - real_doses))
                    mpc_turb_mean = np.mean(mpc_turb)
                    real_turb_mean = np.mean(real_turb)
                    mpc_rate = np.sum(mpc_turb <= config.target_turbidity) / len(mpc_turb) * 100
                    real_rate = np.sum(real_turb <= config.target_turbidity) / len(real_turb) * 100
                    
                    print(f"池{pool_id:<5} {dose_diff:>+10.4f}   {dose_mae:>10.4f}   "
                          f"{mpc_turb_mean:>10.4f}   {real_turb_mean:>10.4f}   "
                          f"{mpc_rate:>10.1f}%   {real_rate:>10.1f}%")
    
    print(f"\n{'='*80}")
    print("[完成] MPC 滚动优化仿真完成!")
    print("\n说明:")
    print("- 本次仿真使用MPC计算的投矾量进行滚动更新")
    print("- 这样可以真正评估'如果使用MPC控制，效果会如何'")
    print("- 投矾差值为负表示MPC比真实操作更节约")
    print('='*80)
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MPC 滚动优化闭环仿真 - 正确的滚动时域实现',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='计算设备 (默认: cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[警告] CUDA不可用，将使用CPU")
    
    print(f"\n计算设备: {device}")
    
    try:
        results = test_mpc_rolling_optimization()
        
        print(f"\n{'='*80}")
        print("[成功] MPC滚动优化仿真完成!")
        print('='*80)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"[错误] 测试失败: {e}")
        print('='*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
