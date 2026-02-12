# -*- encoding: utf-8 -*-
"""
MPC 投药优化器（DosingOptimizer）
================================

功能概述：
    基于模型预测控制（Model Predictive Control, MPC）的投药量优化器
    
继承关系：
    BaseOptimizer (optimizers/base_optimizer.py)
    
主要特点：
    1. 接收预测器输出的浊度预测作为输入（参考 predictors/base_predictor.py）
    2. 基于 MPC 框架进行多步预测优化
    3. 使用智能优化算法求解（PSO/SA/DE，参考 optimizers/optimizer.py）
    4. 考虑约束条件（投矾量上下限、变化率限制）
    5. 输出控制时域内的投矾量控制序列

MPC 控制框架说明：
    - 预测时域 (Np): 6步（每步5分钟，共30分钟）
    - 控制时域 (Nc): 5步（输出5个控制动作）
    - 目标函数: J = Σ||y(t+k) - y_target||² + λ·Σ||Δu(t+k)||²
      * 第一项：跟踪误差，使出水浊度接近目标值
      * 第二项：控制变化惩罚，使投矾量变化平稳
    - 约束条件:
      * 投矾量范围: [dose_min, dose_max]
      * 变化率限制: |Δu| ≤ dose_rate_limit
      
配置文件：
    - 配置加载器: utils/config_loader.py
    - 配置文件: configs/app.yaml
    
使用示例：
    >>> from optimizers.dosing_optimizer import create_dosing_optimizer
    >>> 
    >>> # 创建优化器
    >>> optimizer = create_dosing_optimizer('pool_1')
    >>> 
    >>> # 执行优化
    >>> result = optimizer.optimize(
    ...     predictions={'pool_1': {
    ...         '2024-01-01 12:05': 0.95,
    ...         '2024-01-01 12:10': 0.98,
    ...         '2024-01-01 12:15': 1.02,
    ...         '2024-01-01 12:20': 1.05,
    ...         '2024-01-01 12:25': 1.08,
    ...         '2024-01-01 12:30': 1.10,
    ...     }},
    ...     current_features={'pool_1': {'current_dose': 15.0}},
    ...     last_datetime=datetime(2024, 1, 1, 12, 0)
    ... )
    >>> 
    >>> # 输出格式: {'pool_1': {'2024-01-01 12:05': 15.5, ...}}
    >>> print(result)

作者: Water Treatment Optimization System
日期: 2024
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_optimizer import BaseOptimizer
from .optimizer import create_optimizer
from utils.config_loader import load_config


class DosingOptimizer(BaseOptimizer):
    """
    MPC 投药优化器
    
    功能说明：
        - 基于模型预测控制（MPC）算法优化投药量
        - 接收预测器输出的浊度预测作为输入
        - 通过优化器（PSO/SA/DE）求解最优控制序列
        
    使用示例：
        optimizer = DosingOptimizer(config=yaml_config, pool_id='pool_1')
        result = optimizer.optimize(
            predictions={'pool_1': {'2024-01-01 12:05': 0.95, ...}},
            current_features={'pool_1': {'current_dose': 15.0}},
            last_datetime=datetime(2024, 1, 1, 12, 0)
        )
    """
    
    def __init__(self, config: dict, pool_id: str):
        """
        初始化 MPC 优化器
        
        参数：
            config: 从 app.yaml 读入的配置字典
            pool_id: 池子ID（字符串，如 'pool_1'）
        """
        super().__init__(config, pool_id)
        
        # 提取池子编号（'pool_1' -> 1）
        self.pool_num = int(pool_id.split('_')[1])
        
        # 加载 MPC 配置参数
        self.target_turbidity = config.get('target_turbidity', 1.0)
        self.dose_min = config.get('constraints', {}).get('dose_min', 0.0)
        self.dose_max = config.get('constraints', {}).get('dose_max', 70.0)
        self.dose_rate_limit = config.get('constraints', {}).get('dose_rate_limit', 5.0)
        
        # 优化器配置
        optimizer_config = config.get('optimizer', {})
        self.optimizer_type = optimizer_config.get('type', 'pso')
        self.max_iterations = optimizer_config.get('max_iterations', 100)
        self.population_size = optimizer_config.get('population_size', 50)
        self.tolerance = optimizer_config.get('tolerance', 1e-6)
        
        # 优化器超参数
        self.optimizer_hyperparams = config.get('optimizer_hyperparams', {})
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        print(f"[DosingOptimizer] 初始化完成: {pool_id}")
        print(f"  - 优化器类型: {self.optimizer_type.upper()}")
        print(f"  - 预测时域 Np: {self.prediction_horizon} 步")
        print(f"  - 控制时域 Nc: {self.control_horizon} 步")
        print(f"  - 目标浊度: {self.target_turbidity} NTU")
    
    def _create_optimizer(self):
        """创建优化器实例（使用 optimizer.py 中的工厂函数）"""
        # 构建优化器配置字典（传递给 optimizer.py）
        opt_config = {
            'optimizer': {
                'type': self.optimizer_type,
                'max_iterations': self.max_iterations,
                'population_size': self.population_size,
            },
            'control_horizon': self.control_horizon,
            'constraints': {
                'dose_min': self.dose_min,
                'dose_max': self.dose_max,
                'dose_rate_limit': self.dose_rate_limit,
            },
            'optimizer_hyperparams': self.optimizer_hyperparams,
        }
        
        return create_optimizer(opt_config)
    
    def _optimize_core(self, prepared_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        核心优化逻辑（实现基类抽象方法）
        
        参数：
            prepared_data: 包含以下字段的字典
                - predicted_turbidity: 预测的浊度序列 [Np]
                - predicted_times: 对应的时间戳列表
                - current_dose: 当前投矾量 (float)
                
        返回：
            {'pool_X': [dose_t+1, dose_t+2, ..., dose_t+Nc], ...}
        """
        # 1. 数据提取
        predicted_turbidity = prepared_data.get('predicted_turbidity')
        current_dose = prepared_data.get('current_dose')
        
        if predicted_turbidity is None:
            raise ValueError("必须提供预测的浊度数据 predicted_turbidity")
        if current_dose is None:
            raise ValueError("必须提供当前投矾量 current_dose")
        
        # 2. 定义并求解 MPC 优化问题
        optimal_du_sequence, optimal_cost = self._solve_mpc_optimization(
            predicted_turbidity, 
            current_dose
        )
        
        # 3. 将控制变化量序列转换为绝对投矾量序列
        dose_sequence = self._convert_du_to_dose(optimal_du_sequence, current_dose)
        
        print(f"[DosingOptimizer] 优化完成 - 成本: {optimal_cost:.4f}")
        print(f"  - 控制序列: {[f'{d:.2f}' for d in dose_sequence]}")
        
        return {self.pool_id: dose_sequence}
    
    def _prepare_input(
        self,
        predictions: Dict[str, Dict[str, float]],
        current_features: Dict[str, Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备优化器输入数据（重写基类方法）
        
        功能：数据提取 - 提取每个池子预测的6个出水浊度值和对应的时间
        
        参数：
            predictions: 预测器输出，格式为
                {'pool_1': {'2024-01-01 12:05': 0.95, '2024-01-01 12:10': 0.98, ...}, ...}
            current_features: 当前特征，格式为
                {'pool_1': {'current_dose': 15.0, 'ph': 7.2, ...}, ...}
            **kwargs: 其他参数
            
        返回：
            准备好的数据字典，包含：
                - predicted_turbidity: 提取的浊度预测值 [Np]
                - predicted_times: 对应的时间戳列表
                - current_dose: 当前投矾量
        """
        # 提取当前池子的预测数据
        pool_predictions = predictions.get(self.pool_id, {})
        
        if not pool_predictions:
            raise ValueError(f"未找到池子 {self.pool_id} 的预测数据")
        
        # 按时间排序并提取预测值
        sorted_times = sorted(pool_predictions.keys())
        predicted_turbidity = np.array([pool_predictions[t] for t in sorted_times])
        
        # 检查预测步数
        if len(predicted_turbidity) < self.prediction_horizon:
            raise ValueError(
                f"预测步数不足: 需要 {self.prediction_horizon} 步，"
                f"实际只有 {len(predicted_turbidity)} 步"
            )
        
        # 只取前 Np 个预测值
        predicted_turbidity = predicted_turbidity[:self.prediction_horizon]
        predicted_times = sorted_times[:self.prediction_horizon]
        
        # 提取当前投矾量
        pool_features = current_features.get(self.pool_id, {}) if current_features else {}
        current_dose = pool_features.get('current_dose')
        
        if current_dose is None:
            # 如果没有提供，从 kwargs 中获取
            current_dose = kwargs.get('current_dose')
        
        if current_dose is None:
            raise ValueError("必须提供 current_dose（当前投矾量）")
        
        print(f"[DosingOptimizer] 数据提取完成:")
        print(f"  - 预测浊度: {[f'{v:.3f}' for v in predicted_turbidity]}")
        print(f"  - 当前投矾: {current_dose:.2f} mg/L")
        
        return {
            'predicted_turbidity': predicted_turbidity,
            'predicted_times': predicted_times,
            'current_dose': float(current_dose),
        }
    
    def _solve_mpc_optimization(
        self, 
        predicted_turbidity: np.ndarray, 
        current_dose: float
    ) -> tuple:
        """
        定义并求解 MPC 优化问题
        
        功能：
            1. 定义 MPC 目标函数
            2. 设定约束条件
            3. 使用 optimizer.py 中的求解方法进行求解
        
        参数：
            predicted_turbidity: 预测的浊度序列 [Np]
            current_dose: 当前投矾量
            
        返回：
            optimal_du_sequence: 最优控制变化量序列 [Nc]
            optimal_cost: 最优目标函数值
        """
        # 定义 MPC 目标函数（需要匹配 optimizer.py 的接口）
        def mpc_objective(du_sequence: np.ndarray, initial_dose: float) -> float:
            """
            MPC 目标函数（兼容 optimizer.py 的接口）
            
            目标函数形式:
                J = Σ(k=1 to Np) ||y(t+k|t) - y_target||² + λ * Σ(k=1 to Nc) ||Δu(t+k)||²
                
            其中：
                - y(t+k|t): t时刻预测的t+k时刻出水浊度
                - y_target: 目标浊度（设定值）
                - Δu(t+k): t+k时刻的控制变化量
                - λ: 控制惩罚权重
                
            参数：
                du_sequence: 控制变化量序列 [Nc]
                initial_dose: 初始投矾量（当前投矾量）
            """
            return self._compute_mpc_cost(
                du_sequence, 
                predicted_turbidity, 
                initial_dose
            )
        
        # 调用优化器求解（optimizer.py 中的方法）
        # 优化器会自动处理约束条件（dose_min, dose_max, dose_rate_limit）
        optimal_du_sequence, optimal_cost = self.optimizer.optimize(
            objective_func=mpc_objective,
            initial_dose=current_dose
        )
        
        return optimal_du_sequence, optimal_cost
    
    def _compute_mpc_cost(
        self,
        du_sequence: np.ndarray,
        predicted_turbidity: np.ndarray,
        current_dose: float
    ) -> float:
        """
        计算 MPC 目标函数值
        
        参数：
            du_sequence: 控制变化量序列 [Nc]
            predicted_turbidity: 预测的浊度序列 [Np]
            current_dose: 当前投矾量
            
        返回：
            目标函数值（代价）
        """
        Np = self.prediction_horizon
        Nc = self.control_horizon
        
        # 1. 跟踪误差项：Σ ||y(t+k) - y_target||²
        tracking_error = np.sum((predicted_turbidity - self.target_turbidity) ** 2)
        
        # 2. 控制变化惩罚项：λ * Σ ||Δu(t+k)||²
        control_penalty = self.lambda_du * np.sum(du_sequence ** 2)
        
        # 3. 约束惩罚（软约束）
        constraint_penalty = 0.0
        
        # 检查控制序列是否违反约束
        dose_tmp = current_dose
        for k in range(Nc):
            # 累积投矾量
            dose_tmp = dose_tmp + du_sequence[k]
            
            # 投矾量范围约束惩罚
            if dose_tmp < self.dose_min:
                constraint_penalty += 1e3 * (self.dose_min - dose_tmp) ** 2
            elif dose_tmp > self.dose_max:
                constraint_penalty += 1e3 * (dose_tmp - self.dose_max) ** 2
            
            # 变化率约束惩罚
            if abs(du_sequence[k]) > self.dose_rate_limit:
                constraint_penalty += 1e3 * (abs(du_sequence[k]) - self.dose_rate_limit) ** 2
        
        # 总代价
        total_cost = tracking_error + control_penalty + constraint_penalty
        
        return total_cost
    
    def _convert_du_to_dose(
        self, 
        du_sequence: np.ndarray, 
        current_dose: float
    ) -> List[float]:
        """
        将控制变化量序列转换为绝对投矾量序列
        
        功能：输出格式化 - 将相对变化量转换为实际投矾量
        
        参数：
            du_sequence: 控制变化量序列 [Nc]，Δu(t+k) = u(t+k) - u(t+k-1)
            current_dose: 当前投矾量 u(t)
            
        返回：
            dose_sequence: 绝对投矾量序列 [Nc]，[u(t+1), u(t+2), ..., u(t+Nc)]
        """
        dose_sequence = []
        dose_accumulate = current_dose
        
        for du in du_sequence:
            # 累积变化量
            dose_accumulate = dose_accumulate + du
            
            # 应用硬约束（确保在范围内）
            dose_accumulate = np.clip(dose_accumulate, self.dose_min, self.dose_max)
            
            # 保留2位小数
            dose_sequence.append(round(float(dose_accumulate), 2))
        
        return dose_sequence


def create_dosing_optimizer(pool_id: str, config_path: str = None) -> DosingOptimizer:
    """
    创建投药优化器实例（辅助函数）
    
    功能：
        - 加载配置文件（使用 config_loader.py）
        - 创建并初始化 DosingOptimizer 实例
    
    参数：
        pool_id: 池子ID（如 'pool_1', 'pool_2', ...）
        config_path: 配置文件路径
            - 如果为 None，默认加载 configs/app.yaml
            - 支持绝对路径或相对路径
        
    返回：
        DosingOptimizer 实例
        
    使用示例：
        >>> optimizer = create_dosing_optimizer('pool_1')
        >>> result = optimizer.optimize(
        ...     predictions={'pool_1': {'2024-01-01 12:05': 0.95, ...}},
        ...     current_features={'pool_1': {'current_dose': 15.0}}
        ... )
    """
    # 加载配置（参考 config_loader.py 和 app.yaml）
    config = load_config(config_path)
    
    # 创建优化器实例
    return DosingOptimizer(config=config, pool_id=pool_id)


def create_multi_pool_optimizers(
    pool_ids: List[str], 
    config_path: str = None
) -> Dict[str, DosingOptimizer]:
    """
    批量创建多个池子的优化器实例
    
    参数：
        pool_ids: 池子ID列表（如 ['pool_1', 'pool_2', 'pool_3', 'pool_4']）
        config_path: 配置文件路径，默认为 configs/app.yaml
        
    返回：
        优化器字典 {'pool_1': optimizer1, 'pool_2': optimizer2, ...}
        
    使用示例：
        >>> optimizers = create_multi_pool_optimizers(['pool_1', 'pool_2'])
        >>> for pool_id, optimizer in optimizers.items():
        ...     result = optimizer.optimize(predictions, current_features)
    """
    # 只加载一次配置
    config = load_config(config_path)
    
    # 为每个池子创建优化器
    optimizers = {}
    for pool_id in pool_ids:
        optimizers[pool_id] = DosingOptimizer(config=config, pool_id=pool_id)
    
    return optimizers