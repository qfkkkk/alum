"""
MPC 控制器配置
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class MPCConfig:
    """MPC 控制器配置"""
    
    # 预测与控制时域
    prediction_horizon: int = 6     # Np: 预测时域（步数），每步5分钟
    control_horizon: int = 5        # Nc: 控制时域（步数）
    time_step: float = 5.0          # 每步时间（分钟）
    
    # 目标设定
    target_turbidity: float = 0.5   # y_set: 目标出水浊度 (NTU)
    
    # 目标函数权重
    lambda_du: float = 0.005         # λ: 控制变化量的权重系数
    
    # 约束条件
    dose_min: float = 0           # 投矾量下限 (mg/L)
    dose_max: float = 70          # 投矾量上限 (mg/L)
    dose_rate_limit: float = 10.0    # 单步投矾变化量上限 (mg/L)
    
    # 优化器选择
    optimizer_type: Literal['pso', 'sa', 'de'] = 'pso'  # pso: 粒子群, sa: 模拟退火, de: 差分进化
    
    # 优化器超参数
    max_iterations: int = 100       # 最大迭代次数
    population_size: int = 50       # 种群大小（PSO/DE）
    tolerance: float = 1e-6         # 收敛容差
    
    # PSO 特定参数
    pso_w: float = 0.7              # 惯性权重
    pso_c1: float = 1.5             # 个体学习因子
    pso_c2: float = 1.5             # 社会学习因子
    
    # SA 特定参数
    sa_temp_init: float = 100.0     # 初始温度
    sa_temp_min: float = 1e-3       # 最小温度
    sa_alpha: float = 0.95          # 降温系数
    
    # DE 特定参数
    de_f: float = 0.8               # 缩放因子
    de_cr: float = 0.9              # 交叉概率
    
    # 模型路径
    models_dir: str = '../models'   # 模型权重和scaler目录


@dataclass
class PoolState:
    """单个池子的状态"""
    pool_id: int                    # 池子编号 (1-4)
    current_dose: float             # 当前投矾量
    history_data: list              # 历史数据 [seq_len, 6]，最近60步
    # history_data 特征顺序: [dose, turb_chushui, turb_jinshui, flow, pH, temp_shuimian]
