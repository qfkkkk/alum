"""
MPC 控制器实现
"""
import os
import sys
import pickle
import numpy as np
import torch
from typing import List, Tuple

# 添加 predictions 模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'predictions'))

from config import MPCConfig, PoolState
from optimizer import create_optimizer
from models.xPatch import Model
from train import ModelConfig


class PoolPredictor:
    """单个池子的预测器"""
    
    def __init__(self, pool_id: int, models_dir: str, device=None):
        """
        初始化预测器
        
        Args:
            pool_id: 池子编号 (1-4)
            models_dir: 模型目录
            device: torch device
        """
        self.pool_id = pool_id
        self.device = device if device else torch.device('cpu')
        
        # 加载模型和scaler
        pool_dir = os.path.join(models_dir, f'pool_{pool_id}')
        ckpt_path = os.path.join(pool_dir, 'best_model.pt')
        scaler_path = os.path.join(pool_dir, 'scaler.pkl')
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
        
        # 加载模型
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_cfg = ModelConfig(**checkpoint['model_config'])
        
        self.model = Model(model_cfg).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.seq_len = model_cfg.seq_len
        self.pred_len = model_cfg.pred_len
        self.turb_idx = 1  
        
    def predict(self, history_data: np.ndarray, future_doses: np.ndarray) -> np.ndarray:
        """
        预测未来出水浊度
        
        Args:
            history_data: 历史数据 [seq_len, 6]
                特征顺序: [dose, turb_chushui, turb_jinshui, flow, pH, temp_shuimian]
            future_doses: 未来投矾量序列 [pred_len]
            
        Returns:
            predictions: 预测的出水浊度 [pred_len]
        """
        if len(history_data) != self.seq_len:
            raise ValueError(f"历史数据长度 {len(history_data)} != seq_len {self.seq_len}")
        
        if len(future_doses) != self.pred_len:
            raise ValueError(f"未来投矾量长度 {len(future_doses)} != pred_len {self.pred_len}")
        
        # 准备输入数据
        input_data = history_data.copy()
        
        # 标准化
        input_scaled = self.scaler.transform(input_data).astype(np.float32)
        
        # 直接预测（一次性输出所有未来时刻）
        x = torch.from_numpy(input_scaled).unsqueeze(0).to(self.device)
        
        # 模型预测
        with torch.no_grad():
            pred = self.model(x)  # [1, pred_len, 6]
        
        pred_scaled = pred[0, :, :].cpu().numpy()  # [pred_len, 6]
        
        # 提取出水浊度预测并反标准化
        predictions = []
        for step in range(self.pred_len):
            pred_turb_scaled = pred_scaled[step, self.turb_idx]
            
            # 反标准化浊度
            turb_pred = (pred_turb_scaled * self.scaler.scale_[self.turb_idx] + 
                        self.scaler.mean_[self.turb_idx])
            predictions.append(turb_pred)
        
        return np.array(predictions)


class MPCController:
    """MPC 控制器"""
    
    def __init__(self, config: MPCConfig, device=None):
        """
        初始化MPC控制器
        
        Args:
            config: MPC配置
            device: torch device
        """
        self.config = config
        self.device = device if device else torch.device('cpu')
        
        # 初始化优化器
        self.optimizer = create_optimizer(config)
        
        # 加载所有池子的预测器
        self.predictors = {}
        for pool_id in [1, 2, 3, 4]:
            self.predictors[pool_id] = PoolPredictor(
                pool_id, config.models_dir, self.device
            )
        
        print(f"[成功] MPC控制器初始化完成")
        print(f"  - 优化器: {config.optimizer_type.upper()}")
        print(f"  - 预测时域: {config.prediction_horizon} 步 ({config.prediction_horizon * config.time_step:.0f} 分钟)")
        print(f"  - 控制时域: {config.control_horizon} 步 ({config.control_horizon * config.time_step:.0f} 分钟)")
        print(f"  - 目标浊度: {config.target_turbidity} NTU")
        
    def _objective_function(self, du_sequence: np.ndarray, pool_state: PoolState) -> float:
        """
        MPC 目标函数

        目标函数: min Σ||y(t+k) - y_set||² + λ Σ||Δu(t+k)||²

        Args:
            du_sequence: 控制变化量序列 [Nc]，每个元素是相对于**前一步**的变化量
            pool_state: 池子状态
            
        Returns:
            目标函数值
        """
        Np = self.config.prediction_horizon
        Nc = self.config.control_horizon
        y_set = self.config.target_turbidity
        lambda_du = self.config.lambda_du

        # 计算未来投矾量序列
        future_doses = np.zeros(Np)
        current_dose = pool_state.current_dose

        for k in range(Np):
            if k < Nc:
                # ✅ 每步的变化量是相对于前一步的
                current_dose = current_dose + du_sequence[k]
                # 约束检查
                current_dose = np.clip(current_dose, 
                                        self.config.dose_min, 
                                        self.config.dose_max)
            # k >= Nc 时，保持最后一个控制量不变
            future_doses[k] = current_dose

        # 预测未来浊度
        predictor = self.predictors[pool_state.pool_id]
        try:
            y_pred = predictor.predict(pool_state.history_data, future_doses)
        except Exception as e:
            print(f"[警告] 预测失败: {str(e)}")
            return 1e6

        # 计算目标函数
        tracking_error = np.sum((y_pred - y_set) ** 2)
        control_penalty = lambda_du * np.sum(du_sequence ** 2)

        total_cost = tracking_error + control_penalty

        return total_cost
    
    def compute_control(self, pool_state: PoolState) -> Tuple[float, np.ndarray, dict]:
        """
        计算最优控制量
        
        Args:
            pool_state: 池子状态
            
        Returns:
            optimal_dose: 下一步的最优投矾量
            optimal_sequence: 完整的最优控制序列 [Nc]
            info: 优化信息字典
        """
        # 定义目标函数（固定pool_state）
        def obj_func(du_seq, initial_dose):
            # 临时更新 pool_state 的 current_dose
            temp_state = PoolState(
                pool_id=pool_state.pool_id,
                current_dose=initial_dose,
                history_data=pool_state.history_data
            )
            return self._objective_function(du_seq, temp_state)
        
        # 执行优化
        optimal_du_sequence, optimal_cost = self.optimizer.optimize(
            obj_func, pool_state.current_dose
        )
        
        # 计算下一步投矾量（只应用第一个控制变化量）
        optimal_dose = pool_state.current_dose + optimal_du_sequence[0]
        optimal_dose = np.clip(optimal_dose, 
                              self.config.dose_min, 
                              self.config.dose_max)
        
        # 预测信息
        future_doses = np.zeros(self.config.prediction_horizon)
        current_dose_tmp = pool_state.current_dose
        for k in range(self.config.prediction_horizon):
            if k < self.config.control_horizon:
                current_dose_tmp += optimal_du_sequence[k]
                current_dose_tmp = np.clip(current_dose_tmp,
                                          self.config.dose_min,
                                          self.config.dose_max)
            future_doses[k] = current_dose_tmp
        
        predictor = self.predictors[pool_state.pool_id]
        predicted_turbidity = predictor.predict(pool_state.history_data, future_doses)
        
        info = {
            'optimal_cost': optimal_cost,
            'optimal_du_sequence': optimal_du_sequence,
            'optimal_dose_sequence': future_doses,
            'predicted_turbidity': predicted_turbidity,
            'current_dose': pool_state.current_dose,
            'next_dose': optimal_dose,
            'dose_change': optimal_dose - pool_state.current_dose,
        }
        
        return optimal_dose, optimal_du_sequence, info
    
    def control_step(self, pool_state: PoolState, verbose: bool = True) -> dict:
        """
        执行一步MPC控制
        
        Args:
            pool_state: 池子状态
            verbose: 是否打印详细信息
            
        Returns:
            控制结果字典
        """
        optimal_dose, optimal_sequence, info = self.compute_control(pool_state)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"池子 {pool_state.pool_id} MPC 控制结果")
            print(f"{'='*60}")
            print(f"当前投矾量:    {pool_state.current_dose:.2f} mg/L")
            print(f"最优投矾量:    {optimal_dose:.2f} mg/L")
            print(f"投矾变化量:    {info['dose_change']:+.2f} mg/L")
            print(f"目标函数值:    {info['optimal_cost']:.4f}")
            print(f"\n预测未来浊度:")
            for k, turb in enumerate(info['predicted_turbidity']):
                time_min = (k + 1) * self.config.time_step
                status = "[达标]" if turb <= self.config.target_turbidity else "[超标]"
                print(f"  t+{k+1} ({time_min:.0f}min): {turb:.4f} NTU {status}")
            print(f"{'='*60}")
        
        return info


# 便捷函数
def create_mpc_controller(config: MPCConfig = None, device=None) -> MPCController:
    """
    创建MPC控制器
    
    Args:
        config: MPC配置，如果为None则使用默认配置
        device: torch device
        
    Returns:
        MPC控制器实例
    """
    if config is None:
        config = MPCConfig()
    
    return MPCController(config, device)
