# -*- encoding: utf-8 -*-
"""
xPatch 出水浊度预测器
功能：使用 xPatch 模型预测未来出水浊度（单池）
输入：60个时间点 × 6个特征
输出：6个时间点的预测浊度值

继承：BasePredictor
"""
import pickle
import numpy as np
import torch
from typing import Dict
from pathlib import Path
from dataclasses import dataclass

from .base_predictor import BasePredictor


class TurbidityPredictor(BasePredictor):
    """
    xPatch 出水浊度预测类（单池）

    功能说明：
        一个实例负责一个池子的浊度预测。
        构造时自动加载对应池子的模型权重和 scaler。
        只实现 _load_model() 和 _infer()，通用流程由 BasePredictor.predict() 完成。

    使用示例：
        predictor = TurbidityPredictor(pool_id=1, config=yaml_config)
        predictions = predictor.predict(input_data)  # [60, 6] -> [6]
    """

    def __init__(self, pool_id: int, config: dict):
        """
        初始化 xPatch 浊度预测器

        参数：
            pool_id: 池子编号 (1, 2, 3, 4)
            config: 从 app.yaml 读入的配置字典
        """
        super().__init__(pool_id, config)
        self.device = self._resolve_device(config.get('device', 'auto'))
        self.use_diff = False

        self._load_model()

    def _resolve_device(self, device_str: str) -> torch.device:
        """解析设备配置"""
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_str)

    def _load_model(self):
        """
        加载 xPatch 预训练模型和 scaler

        说明：
            从 checkpoints/{model_arch}/pool_{id}/ 加载：
            - best_model.pt: 包含 model_state_dict, model_config, train_config
            - scaler.pkl: StandardScaler
        """
        arch = self.config.get('model_arch', 'xPatch')
        base_dir = self.config.get('checkpoint_base_dir', 'checkpoints')

        # 构建路径（相对于项目根目录）
        project_root = Path(__file__).parent.parent
        pool_dir = project_root / base_dir / arch / f'pool_{self.pool_id}'
        ckpt_path = pool_dir / 'best_model.pt'
        scaler_path = pool_dir / 'scaler.pkl'

        if not ckpt_path.exists():
            raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"未找到 scaler: {scaler_path}")

        # 加载 checkpoint
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # 重建模型配置
        from types import SimpleNamespace
        model_cfg = SimpleNamespace(**checkpoint['model_config'])
        self.use_diff = checkpoint['train_config']['use_diff']
        self.model_config = model_cfg

        # 导入并实例化模型
        from models.xPatch.models.xPatch import Model
        self.model = Model(model_cfg).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 加载 scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def _infer(self, scaled_input: np.ndarray) -> np.ndarray:
        """
        xPatch 专属推理逻辑

        参数：
            scaled_input: 标准化后的输入数据, shape [seq_len, n_features]

        返回：
            np.ndarray: 标准化空间下的目标特征预测值, shape [pred_len]

        说明：
            包含 xPatch 特有的反差分处理。
            返回值仍在标准化空间，反标准化由 BasePredictor.predict() 完成。
        """
        # 记录最后一步的 target 值（用于反差分）
        last_val_scaled = scaled_input[-1, self.target_idx]

        # 推理
        x = torch.from_numpy(scaled_input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x)  # [1, pred_len, n_features]

        pred_target_scaled = pred[0, :, self.target_idx].cpu().numpy()

        # 反差分（xPatch 专属）
        if self.use_diff:
            pred_target_scaled = pred_target_scaled + last_val_scaled

        return pred_target_scaled
