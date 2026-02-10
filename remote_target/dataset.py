"""
出水浊度预测 - 数据集与数据预处理

滑动窗口 Dataset:
  输入: [seq_len, n_features]  (30 个时间步 × 6 特征)
  目标: [pred_len, n_features] (6 个时间步，全通道差分值)
  last_vals: [n_features]       (最后输入步的值，用于反差分)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle


# 每个池子对应的特征列
POOL_FEATURES = {
    1: ['dose_1', 'turb_chushui_1', 'turb_jinshui_1', 'flow_1', 'pH', 'temp_shuimian'],
    2: ['dose_2', 'turb_chushui_2', 'turb_jinshui_2', 'flow_2', 'pH', 'temp_shuimian'],
    3: ['dose_3', 'turb_chushui_3', 'turb_jinshui_3', 'flow_3', 'pH', 'temp_shuimian'],
    4: ['dose_4', 'turb_chushui_4', 'turb_jinshui_4', 'flow_4', 'pH', 'temp_shuimian'],
}

# turb_chushui 在每个池子的特征列中的索引（始终为第 1 列，0-indexed）
TURB_CHUSHUI_IDX = 1


def load_and_preprocess(csv_path: str, pool_id: int):
    """
    加载 CSV 并提取指定池子的特征。

    Returns:
        df_pool: 处理后的 DataFrame（仅包含 6 个特征列）
        feature_names: 特征名称列表
    """
    df = pd.read_csv(csv_path, parse_dates=['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)

    feature_names = POOL_FEATURES[pool_id]
    df_pool = df[feature_names].copy()

    # 缺失值处理: 前向填充 → 线性插值 → 剩余用 0 填充
    df_pool = df_pool.ffill().interpolate(method='linear').fillna(0)

    return df_pool, feature_names


class TurbidityDataset(Dataset):
    """
    滑动窗口数据集，支持差分目标（全通道）。

    xPatch 使用 Channel Independence，输出 [Batch, Pred_Len, Enc_In]，
    所以目标也是全通道的，训练时对所有通道计算 loss。

    Args:
        data: numpy array, shape [T, n_features]
        seq_len: 输入序列长度 (默认 30)
        pred_len: 预测长度 (默认 6)
        use_diff: 是否使用差分目标
    """

    def __init__(self, data: np.ndarray, seq_len: int = 30, pred_len: int = 6,
                 use_diff: bool = True):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_diff = use_diff

        # 有效样本数
        self.n_samples = len(data) - seq_len - pred_len + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"数据长度 {len(data)} 不足以创建样本 "
                f"(需要至少 {seq_len + pred_len} 行)"
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 输入窗口: [idx, idx + seq_len)
        x = self.data[idx: idx + self.seq_len]  # [seq_len, n_features]

        # 目标窗口: [idx + seq_len, idx + seq_len + pred_len) 全通道
        y_raw = self.data[
            idx + self.seq_len: idx + self.seq_len + self.pred_len
        ]  # [pred_len, n_features]

        # 最后输入步的值（全通道）
        last_vals = self.data[idx + self.seq_len - 1]  # [n_features]

        if self.use_diff:
            # 差分目标: y_{t+k} - y_t（每个通道独立差分）
            y = y_raw - last_vals[np.newaxis, :]
        else:
            y = y_raw

        return (
            torch.from_numpy(x),           # [seq_len, n_features]
            torch.from_numpy(y),           # [pred_len, n_features]
            torch.from_numpy(last_vals),   # [n_features]
        )


def create_datasets(csv_path: str, pool_id: int, seq_len: int = 30,
                    pred_len: int = 6, use_diff: bool = True,
                    train_ratio: float = 0.7, val_ratio: float = 0.15,
                    scaler_path: str = None):
    """
    创建 train/val/test 数据集，按时间顺序切分。

    Returns:
        train_ds, val_ds, test_ds, scaler, feature_names
    """
    df_pool, feature_names = load_and_preprocess(csv_path, pool_id)
    data = df_pool.values  # [T, n_features]

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 用训练集拟合 scaler
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # 保存 scaler
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    train_ds = TurbidityDataset(train_scaled, seq_len, pred_len, use_diff)
    val_ds = TurbidityDataset(val_scaled, seq_len, pred_len, use_diff)
    test_ds = TurbidityDataset(test_scaled, seq_len, pred_len, use_diff)

    return train_ds, val_ds, test_ds, scaler, feature_names
