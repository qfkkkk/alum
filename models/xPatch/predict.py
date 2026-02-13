"""
出水浊度预测 - 推理脚本

加载训练好的模型，输入最近 60 个时间步的数据，预测未来 6 步出水浊度。

用法:
    # 从 CSV 文件中取最后 60 行作为输入
    python predict.py --pool 1 --csv ../train_data.csv

    # 指定起始行（从该行开始取 60 行）
    python predict.py --pool 1 --csv ../train_data.csv --start-row 1000

    # 从 JSON 文件输入
    python predict.py --pool 1 --json input.json
"""

import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from .dataset import load_and_preprocess, POOL_FEATURES, TURB_CHUSHUI_IDX
from .models.xPatch import Model

def load_model(pool_id: int, output_dir: str = 'output', device=None):
    """
    加载指定池子的训练模型和 scaler。

    Returns:
        model, scaler, model_cfg, use_diff
    """
    pool_dir = os.path.join(output_dir, f'pool_{pool_id}')
    ckpt_path = os.path.join(pool_dir, 'best_model.pt')
    scaler_path = os.path.join(pool_dir, 'scaler.pkl')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型: {ckpt_path}")

    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = SimpleNamespace(**checkpoint['model_config'])
    use_diff = checkpoint['train_config']['use_diff']

    model = Model(model_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载 scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler, model_cfg, use_diff


def predict(model, scaler, model_cfg, use_diff: bool,
            input_data: np.ndarray, device=None):
    """
    执行预测。

    Args:
        model: 训练好的 xPatch 模型
        scaler: StandardScaler
        model_cfg: 模型配置
        use_diff: 是否使用差分模式
        input_data: 原始空间的输入数据, shape [seq_len, n_features]
        device: torch device

    Returns:
        predictions: 原始空间的预测值, shape [pred_len]
    """
    seq_len = model_cfg.seq_len
    turb_idx = TURB_CHUSHUI_IDX

    if len(input_data) != seq_len:
        raise ValueError(f"输入数据长度 {len(input_data)} != seq_len {seq_len}")

    # 标准化
    input_scaled = scaler.transform(input_data).astype(np.float32)

    # 最后一步的 turb_chushui（scaled，用于反差分）
    last_val_scaled = input_scaled[-1, turb_idx]

    # 推理
    x = torch.from_numpy(input_scaled).unsqueeze(0).to(device)  # [1, seq_len, n_features]
    with torch.no_grad():
        pred = model(x)  # [1, pred_len, n_features]

    pred_turb_scaled = pred[0, :, turb_idx].cpu().numpy()  # [pred_len]

    # 反差分
    if use_diff:
        pred_abs_scaled = pred_turb_scaled + last_val_scaled
    else:
        pred_abs_scaled = pred_turb_scaled

    # 反标准化
    mean_val = scaler.mean_[turb_idx]
    std_val = scaler.scale_[turb_idx]
    predictions = pred_abs_scaled * std_val + mean_val

    return predictions


def predict_from_csv(pool_id: int, csv_path: str, start_row: int = None,
                     output_dir: str = 'output', device=None):
    """从 CSV 数据预测"""

    model, scaler, model_cfg, use_diff = load_model(pool_id, output_dir, device)
    seq_len = model_cfg.seq_len

    # 加载数据
    df_pool, feature_names = load_and_preprocess(csv_path, pool_id)
    data = df_pool.values

    if start_row is None:
        start_row = len(data) - seq_len  # 取最后 seq_len 行

    if start_row < 0 or start_row + seq_len > len(data):
        raise ValueError(
            f"start_row={start_row} 无效，数据总行数={len(data)}，"
            f"需要从 start_row 开始取 {seq_len} 行"
        )

    input_data = data[start_row: start_row + seq_len]  # [seq_len, n_features]
    predictions = predict(model, scaler, model_cfg, use_diff, input_data, device)

    return predictions, input_data, feature_names


def predict_from_json(pool_id: int, json_path: str,
                      output_dir: str = 'output', device=None):
    """
    从 JSON 文件预测。

    JSON 格式:
    {
        "data": [
            {"dose": 15.3, "turb_chushui": 0.31, "turb_jinshui": 54.8, "flow": 4098.5, "pH": 5.99, "temp_shuimian": 15.26},
            ...  // 共 60 行
        ]
    }
    """
    model, scaler, model_cfg, use_diff = load_model(pool_id, output_dir, device)

    with open(json_path, 'r') as f:
        input_json = json.load(f)

    rows = input_json['data']
    feature_keys = ['dose', 'turb_chushui', 'turb_jinshui', 'flow', 'pH', 'temp_shuimian']
    input_data = np.array([[row[k] for k in feature_keys] for row in rows], dtype=np.float32)

    predictions = predict(model, scaler, model_cfg, use_diff, input_data, device)
    return predictions


def main():
    parser = argparse.ArgumentParser(description='xPatch 出水浊度预测推理')
    parser.add_argument('--pool', type=int, required=True, choices=[1, 2, 3, 4],
                        help='池子编号')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV 数据路径（从中取输入数据）')
    parser.add_argument('--json', type=str, default=None,
                        help='JSON 输入文件路径')
    parser.add_argument('--start-row', type=int, default=None,
                        help='CSV 中的起始行号（默认取最后 60 行）')
    parser.add_argument('--output', type=str, default='output',
                        help='模型目录')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.json:
        predictions = predict_from_json(args.pool, args.json, args.output, device)
        print(f"\n池子 {args.pool} 预测结果 (未来 {len(predictions)} 步出水浊度):")
        for i, val in enumerate(predictions):
            print(f"  t+{i+1}: {val:.4f}")

    elif args.csv:
        predictions, input_data, feature_names = predict_from_csv(
            args.pool, args.csv, args.start_row, args.output, device
        )

        turb_idx = TURB_CHUSHUI_IDX
        last_turb = input_data[-1, turb_idx]

        print(f"\n池子 {args.pool} 预测结果:")
        print(f"  当前出水浊度 (t):  {last_turb:.4f}")
        print(f"  预测 (未来 {len(predictions)} 步):")
        for i, val in enumerate(predictions):
            delta = val - last_turb
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"    t+{i+1}: {val:.4f}  ({arrow} {delta:+.4f})")
    else:
        print("请指定 --csv 或 --json 输入数据")
        return


if __name__ == '__main__':
    main()