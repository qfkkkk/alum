"""
出水浊度预测 - 评估与可视化脚本

功能:
  1. 在测试集上计算 MAE、RMSE、R²、方向准确率
  2. 从测试集中抽样若干样本，绘制:
     - 30 点历史输入（turb_chushui）
     - 6 点预测 vs 实际对比

用法:
    python evaluate.py --pool 1
    python evaluate.py --pool 1 --n-samples 8   # 抽样 8 个样本绘图
    python evaluate.py --all                     # 评估所有池子
"""

import os
import argparse
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset import (
    create_datasets, load_and_preprocess,
    TURB_CHUSHUI_IDX, POOL_FEATURES,
)
from models.xPatch import Model
from train import ModelConfig


# ============================================================
# 指标计算
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    计算评估指标。

    Args:
        y_true, y_pred: shape [N, pred_len]

    Returns:
        dict 包含各步和总体指标
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # 方向准确率: 预测的变化方向与实际是否一致
    # 跟上一步比较 (first step vs last input is implicitly 0 in diff mode)
    pred_diff = np.diff(y_pred, axis=1, prepend=0)
    true_diff = np.diff(y_true, axis=1, prepend=0)
    direction_acc = np.mean(np.sign(pred_diff) == np.sign(true_diff))

    # 按步指标
    per_step = {}
    for step in range(y_true.shape[1]):
        step_mae = np.mean(np.abs(y_true[:, step] - y_pred[:, step]))
        step_rmse = np.sqrt(np.mean((y_true[:, step] - y_pred[:, step]) ** 2))
        per_step[f't+{step+1}'] = {'MAE': round(float(step_mae), 6),
                                     'RMSE': round(float(step_rmse), 6)}

    return {
        'MAE': round(float(mae), 6),
        'RMSE': round(float(rmse), 6),
        'R2': round(float(r2), 6),
        'Direction_Accuracy': round(float(direction_acc), 4),
        'per_step': per_step,
    }


# ============================================================
# 反差分 + 反标准化
# ============================================================

def inverse_transform_predictions(pred_scaled_diff, last_vals_scaled, scaler,
                                  use_diff=True):
    """
    将模型输出转换回原始空间。

    Args:
        pred_scaled_diff: [N, pred_len] 模型预测的 turb_chushui 通道（scaled, 可能是差分）
        last_vals_scaled: [N] 最后输入步的 turb_chushui 值（scaled）
        scaler: StandardScaler
        use_diff: 是否使用了差分

    Returns:
        pred_original: [N, pred_len] 原始空间的预测值
    """
    turb_idx = TURB_CHUSHUI_IDX
    n_features = scaler.n_features_in_

    if use_diff:
        # 反差分: pred_abs_scaled = last_val + pred_diff
        pred_abs_scaled = pred_scaled_diff + last_vals_scaled[:, np.newaxis]
    else:
        pred_abs_scaled = pred_scaled_diff

    # 反标准化 (只对 turb_chushui 通道)
    mean_val = scaler.mean_[turb_idx]
    std_val = scaler.scale_[turb_idx]

    pred_original = pred_abs_scaled * std_val + mean_val
    return pred_original


def inverse_transform_values(values_scaled, scaler, feature_idx):
    """反标准化单个特征"""
    mean_val = scaler.mean_[feature_idx]
    std_val = scaler.scale_[feature_idx]
    return values_scaled * std_val + mean_val


# ============================================================
# 评估单个池子
# ============================================================

def evaluate_pool(pool_id: int, csv_path: str, output_dir: str,
                  n_samples: int = 6, device: torch.device = None):
    """评估单个池子，输出指标和图表"""

    pool_dir = os.path.join(output_dir, f'pool_{pool_id}')
    ckpt_path = os.path.join(pool_dir, 'best_model.pt')
    scaler_path = os.path.join(pool_dir, 'scaler.pkl')

    if not os.path.exists(ckpt_path):
        print(f"  ❌ 池子 {pool_id}: 未找到模型 {ckpt_path}")
        return

    print(f"\n{'='*60}")
    print(f"  评估池子 {pool_id}")
    print(f"{'='*60}")

    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg_dict = checkpoint['model_config']
    train_info = checkpoint['train_config']
    use_diff = train_info['use_diff']

    # 重建模型配置
    model_cfg = ModelConfig(**model_cfg_dict)
    model = Model(model_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载 scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 创建测试数据集
    _, _, test_ds, _, feature_names = create_datasets(
        csv_path=csv_path,
        pool_id=pool_id,
        seq_len=model_cfg.seq_len,
        pred_len=model_cfg.pred_len,
        use_diff=use_diff,
    )

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # 收集预测
    all_preds = []
    all_targets = []
    all_last_vals = []
    all_inputs = []

    turb_idx = TURB_CHUSHUI_IDX

    with torch.no_grad():
        for x, y, last_vals in test_loader:
            x = x.to(device)
            pred = model(x)  # [B, pred_len, n_features]

            # 只取 turb_chushui 通道
            pred_turb = pred[:, :, turb_idx].cpu().numpy()   # [B, pred_len]
            y_turb = y[:, :, turb_idx].numpy()               # [B, pred_len]
            lv = last_vals[:, turb_idx].numpy()               # [B]
            inp_turb = x[:, :, turb_idx].cpu().numpy()        # [B, seq_len]

            all_preds.append(pred_turb)
            all_targets.append(y_turb)
            all_last_vals.append(lv)
            all_inputs.append(inp_turb)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_last_vals = np.concatenate(all_last_vals, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)

    # 转回原始空间
    pred_original = inverse_transform_predictions(
        all_preds, all_last_vals, scaler, use_diff=use_diff
    )
    target_original = inverse_transform_predictions(
        all_targets, all_last_vals, scaler, use_diff=use_diff
    )
    input_original = inverse_transform_values(all_inputs, scaler, turb_idx)
    last_vals_original = inverse_transform_values(all_last_vals, scaler, turb_idx)

    # 计算指标
    metrics = compute_metrics(target_original, pred_original)
    print(f"\n  📊 总体指标:")
    print(f"     MAE:  {metrics['MAE']:.4f}")
    print(f"     RMSE: {metrics['RMSE']:.4f}")
    print(f"     R²:   {metrics['R2']:.4f}")
    print(f"     方向准确率: {metrics['Direction_Accuracy']:.2%}")
    print(f"\n  📊 分步指标:")
    for step_name, step_metrics in metrics['per_step'].items():
        print(f"     {step_name}: MAE={step_metrics['MAE']:.4f}, RMSE={step_metrics['RMSE']:.4f}")

    # 保存指标
    fig_dir = os.path.join(pool_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(pool_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---- 可视化 ----

    # 1. 分步 MAE/RMSE 柱状图
    _plot_per_step_metrics(metrics, pool_id, fig_dir)

    # 2. 抽样样本: 30 点历史 + 6 点预测 vs 实际
    _plot_sample_predictions(
        input_original, pred_original, target_original,
        pool_id, n_samples, fig_dir,
        seq_len=model_cfg.seq_len, pred_len=model_cfg.pred_len,
    )

    # 3. 全测试集散点图
    _plot_scatter(target_original, pred_original, pool_id, fig_dir)

    print(f"\n  📁 图表已保存至: {fig_dir}")
    return metrics


# ============================================================
# 可视化函数
# ============================================================

def _plot_per_step_metrics(metrics, pool_id, fig_dir):
    """分步 MAE/RMSE 柱状图"""
    steps = list(metrics['per_step'].keys())
    maes = [metrics['per_step'][s]['MAE'] for s in steps]
    rmses = [metrics['per_step'][s]['RMSE'] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(steps))
    width = 0.35
    ax.bar(x_pos - width/2, maes, width, label='MAE', color='#4ecdc4')
    ax.bar(x_pos + width/2, rmses, width, label='RMSE', color='#ff6b6b')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(steps)
    ax.set_ylabel('Error')
    ax.set_title(f'Pool {pool_id} — Per-Step Errors')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'per_step_errors.png'), dpi=150)
    plt.close(fig)


def _plot_sample_predictions(input_orig, pred_orig, target_orig,
                             pool_id, n_samples, fig_dir,
                             seq_len=30, pred_len=6):
    """
    从测试集中抽样 n_samples 个例子，每个例子画:
    - 30 点历史输入 (turb_chushui)
    - 6 点预测 vs 6 点实际
    """
    total = len(input_orig)
    # 均匀抽样
    indices = np.linspace(0, total - 1, n_samples, dtype=int)

    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]

        # 历史部分
        hist = input_orig[idx]          # [seq_len]
        pred = pred_orig[idx]           # [pred_len]
        actual = target_orig[idx]       # [pred_len]

        t_hist = np.arange(seq_len)
        t_pred = np.arange(seq_len, seq_len + pred_len)

        ax.plot(t_hist, hist, 'b-o', markersize=2, linewidth=1.2,
                label='History', alpha=0.8)
        ax.plot(t_pred, actual, 'g-s', markersize=4, linewidth=1.5,
                label='Actual', alpha=0.9)
        ax.plot(t_pred, pred, 'r-^', markersize=4, linewidth=1.5,
                label='Predicted', alpha=0.9)

        # 连接线: 最后历史点 → 第一个预测/实际点
        ax.plot([t_hist[-1], t_pred[0]], [hist[-1], actual[0]],
                'g--', alpha=0.4, linewidth=0.8)
        ax.plot([t_hist[-1], t_pred[0]], [hist[-1], pred[0]],
                'r--', alpha=0.4, linewidth=0.8)

        # 预测区域背景色
        ax.axvspan(seq_len - 0.5, seq_len + pred_len - 0.5,
                   alpha=0.08, color='orange')

        ax.set_title(f'Sample #{idx}', fontsize=10)
        ax.set_xlabel('Time Step (5min)')
        ax.set_ylabel('turb_chushui')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Pool {pool_id} — Sample Predictions (30-in → 6-out)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(fig_dir, 'sample_predictions.png'), dpi=150)
    plt.close(fig)
    print(f"  📈 抽样预测图已保存 ({n_samples} 个样本)")


def _plot_scatter(target_orig, pred_orig, pool_id, fig_dir):
    """全测试集预测 vs 实际散点图"""
    fig, ax = plt.subplots(figsize=(6, 6))

    y_flat = target_orig.flatten()
    p_flat = pred_orig.flatten()

    ax.scatter(y_flat, p_flat, alpha=0.05, s=2, c='#2196f3')

    # 对角线
    mn = min(y_flat.min(), p_flat.min())
    mx = max(y_flat.max(), p_flat.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1, label='y=x')

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Pool {pool_id} — Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'scatter_pred_vs_actual.png'), dpi=150)
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='xPatch 出水浊度评估')
    parser.add_argument('--csv', type=str, default='../train_data.csv')
    parser.add_argument('--pool', type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--all', action='store_true', help='评估所有池子')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--n-samples', type=int, default=6,
                        help='抽样绘图的样本数')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    pools = [1, 2, 3, 4] if args.all else [args.pool]

    for pool_id in pools:
        evaluate_pool(
            pool_id=pool_id,
            csv_path=args.csv,
            output_dir=args.output,
            n_samples=args.n_samples,
            device=device,
        )

    print("\n🎉 评估完成！")


if __name__ == '__main__':
    main()
