"""
出水浊度预测 - xPatch 训练脚本

用法:
    # 训练单个池子
    python train.py --pool 1 --epochs 100

    # 训练所有池子
    python train.py --all --epochs 100

    # CPU 快速测试模式（少量数据 + 少量 epoch）
    python train.py --pool 1 --epochs 3 --test-mode

    # 直接预测模式 — 预测绝对值
    python train.py --pool 1 --epochs 100 --no-diff
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass

from dataset import create_datasets, TURB_CHUSHUI_IDX
from models.xPatch import Model


# ============================================================
# 配置
# ============================================================

@dataclass
class ModelConfig:
    """xPatch 模型配置"""
    seq_len: int = 30       # 输入序列长度
    pred_len: int = 6       # 预测长度
    enc_in: int = 6         # 输入通道数
    patch_len: int = 6      # Patch 长度
    stride: int = 3         # Patch 步长
    padding_patch: str = 'end'  # Patch 填充方式
    d_model: int = 64       # 模型隐藏维度（unused in new xPatch, kept for compat）
    revin: bool = True      # 是否使用 RevIN
    ma_type: str = 'ema'    # 移动平均类型: 'ema', 'dema', 'reg'（不分解）
    alpha: float = 0.5      # EMA 平滑因子
    beta: float = 0.5       # DEMA 平滑因子（仅 dema 模式用）


@dataclass
class TrainConfig:
    """训练配置"""
    csv_path: str = ''
    pool_id: int = 1
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10      # 早停耐心值
    use_diff: bool = True   # 差分目标
    output_dir: str = 'output'
    test_mode: bool = False  # CPU 快速测试模式


# ============================================================
# 训练逻辑
# ============================================================

def train_one_pool(train_cfg: TrainConfig, model_cfg: ModelConfig, device: torch.device):
    """训练单个池子的 xPatch 模型"""
    pool_id = train_cfg.pool_id
    print(f"\n{'='*60}")
    print(f"  训练池子 {pool_id}  |  设备: {device}")
    print(f"{'='*60}")

    # 输出目录
    pool_dir = os.path.join(train_cfg.output_dir, f'pool_{pool_id}')
    os.makedirs(pool_dir, exist_ok=True)
    scaler_path = os.path.join(pool_dir, 'scaler.pkl')

    # 创建数据集
    print("[1/4] 加载数据...")
    train_ds, val_ds, test_ds, scaler, feature_names = create_datasets(
        csv_path=train_cfg.csv_path,
        pool_id=pool_id,
        seq_len=model_cfg.seq_len,
        pred_len=model_cfg.pred_len,
        use_diff=train_cfg.use_diff,
        scaler_path=scaler_path,
    )
    print(f"  特征: {feature_names}")
    print(f"  训练集: {len(train_ds)} | 验证集: {len(val_ds)} | 测试集: {len(test_ds)}")

    # 测试模式: 只用少量数据
    if train_cfg.test_mode:
        from torch.utils.data import Subset
        n_test_samples = min(500, len(train_ds))
        train_ds = Subset(train_ds, range(n_test_samples))
        val_ds = Subset(val_ds, range(min(100, len(val_ds))))
        print(f"  [测试模式] 缩减为: 训练 {len(train_ds)} | 验证 {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size,
                            shuffle=False, num_workers=0, drop_last=False)

    # 创建模型
    print("[2/4] 构建模型...")
    model = Model(model_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.epochs, eta_min=1e-6
    )
    criterion = nn.MSELoss()

    # 训练循环
    print("[3/4] 开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, train_cfg.epochs + 1):
        # --- 训练 ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y, last_vals in train_loader:
            x = x.to(device)       # [B, seq_len, n_features]
            y = y.to(device)       # [B, pred_len, n_features]

            optimizer.zero_grad()
            pred = model(x)        # [B, pred_len, n_features]
            loss = criterion(pred, y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y, last_vals in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)

        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)

        scheduler.step()

        # 日志
        if epoch % max(1, train_cfg.epochs // 20) == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d}/{train_cfg.epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {lr_now:.2e}")

        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'model_config': model_cfg.__dict__,
                'train_config': {
                    'use_diff': train_cfg.use_diff,
                    'pool_id': pool_id,
                    'feature_names': feature_names,
                },
            }
            torch.save(checkpoint, os.path.join(pool_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                print(f"  早停: 验证损失 {train_cfg.patience} 个 epoch 未改善")
                break

    # 保存训练历史
    print("[4/4] 保存结果...")
    history = {'train_loss': train_losses, 'val_loss': val_losses}
    with open(os.path.join(pool_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 保存训练曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'Pool {pool_id} Training Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(pool_dir, 'train_curve.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [警告] 绘图失败: {e}")

    print(f"  ✅ 池子 {pool_id} 训练完成 | 最佳验证损失: {best_val_loss:.6f}")
    print(f"  模型保存: {os.path.join(pool_dir, 'best_model.pt')}")

    return best_val_loss


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='xPatch 出水浊度预测训练')
    parser.add_argument('--csv', type=str, default='../train_data.csv',
                        help='CSV 数据路径')
    parser.add_argument('--pool', type=int, default=1, choices=[1, 2, 3, 4],
                        help='池子编号')
    parser.add_argument('--all', action='store_true', help='训练所有池子')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--pred-len', type=int, default=6)
    parser.add_argument('--patch-len', type=int, default=6)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--ma-type', type=str, default='ema',
                        choices=['ema', 'dema', 'reg'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--no-diff', action='store_true',
                        help='不使用差分目标（直接预测原始值）')
    parser.add_argument('--no-revin', action='store_true',
                        help='不使用 RevIN')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--test-mode', action='store_true',
                        help='CPU 测试模式（少量数据快速验证）')
    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model_cfg = ModelConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=6,
        patch_len=args.patch_len,
        stride=args.stride,
        revin=not args.no_revin,
        ma_type=args.ma_type,
        alpha=args.alpha,
        beta=args.beta,
    )

    pools = [1, 2, 3, 4] if args.all else [args.pool]

    for pool_id in pools:
        train_cfg = TrainConfig(
            csv_path=args.csv,
            pool_id=pool_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_diff=not args.no_diff,
            patience=args.patience,
            output_dir=args.output,
            test_mode=args.test_mode,
        )
        train_one_pool(train_cfg, model_cfg, device)

    print("\n🎉 全部训练完成！")


if __name__ == '__main__':
    main()
