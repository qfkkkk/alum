"""
LightGBM 出水浊度预测模型
- 基于流量分箱的动态时滞特征构建
- 时间序列划分：前90%训练，后10%测试
"""
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# 图注显示中文与负号
plt.rcParams["font.sans-serif"] = ["SimHei"]d
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置参数 ============
DATA_PATH = Path("pro_data") / "processed_data.csv"
TARGET_COL = "turb_chushui_1"
FIG_DIR = Path("model")

# 流量分箱与对应时滞（小时）
# 根据图片中的理论时滞数据
FLOW_LAG_CONFIG = {
    # (flow_min, flow_max): lag_hours
    (0, 4000): 4.027,      # 3000~4000 区间，时滞约 4.027h
    (4000, 6000): 2.777,   # 4000~6000 区间，时滞约 2.777h
    (6000, 7000): 2.177,   # 6000~7000 区间，时滞约 2.177h
    (7000, float('inf')): 1.805,  # 7000~8500+ 区间，时滞约 1.805h
}

# 时间间隔（分钟）
TIME_INTERVAL_MIN = 5


def get_lag_steps(flow: float) -> int:
    """
    根据流量值获取对应的时滞步数。
    时滞步数 = 时滞小时数 * 60 / 时间间隔(分钟)
    """
    for (flow_min, flow_max), lag_hours in FLOW_LAG_CONFIG.items():
        if flow_min <= flow < flow_max:
            lag_steps = int(round(lag_hours * 60 / TIME_INTERVAL_MIN))
            return lag_steps
    # 默认返回中等时滞
    return int(round(2.5 * 60 / TIME_INTERVAL_MIN))


def load_data(path: Path) -> pd.DataFrame:
    """读取并按时间排序数据集。"""
    df = pd.read_csv(path)
    # 解析 DateTime 列
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    # 按时间排序
    df = df.sort_values("DateTime").reset_index(drop=True)
    # 添加时间相关特征
    df["hour"] = df["DateTime"].dt.hour
    df["weekday"] = df["DateTime"].dt.weekday  # 0=周一
    df["month"] = df["DateTime"].dt.month
    print(f"数据加载完成: {len(df)} 条记录")
    print(f"时间范围: {df['DateTime'].min()} ~ {df['DateTime'].max()}")
    return df


def build_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据流量分箱构建动态时滞特征。
    对于每一行，根据当前流量确定时滞，然后从对应时滞前的时间点获取特征。
    """
    df = df.copy()
    
    # 计算每行对应的时滞步数
    df["lag_steps"] = df["flow_1"].apply(get_lag_steps)
    
    # 定义需要构建时滞特征的列
    lag_feature_cols = ["dose_1", "turb_jinshui_1", "pH", "temp_down", "temp_shuimian"]
    
    # 为每个特征创建对应时滞的版本
    print("构建动态时滞特征...")
    for col in lag_feature_cols:
        lagged_values = []
        for idx in range(len(df)):
            lag_step = df.iloc[idx]["lag_steps"]
            # 时滞前的索引
            lagged_idx = idx - lag_step
            if lagged_idx >= 0:
                lagged_values.append(df.iloc[lagged_idx][col])
            else:
                lagged_values.append(np.nan)
        df[f"{col}_lagged"] = lagged_values
    
    # 添加流量区间标签（用于分析）
    def get_flow_bin(flow):
        if flow < 4000:
            return "3000-4000"
        elif flow < 6000:
            return "4000-6000"
        elif flow < 7000:
            return "6000-7000"
        else:
            return "7000+"
    
    df["flow_bin"] = df["flow_1"].apply(get_flow_bin)
    
    # 移除因时滞产生的缺失值
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"移除缺失值后: {len(df)} 条记录 (移除 {initial_len - len(df)} 条)")
    
    return df


def train_test_split_by_time(df: pd.DataFrame, test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间序列划分训练集和测试集。
    前 (1-test_ratio) 为训练集，后 test_ratio 为测试集。
    """
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    print(f"训练集: {len(train_df)} 条 ({len(train_df)/n*100:.1f}%)")
    print(f"测试集: {len(test_df)} 条 ({len(test_df)/n*100:.1f}%)")
    print(f"训练集时间范围: {train_df['DateTime'].min()} ~ {train_df['DateTime'].max()}")
    print(f"测试集时间范围: {test_df['DateTime'].min()} ~ {test_df['DateTime'].max()}")
    
    return train_df, test_df


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """构建训练与测试数据集。"""
    # 构建时滞特征
    df = build_lagged_features(df)
    
    # 划分训练集和测试集
    train_df, test_df = train_test_split_by_time(df, test_ratio=0.1)
    
    # 定义特征列（使用时滞后的特征 + 其他特征）
    feature_cols = [
        "dose_1_lagged", "turb_jinshui_1_lagged", "pH_lagged", 
        "temp_down_lagged", "temp_shuimian_lagged",
        "flow_1", "low_flow", "high_turb",
        "hour", "weekday", "month", "lag_steps"
    ]
    
    # 检查特征列是否存在
    available_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"使用特征: {available_cols}")
    
    X_train = train_df[available_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[available_cols]
    y_test = test_df[TARGET_COL]
    test_time = test_df["DateTime"]
    
    return X_train, y_train, X_test, y_test, test_time, test_df


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
    """训练 LightGBM 回归模型。"""
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,  # 使用单进程避免Windows权限问题
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[lgb.LGBMRegressor, dict, float]:
    """通过网格搜索调整超参数，返回最佳模型与指标。"""
    base_model = lgb.LGBMRegressor(random_state=42, n_jobs=1, verbose=-1)
    param_grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.03],
        "num_leaves": [31, 63],
        "max_depth": [-1, 10],
        "subsample": [0.8],
        "colsample_bytree": [0.8, 0.95],
    }
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=1,  # 使用单进程避免Windows权限问题
        verbose=1,
    )
    grid.fit(X_train, y_train)
    best_model: lgb.LGBMRegressor = grid.best_estimator_
    best_params = grid.best_params_
    cv_rmse = float(np.sqrt(-grid.best_score_))
    print("网格搜索完成：")
    print(f"最佳参数: {best_params}")
    print(f"CV RMSE: {cv_rmse:.4f}")
    return best_model, best_params, cv_rmse


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """计算评估指标。"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_feature_importance(model: lgb.LGBMRegressor, feature_names: List[str], save_path: Path) -> None:
    """绘制并保存特征重要性。"""
    importance = model.feature_importances_
    order = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(np.array(feature_names)[order][::-1], importance[order][::-1])
    plt.xlabel("重要性 (分裂次数)")
    plt.title("LightGBM 特征重要性")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"特征重要性图已保存至: {save_path.resolve()}")


def plot_prediction_compare(
    y_true: pd.Series, y_pred: np.ndarray, timestamps: pd.Series, save_path: Path
) -> None:
    """绘制预测 vs 实际对比图。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子图1: 时间序列对比
    ax1 = axes[0]
    ax1.plot(timestamps, y_true.values, label="实际出水浊度", alpha=0.7, linewidth=1)
    ax1.plot(timestamps, y_pred, label="预测出水浊度", alpha=0.7, linewidth=1)
    ax1.set_xlabel("时间")
    ax1.set_ylabel("出水浊度")
    ax1.set_title("出水浊度预测对比（测试集时间序列）")
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 子图2: 散点图
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label="理想预测线")
    ax2.set_xlabel("实际出水浊度")
    ax2.set_ylabel("预测出水浊度")
    ax2.set_title("预测值 vs 实际值散点图")
    ax2.legend()
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"预测对比图已保存至: {save_path.resolve()}")


def plot_error_distribution(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    """绘制预测误差分布图。"""
    errors = y_pred - y_true.values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 误差直方图
    ax1 = axes[0]
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='r', linestyle='--', label='零误差线')
    ax1.set_xlabel("预测误差 (预测值 - 实际值)")
    ax1.set_ylabel("频次")
    ax1.set_title(f"预测误差分布 (均值={errors.mean():.4f}, 标准差={errors.std():.4f})")
    ax1.legend()
    
    # 误差绝对值累积分布
    ax2 = axes[1]
    abs_errors = np.abs(errors)
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax2.plot(sorted_errors, cumulative)
    ax2.set_xlabel("绝对误差")
    ax2.set_ylabel("累积概率")
    ax2.set_title("预测绝对误差累积分布")
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%分位')
    ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95%分位')
    ax2.legend()
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"误差分布图已保存至: {save_path.resolve()}")


def plot_flow_bin_analysis(test_df: pd.DataFrame, y_pred: np.ndarray, save_path: Path) -> None:
    """按流量分箱分析预测性能。"""
    test_df = test_df.copy()
    test_df["y_pred"] = y_pred
    test_df["error"] = y_pred - test_df[TARGET_COL].values
    test_df["abs_error"] = np.abs(test_df["error"])
    
    # 按流量分箱统计
    flow_bins = ["3000-4000", "4000-6000", "6000-7000", "7000+"]
    metrics_by_bin = []
    
    for bin_name in flow_bins:
        bin_data = test_df[test_df["flow_bin"] == bin_name]
        if len(bin_data) > 0:
            mae = bin_data["abs_error"].mean()
            rmse = np.sqrt((bin_data["error"] ** 2).mean())
            r2 = r2_score(bin_data[TARGET_COL], bin_data["y_pred"])
            metrics_by_bin.append({
                "流量区间": bin_name,
                "样本数": len(bin_data),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "平均时滞步数": bin_data["lag_steps"].mean()
            })
    
    metrics_df = pd.DataFrame(metrics_by_bin)
    print("\n按流量区间的预测性能:")
    print(metrics_df.to_string(index=False))
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 各区间MAE
    ax1 = axes[0, 0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(metrics_df)))
    bars = ax1.bar(metrics_df["流量区间"], metrics_df["MAE"], color=colors, edgecolor='black')
    ax1.set_xlabel("流量区间 (m³/h)")
    ax1.set_ylabel("MAE")
    ax1.set_title("各流量区间的平均绝对误差 (MAE)")
    for bar, val in zip(bars, metrics_df["MAE"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 各区间R2
    ax2 = axes[0, 1]
    bars = ax2.bar(metrics_df["流量区间"], metrics_df["R2"], color=colors, edgecolor='black')
    ax2.set_xlabel("流量区间 (m³/h)")
    ax2.set_ylabel("R²")
    ax2.set_title("各流量区间的决定系数 (R²)")
    for bar, val in zip(bars, metrics_df["R2"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 各区间样本数
    ax3 = axes[1, 0]
    bars = ax3.bar(metrics_df["流量区间"], metrics_df["样本数"], color=colors, edgecolor='black')
    ax3.set_xlabel("流量区间 (m³/h)")
    ax3.set_ylabel("样本数")
    ax3.set_title("各流量区间的测试样本数")
    for bar, val in zip(bars, metrics_df["样本数"]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{val}', ha='center', va='bottom', fontsize=9)
    
    # 各区间时滞
    ax4 = axes[1, 1]
    bars = ax4.bar(metrics_df["流量区间"], metrics_df["平均时滞步数"] * TIME_INTERVAL_MIN / 60, 
                   color=colors, edgecolor='black')
    ax4.set_xlabel("流量区间 (m³/h)")
    ax4.set_ylabel("平均时滞 (小时)")
    ax4.set_title("各流量区间使用的平均时滞")
    for bar, val in zip(bars, metrics_df["平均时滞步数"] * TIME_INTERVAL_MIN / 60):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}h', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"流量分箱分析图已保存至: {save_path.resolve()}")
    
    # 保存指标到CSV
    csv_path = save_path.parent / "flow_bin_metrics.csv"
    metrics_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"流量分箱指标已保存至: {csv_path.resolve()}")



def main():
    print("=" * 60)
    print("LightGBM 出水浊度预测模型 - 动态时滞版本")
    print("=" * 60)
    
    # 加载数据
    df = load_data(DATA_PATH)
    
    # 构建数据集
    X_train, y_train, X_test, y_test, test_time, test_df = build_dataset(df)
    
    print("\n" + "=" * 60)
    print("开始模型训练...")
    print("=" * 60)
    
    # 网格搜索获得最佳模型
    model, best_params, cv_rmse = tune_hyperparameters(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    metrics = evaluate(y_test, y_pred)
    print("\n" + "=" * 60)
    print("测试集评估指标：")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 保存可视化图
    print("\n" + "=" * 60)
    print("生成可视化图...")
    print("=" * 60)
    
    # 1. 特征重要性图
    plot_feature_importance(
        model, X_train.columns.tolist(), 
        FIG_DIR / "feature_importance.png"
    )
    
    # 2. 预测对比图
    plot_prediction_compare(
        y_test, y_pred, test_time, 
        FIG_DIR / "pred_vs_actual.png"
    )
    
    # 3. 误差分布图
    plot_error_distribution(
        y_test, y_pred, 
        FIG_DIR / "error_distribution.png"
    )
    
    # 4. 流量分箱分析图
    plot_flow_bin_analysis(
        test_df, y_pred, 
        FIG_DIR / "flow_bin_analysis.png"
    )
    

    
    print("\n" + "=" * 60)
    print("模型训练与评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
