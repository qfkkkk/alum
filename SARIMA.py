import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
# 图注显示中文与负号
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 中文支持字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题


def load_and_transform_to_5min_series(
    csv_path: str = "shuizhi_chushui_zhuodu.csv",
    a_cols: list[str] | None = None,
) -> pd.Series:
    """
    将按小时存储且每行包含 A01~A12（12 个 5 分钟段）的宽表数据，
    转换为按 5 分钟频率的时间序列。

    - DateDay: 年月日（例如 '2022-01-01'）
    - DateHour: 小时（0~23）
    - A01~A12: 对应该小时内的 12 个 5 分钟段浊度
      假定：
        A01 -> 分钟 0
        A02 -> 分钟 5
        ...
        A12 -> 分钟 55
    """
    if a_cols is None:
        a_cols = [f"A{i:02d}" for i in range(1, 13)]

    df = pd.read_csv(csv_path)

    # 保证时间字段存在
    required_cols = {"DateDay", "DateHour"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 中必须包含列: {required_cols}")

    # 仅保留需要的列
    use_cols = ["DateDay", "DateHour"] + [c for c in a_cols if c in df.columns]
    df = df[use_cols].copy()

    # 宽表转长表：一行变为 12 行，每行一个 5 分钟浊度值
    df_long = df.melt(
        id_vars=["DateDay", "DateHour"],
        value_vars=[c for c in a_cols if c in df.columns],
        var_name="slot",
        value_name="turbidity",
    )

    # 计算每个 slot 相对于该小时起点的分钟偏移
    # A01 -> 0 分钟, A02 -> 5 分钟, ...
    df_long["slot_index"] = df_long["slot"].str[1:].astype(int) - 1
    df_long["minute_offset"] = df_long["slot_index"] * 5

    # 构造完整时间戳
    base_time = pd.to_datetime(df_long["DateDay"]) + pd.to_timedelta(
        df_long["DateHour"], unit="h"
    )
    df_long["DateTime"] = base_time + pd.to_timedelta(
        df_long["minute_offset"], unit="m"
    )

    # 排序并设置索引
    df_long = df_long.sort_values("DateTime")
    ts = df_long.set_index("DateTime")["turbidity"].astype(float)

    # 由于是规则的 5 分钟序列，指定频率为 5T（5 分钟）
    ts = ts.asfreq("5T")

    return ts


def load_hourly_a_mean_series(
    csv_path: str = "shuizhi_chushui_zhuodu_A_mean.csv",
) -> pd.Series:
    """
    从按小时存储的 CSV 中，仅使用 A_mean 列，构造按 1 小时时间步长的时间序列。

    要求 CSV 至少包含以下列:
    - DateDay: 年月日（例如 '2024-01-01'）
    - DateHour: 小时（0~23）
    - A_mean: 该小时 12 个 5 分钟浊度的均值
    """
    df = pd.read_csv(csv_path)

    required_cols = {"DateDay", "DateHour", "A_mean"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 中必须包含列: {required_cols}")

    # 仅保留需要的列
    df = df[["DateDay", "DateHour", "A_mean"]].copy()

    # 构造时间戳（按小时）
    base_time = pd.to_datetime(df["DateDay"]) + pd.to_timedelta(df["DateHour"], unit="h")
    df["DateTime"] = base_time

    # 排序并设置索引为 DatetimeIndex
    df = df.sort_values("DateTime")
    ts = df.set_index("DateTime")["A_mean"].astype(float)

    # 指定为 1 小时频率
    ts = ts.asfreq("H")

    return ts


def fit_sarima_and_forecast(
    ts: pd.Series,
    forecast_minutes: int = 60,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 288),
) -> pd.Series:
    """
    拟合 SARIMA 模型并预测未来指定分钟数的浊度。

    参数:
    - ts: 按 5 分钟频率的浊度时间序列（索引为 DateTime，freq='5T'）
    - forecast_minutes: 预测未来多少分钟（默认 60 分钟）
    - order: SARIMA 中的 (p, d, q)
    - seasonal_order: SARIMA 中的季节项 (P, D, Q, s)
        这里默认 s=288，对应 24 小时 * 60 / 5 = 288 个 5 分钟点（即日周期）
    """
    if ts.empty:
        raise ValueError("时间序列为空，无法训练模型。")

    # 确定要预测多少个 5 分钟步长
    step_per_5min = max(int(forecast_minutes / 5), 1)

    # 训练模型（可选：留一部分做验证，这里简单地用全部数据训练）
    model = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)

    # 预测未来若干个时间点
    forecast = results.forecast(steps=step_per_5min)

    return forecast


def fit_sarima_and_forecast_hourly(
    ts: pd.Series,
    forecast_hours: int = 24,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
) -> pd.Series:
    """
    使用按小时的 A_mean 序列拟合 SARIMA 并预测未来若干小时的浊度。

    参数:
    - ts: 按小时频率的浊度均值时间序列（索引为 DateTime，freq='H'）
    - forecast_hours: 预测未来多少小时（默认 24 小时）
    - order: SARIMA 中的 (p, d, q)
    - seasonal_order: SARIMA 中的季节项 (P, D, Q, s)，
        这里默认 s=24，对应 24 小时的日周期
    """
    if ts.empty:
        raise ValueError("时间序列为空，无法训练模型。")

    steps = max(int(forecast_hours), 1)

    model = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)

    forecast = results.forecast(steps=steps)
    return forecast


def main():
    # 1. 加载按小时的 A_mean 时间序列
    ts_hourly = load_hourly_a_mean_series("shuizhi_chushui_zhuodu_A_mean.csv")

    if len(ts_hourly) < 48:
        raise ValueError("数据长度不足以切分出最后一天作为测试集（至少需要 48 个小时）。")

    # 2. 取最后一天（24 个点）作为真实值，之前的数据用于训练
    points_per_day = 24
    ts_train = ts_hourly.iloc[:-points_per_day]
    ts_test = ts_hourly.iloc[-points_per_day:]

    # 3. 使用训练集拟合模型，并预测“最后一天”的 24 小时
    forecast_hourly = fit_sarima_and_forecast_hourly(
        ts_train,
        forecast_hours=points_per_day,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
    )

    # 4. 对齐预测索引到测试集（使预测时间段与最后一天一致）
    forecast_hourly.index = ts_test.index

    # 5. 保存预测结果到 CSV
    forecast_df = pd.DataFrame(
        {
            "A_mean_true": ts_test,
            "A_mean_pred": forecast_hourly,
        }
    )
    forecast_df.to_csv("sarima_forecast_last_day_compare.csv", index_label="DateTime")

    # 6. 画图：真实值 vs 预测值（最后一天）
    plt.figure(figsize=(12, 6))
    ts_test.plot(label="实际出水浊度", marker="o")
    forecast_hourly.plot(label="预测出水浊度", marker="x")
    plt.title("出水浊度  SARIMA ")
    plt.xlabel("时间")
    plt.ylabel("浊度")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sarima_forecast_last_day_compare.png", dpi=150)


if __name__ == "__main__":
    main()

