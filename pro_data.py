"""
1.读取csv文件，合并时间数据，年月日-小时-分钟
2.然后对数据进行基础的统计性分析，缺失值、零值、特征描述性统计
3.数据预处理（异常值处理、缺失值处理、数据聚合（5分钟））
4.特征工程（异常事件标注（高浊、低流量），统计特征
5.数据频率直方图和时序图，模型间的相关性可视化分析
6.投加药耗和其它变量之间的关系图
7.保存预预处理、特征工程后的数据，并对变量重命名
画图时频率1个小时，模型训练时频率5min
"""
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体，消除字体警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False 
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

ANALYSIS_COLS = [
    "1#投加药耗",
    "1#出水浊度",
    "1#反应池浊度",
    "1#进水瞬时流量",
    "2#反应池pH",
    "前池水下温度",
    "前池水面温度"
]


def read_data(input:str)->pd.DataFrame:
    df = pd.read_csv(input)
    df['DateTime'] = (pd.to_datetime(df['DateDay']) + 
                          pd.to_timedelta(df['DateHour'], unit='h') + 
                          pd.to_timedelta(df['Minute'], unit='m'))
    df.set_index("DateTime", inplace=True)

    return df

def analysis_data(df:pd.DataFrame,output:str)->None:
    df_analysis = df.copy()
    data_result = []
    for cols in ANALYSIS_COLS:
        data_info = df_analysis[cols].describe()
        print(f"{cols}的描述性统计\n{data_info}\n")
        missing_count = df_analysis[cols].isna().sum()
        zero_count = (df_analysis[cols] == 0).sum()
        print(f"{cols}的缺失值数量: {missing_count}\n, 零值数量: {zero_count}\n")
        missing_ratio = missing_count / data_info["count"]*100
        zero_ratio = zero_count / data_info["count"]*100
        print(f"{cols}的缺失值比例: {missing_ratio:.2f}%\n, 零值比例: {zero_ratio:.2f}%\n")
        skew_result = df[cols].skew()
        kurt_result = df[cols].kurt()
        print(f"{cols}的偏度: {skew_result:.2f}\n, 峰度: {kurt_result:.2f}\n")
        data_result.append({
            "Column": cols,
            "Count":data_info["count"],
            "Max": data_info["max"],
            "Min": data_info["min"],
            "Mena": data_info["mean"],
            "Std": data_info["std"],
            "MissingCount": missing_count,
            "MissingRatio(%)": missing_ratio,
            "ZeroCount": zero_count,
            "ZeroRatio(%)": zero_ratio,
            "Skewness": skew_result,
            "Kurtosis": kurt_result
        })
    df_result = pd.DataFrame(data_result)
    df_result.to_csv("data_analysis.csv",index=False)


def pro_data(df:pd.DataFrame,window_size:int,min_periods:int,
                SIGMA_k:float = 3.0,freq:str = "5min",
        )->pd.DataFrame:
    df_pro = df.copy()

    # 水下温度处理,将小于0，大于50的数据用前池水面温度的特征数值进行替换
    df_pro.loc[(df_pro["前池水下温度"]<0) | (df_pro["前池水下温度"]>50),"前池水下温度"] = df_pro["前池水面温度"]
    # 业务规则处理
    for cols in ANALYSIS_COLS:
        df_pro = df_pro[df_pro[cols]>=0]
    # 异常值处理，建立滑动窗口
    for cols in ANALYSIS_COLS:
        roll_window = df_pro[cols].rolling(
            window = window_size,
            center = False,
            min_periods = min_periods
        )
        # 计算窗口内的均值和标准差
        df_pro["window_mean"] = roll_window.mean()
        df_pro["window_std"] = roll_window.std()
        # 计算窗口内的边界
        df_pro["lower_bound"] = df_pro["window_mean"] - SIGMA_k * df_pro["window_std"]
        df_pro["upper_bound"] = df_pro["window_mean"] + SIGMA_k * df_pro["window_std"]
        # 异常值处理
        df_pro.loc[(df_pro[cols] < df_pro["lower_bound"]) | 
                (df_pro[cols] > df_pro["upper_bound"]), cols] = df_pro["window_mean"]
        # 统计异常值信息
        num_anomalies = df_pro[
            (df_pro[cols] < df_pro["lower_bound"]) |
            (df_pro[cols] > df_pro["upper_bound"])
        ].shape[0]
        print(f"列 {cols} 中检测到的异常值数量: {num_anomalies}")
        
        # 删除临时列
        df_pro.drop(columns=["window_mean", "window_std", "lower_bound", "upper_bound"], inplace=True)

    # 缺失值处理
    df_pro.ffill(inplace=True)
    df_pro.bfill(inplace=True)
    
    # 数据聚合（5分钟）
    df_pro = df_pro.resample(freq).mean(numeric_only=True)
    
    # 新增特征列：低流量段和高浊度段标识
    df_pro["is_low_flow"] = (df_pro["1#进水瞬时流量"] < 500).astype(int)
    df_pro["is_high_turb"] = (df_pro["1#反应池浊度"] > 500).astype(int)
    
    low_flow_count = df_pro["is_low_flow"].sum()
    high_turb_count = df_pro["is_high_turb"].sum()
    print(f"低流量段数量 (流量<500): {low_flow_count:,}")
    print(f"高浊度段数量 (浊度>500): {high_turb_count:,}")
    
    return df_pro
def plot_data(df, cols, out_dir="figure_result", freq='60min', flow_threshold=500, 
                       high_turbidity_threshold=500):
    """
    按指定频率（默认60分钟/1小时）聚合数据
    并绘制时间序列折线图和频率分布直方图
    在时序图中标注流量低于阈值的时间段和高浊事件
    """
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("数据可视化")
    print(f"{'='*60}")
    print(f"聚合频率: {freq}")
    print(f"低流量阈值: {flow_threshold}")
    print(f"高浊度阈值: {high_turbidity_threshold}")
    
    # 按照 freq 聚合
    agg = df[cols].resample(freq).mean()
    agg_counts = df[cols].resample(freq).count()
    
    agg.to_csv(os.path.join(out_dir, "aggregated_mean.csv"), encoding='utf-8-sig')
    agg_counts.to_csv(os.path.join(out_dir, "aggregated_count.csv"), encoding='utf-8-sig')
    print(f"聚合数据已保存: {os.path.join(out_dir, 'aggregated_mean.csv')}")
    
    # 检查是否有流量列用于标注低流量时段
    flow_col = "1#进水瞬时流量"
    has_flow_col = flow_col in agg.columns
    low_flow_mask = None
    low_flow_count = 0
    
    if has_flow_col:
        # 找出流量低于阈值的时间点
        low_flow_mask = (agg[flow_col] < flow_threshold) & agg[flow_col].notna()
        low_flow_count = low_flow_mask.sum()
        print(f"低流量时间点数量 (流量 < {flow_threshold}): {low_flow_count:,}")
    else:
        print(f"警告：未找到流量列 '{flow_col}'，无法标注低流量时段")
    
    # 检查是否有浊度列用于标注高浊事件
    turb_col = "1#反应池浊度"
    has_turb_col = turb_col in agg.columns
    high_turb_mask = None
    high_turb_count = 0
    
    if has_turb_col:
        # 找出浊度高于阈值的时间点（高浊事件）
        high_turb_mask = (agg[turb_col] > high_turbidity_threshold) & agg[turb_col].notna()
        high_turb_count = high_turb_mask.sum()
        print(f"高浊事件数量 (浊度 > {high_turbidity_threshold}): {high_turb_count:,}")
        
    else:
        print(f"警告：未找到浊度列 '{turb_col}'，无法标注高浊事件")
    
    # 为每个列画合并的时间序列图与直方图
    for c in cols:
        if c not in agg.columns:
            continue
        
        stats = agg[c].describe()
        skew = agg[c].skew()
        kurt = agg[c].kurt()
        
        # 创建包含两个子图的Figure：上面时间序列，下面直方图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                        gridspec_kw={'height_ratios': [1.5, 1]})
        
        # 上方：时间序列散点图
        if has_flow_col and low_flow_mask is not None:
            # 分离正常流量和低流量时段的数据
            normal_flow_mask = ~low_flow_mask & agg[c].notna()
            normal_flow_data = agg.loc[normal_flow_mask, c]
            low_flow_data = agg.loc[low_flow_mask, c].dropna()
            
            # 绘制正常流量时段的散点（蓝色）
            if len(normal_flow_data) > 0:
                ax1.scatter(normal_flow_data.index, normal_flow_data.values,
                           c='steelblue', s=10, alpha=0.6,
                           label=f'正常流量时段 (n={len(normal_flow_data):,})')
            
            # 绘制低流量时段的散点（橙色）
            if len(low_flow_data) > 0:
                ax1.scatter(low_flow_data.index, low_flow_data.values, 
                           c='orange', s=15, alpha=0.7, zorder=5,
                           label=f'低流量时段 (流量<{flow_threshold}, n={len(low_flow_data):,})')
        else:
            # 没有流量列时，直接用蓝色散点绘制所有数据
            data_series = agg[c].dropna()
            ax1.scatter(data_series.index, data_series.values,
                       c='steelblue', s=10, alpha=0.6,
                       label=f'数据点 (n={len(data_series):,})')
        
        # 标注高浊事件
        if has_turb_col and high_turb_mask is not None and high_turb_count > 0:
            # 获取当前列在高浊事件时的数据
            high_turb_data = agg.loc[high_turb_mask, c].dropna()
            if len(high_turb_data) > 0:
                ax1.scatter(high_turb_data.index, high_turb_data.values,
                           c='red', s=60, alpha=0.9, zorder=10,
                           marker='^', 
                           edgecolors='darkred', linewidths=0.5,
                           label=f'高浊事件 (浊度>{high_turbidity_threshold}, n={len(high_turb_data):,})')
        
        # 添加统计信息文本框
        text_ts = (
            f"样本数: {int(stats['count']):,}\n"
            f"均值: {stats['mean']:.3f}\n"
            f"标准差: {stats['std']:.3f}\n"
            f"最小值: {stats['min']:.3f}\n"
            f"最大值: {stats['max']:.3f}"
        )
        ax1.text(0.02, 0.95, text_ts, transform=ax1.transAxes, fontsize=10, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title(f"{c} - {freq}", fontsize=12)
        ax1.set_xlabel('时间', fontsize=10)
        ax1.set_ylabel(c, fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        
        # 下方：频率分布直方图（排除0值和低流量时段）
        hist_data = agg[c].dropna()
        
        # 排除低流量时段的数据
        if has_flow_col and low_flow_mask is not None:
            # 只保留正常流量时段的数据
            normal_flow_indices = ~low_flow_mask
            hist_data = hist_data[hist_data.index.isin(agg.index[normal_flow_indices])]
            low_flow_excluded = low_flow_count
        else:
            low_flow_excluded = 0
        
        # 排除0值
        zero_count = (hist_data == 0).sum()
        hist_data_nonzero = hist_data[hist_data != 0]
        
        if len(hist_data_nonzero) > 0:
            sns.histplot(hist_data_nonzero, kde=True, bins=50, ax=ax2, color='steelblue', alpha=0.7)
        
        # 设置标题（说明排除了哪些数据）
        exclude_info = "排除"
        if low_flow_excluded > 0:
            exclude_info += f"低流量时段"
        ax2.set_title(f"{c} - ({exclude_info})", fontsize=12)
        ax2.set_xlabel(c, fontsize=10)
        ax2.set_ylabel('频次', fontsize=10)
        
        # 计算非零数据的统计信息
        if len(hist_data_nonzero) > 0:
            stats_nonzero = hist_data_nonzero.describe()
            skew_nonzero = hist_data_nonzero.skew()
            kurt_nonzero = hist_data_nonzero.kurt()
        else:
            stats_nonzero = stats
            skew_nonzero = skew
            kurt_nonzero = kurt
        
        # 添加更详细的统计信息（基于筛选后数据）
        exclude_detail = f"排除{zero_count:,}个0值"
        if low_flow_excluded > 0:
            exclude_detail += f", {low_flow_excluded:,}个低流量时段"
        stats_text = (
            f"n = {len(hist_data_nonzero):,}\n"
            f"({exclude_detail})\n"
            f"均值 = {stats_nonzero['mean']:.3f}\n"
            f"标准差 = {stats_nonzero['std']:.3f}\n"
            f"偏度 = {skew_nonzero:.3f}\n"
            f"峰度 = {kurt_nonzero:.3f}"
        )
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 在直方图上添加均值和中位数的垂直线（基于筛选后数据）
        if len(hist_data_nonzero) > 0:
            mean_val = stats_nonzero['mean']
            median_val = stats_nonzero['50%']
            ax2.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'均值: {mean_val:.2f}')
            ax2.axvline(median_val, color='green', linestyle='-.', linewidth=1.5, label=f'中位数: {median_val:.2f}')
            ax2.legend(loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        # 保存合并图
        safe_col_name = c.replace('#', '_').replace('/', '_')
        out_combined = os.path.join(out_dir, f"{safe_col_name}.png")
        plt.savefig(out_combined, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {out_combined}")
    
    return agg

def main():
    input = r"E:\Water_Plan_Control\alum\pro_data\merged_data.csv"
    output = r"E:\Water_Plan_Control\alum\pro_data\processed_data.csv"
    # 列名重命名映射（包含新增的标识列）
    COLUMN_RENAME_MAP = {
        "1#投加药耗": "dose_1",
        "1#出水浊度": "turb_chushui_1",
        "1#反应池浊度": "turb_jinshui_1",
        "1#进水瞬时流量": "flow_1",
        "2#反应池pH": "pH",
        "前池水下温度": "temp_down",
        "前池水面温度": "temp_shuimian",
        "is_low_flow": "low_flow",
        "is_high_turb": "high_turb"
    }
    SAVE_COLS = ANALYSIS_COLS + ["is_low_flow", "is_high_turb"]

    df = read_data(input)
    analysis_data(df, output)
    df_pro = pro_data(df, window_size=60, min_periods=3,
                      SIGMA_k=3.0, freq="5min")
    
    # 重命名列名后保存
    df_save = df_pro[SAVE_COLS].rename(columns=COLUMN_RENAME_MAP)
    df_save.to_csv(output)
    print(f"预处理后的数据已保存到 {output}")
    print(f"列名映射: {COLUMN_RENAME_MAP}")

    plot_data(df_pro,ANALYSIS_COLS , out_dir="figure_result", freq='60T', 
            flow_threshold=500, high_turbidity_threshold=500)

if __name__ == "__main__":
    main()
    

