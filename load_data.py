from operator import index
from tracemalloc import start
from typing import Sequence
import pandas as pd
import numpy as np
import os

from pandas.core import frame
TAG_MAP = [
    {
        "path":"data/shuizhi__202508221013.csv",
        "tag":[4,5,66,67],
        "alias_map":{4: "1#出水浊度",5: "2#出水浊度",66: "3#出水浊度",67: "4#出水浊度"},
    },
    {
        "path":"data/ZHJY__202201-202508(20251205).csv",
        "tag":[431, 432, 433,434,435,436,437,438,455,
               459,463,467,471,477,478,479,480,481,
               473,474,475,476,491,492,493,494],
        "alias_map":{
            431: "1#储液池液位",
                432: "2#储液池液位",
                433: "3#储液池液位",
                434: "1#投加流量",
                435: "2#投加流量",
                436: "3#投加流量",
                437: "4#投加流量",
                438: "5#投加流量",
                455: "1#泵投加流量",
                459: "2#泵投加流量",
                463: "3#泵投加流量",
                467: "4#泵投加流量",
                471: "5#泵投加流量",
                477: "1#投加药耗",
                478: "2#投加药耗",
                479: "3#投加药耗",
                480: "4#投加药耗",
                481: "5#投加药耗",
                473: "1#流量计小时流量",
                474: "2#流量计小时流量",
                475: "3#流量计小时流量",
                476: "4#流量计小时流量",
                491:"1#矾花识别——投加药耗",
                492:"2#矾花识别——投加药耗",
                493:"3#矾花识别——投加药耗",
                494:"4#矾花识别——投加药耗",
        }
    },
    {
        "path": "data/flowpress_202401-202507.csv", 
        "tag": [6,10,244,248],
        "alias_map": {244: "1#进水瞬时流量",
                        248: "2#进水瞬时流量",
                        6:"3#进水瞬时流量",
                        10:"4#进水瞬时流量"
                },
    },
    {
        "path": "data/ShuiZhi_202201-202508(20251205).csv",  
        "tag": [84,85,87,88,82],
        "alias_map": {
            84: "1#反应池浊度",
            85: "2#反应池浊度",
            87: "3#反应池浊度",
            88: "4#反应池浊度",
            82: "2#反应池pH",
        },
    },
    {
        "path": "data/JingShui_202201-202508(20251218).csv",  
        "tag": [213,214],
        "alias_map": {
            213: "前池水面温度",
            214: "前池水下温度",
        },
    },
]
def load_data(start_dt:str,
    end_dt:str,
    tag_map:Sequence[dict],
    output:str,
    chunksize:int = 100000
    )->pd.DataFrame:
    """
    优化版本：分块读取 + 提前过滤 + 避免outer merge内存爆炸
    
    参数:
        chunksize: 每次读取的行数（默认100000）
    """
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    
    # 收集所有需要的tag
    all_tags = sorted({n for item in tag_map for n in item["tag"]})
    
    # 构建别名映射
    alias_lookup = {
        int(k): v for item in tag_map for k, v in item["alias_map"].items()
    }
    
    # 生成时间列
    time_min = np.arange(1, 61)
    time_cols = np.char.add("T", np.char.mod("%02d", time_min)).tolist()
    
    # 存储每个tag处理后的长格式数据
    all_tag_data = []
    
    for item in tag_map:
        path = item["path"]
        tags_in_file = item["tag"]
        
        print(f"正在处理文件: {path}")
        
        # 分块读取文件
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            # 立即过滤：只保留需要的tag和时间范围
            chunk["DateDay"] = pd.to_datetime(chunk["DateDay"])
            
            filtered_chunk = chunk[
                (chunk["TagIndex"].isin(tags_in_file)) &
                (chunk["DateDay"] >= start_dt) &
                (chunk["DateDay"] <= end_dt)
            ].copy()
            
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)
            
            # 释放内存
            del chunk
        
        if not chunks:
            print(f"  文件 {path} 在指定时间范围内无数据")
            continue
        
        # 合并该文件的所有chunk
        df_file = pd.concat(chunks, ignore_index=True)
        del chunks
        
        # 逐个tag处理
        for tag_idx in tags_in_file:
            df_single_tag = df_file[df_file["TagIndex"] == tag_idx].copy()
            
            if df_single_tag.empty:
                continue
            
            # 每一个编号均确保所有时间列都存在
            for col in time_cols:
                if col not in df_single_tag.columns:
                    df_single_tag[col] = pd.NA
            
            existing_time_cols = [col for col in time_cols if col in df_single_tag.columns]
            
            # melt操作
            df_tag_long = df_single_tag.melt(
                id_vars=["DateDay", "DateHour"],
                value_vars=existing_time_cols,
                var_name="Minute",
                value_name="value"
            )
            
            df_tag_long["Minute"] = df_tag_long["Minute"].str.extract(r"(\d+)").astype(int)
            
            # 删除空值行，减少数据量
            df_tag_long = df_tag_long.dropna(subset=["value"])
            
            # 使用别名作为列名
            col_name = alias_lookup.get(int(tag_idx), f"TAG_{tag_idx}")
            df_tag_long.rename(columns={"value": col_name}, inplace=True)
            
            # 只保留需要的列
            df_tag_long = df_tag_long[["DateDay", "DateHour", "Minute", col_name]]
            
            all_tag_data.append((col_name, df_tag_long))
            print(f"  已处理tag: {col_name}, 数据行数: {len(df_tag_long)}")
            
            del df_single_tag, df_tag_long
        
        del df_file
    
    if not all_tag_data:
        print("没有找到符合条件的数据")
        return pd.DataFrame()
    
    # 创建完整的时间索引（只包含有数据的时间点）
    print("\n正在创建时间索引...")
    time_points = set()
    for _, df in all_tag_data:
        for _, row in df.iterrows():
            time_points.add((row["DateDay"], row["DateHour"], row["Minute"]))
    
    time_index = pd.DataFrame(list(time_points), 
                             columns=["DateDay", "DateHour", "Minute"])
    time_index.sort_values(["DateDay", "DateHour", "Minute"], inplace=True)
    time_index.reset_index(drop=True, inplace=True)
    
    print(f"时间索引包含 {len(time_index)} 个时间点")
    
    # 使用外连接
    print("\n正在合并数据...")
    merged = time_index.copy()
    
    for col_name, df_tag in all_tag_data:
        # 使用left merge，基于时间索引
        merged = merged.merge(
            df_tag,
            on=["DateDay", "DateHour", "Minute"],
            how="outer"
        )
        print(f"  已合并: {col_name}, 当前行数: {len(merged)}")
    
    # 最终排序
    merged.sort_values(["DateDay", "DateHour", "Minute"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    
    # 计算投加药耗：投加药耗 = (泵投加流量 * 1.486) / 进水瞬时流量
    print("\n正在计算投加药耗...")
    dosing_calc = [
        ("1#投加药耗", "1#泵投加流量", "1#进水瞬时流量"),
        ("2#投加药耗", "2#泵投加流量", "2#进水瞬时流量"),
        ("3#投加药耗", "3#泵投加流量", "3#进水瞬时流量"),
        ("4#投加药耗", "4#泵投加流量", "4#进水瞬时流量"),
        ("5#投加药耗", "5#泵投加流量", "1#进水瞬时流量"),  # 5#使用1#进水流量
    ]
    
    for output_col, pump_col, flow_col in dosing_calc:
        if pump_col in merged.columns and flow_col in merged.columns:
            # 避免除以0：将进水流量为0的情况设置为NaN
            merged[output_col] = np.where(
                merged[flow_col] != 0,
                (merged[pump_col] * 1.486) / merged[flow_col],
                np.nan
            )
            print(f"  已计算: {output_col} = ({pump_col} * 1.486) / {flow_col}")
        else:
            missing_cols = []
            if pump_col not in merged.columns:
                missing_cols.append(pump_col)
            if flow_col not in merged.columns:
                missing_cols.append(flow_col)
            print(f"  跳过 {output_col}: 缺少列 {missing_cols}")
    
    # 保存结果
    print(f"\n正在保存到 {output}...")
    merged.to_csv(output, index=False)
    print(f"数据已保存，总行数: {len(merged)}")
    print("\n前5行数据:")
    print(merged.head())
    print("\n数据信息:")
    print(merged.info())

    return merged

def main():
    load_data(
        start_dt="2024-01-01",
        end_dt="2025-07-01",
        tag_map=TAG_MAP,
        output="E:/Water_Plan_Control/alum/pro_data/merged_data.csv",
        chunksize=100000  # 可根据可用内存调整：内存小用50000，内存大用200000
    )

if __name__ == "__main__":
    main()
# 数据处理，滑动窗口，差分，平滑，小波变换，【异常值处理，缺失值处理（0值和NAN值），平滑处理（突变值）
# 特征工程：统计数据、时滞、组合特征、统计特征等。