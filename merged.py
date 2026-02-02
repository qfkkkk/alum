"""
CSV文件合并工具
功能：将不同目录下的多个CSV文件合并为一个，并按指定列排序后保存
"""

import pandas as pd
import os
from pathlib import Path


def merge_csv_files(
    file_paths: list,
    output_path: str,
    sort_columns: list = None,
    ascending: bool = True,
    encoding: str = 'utf-8',
    sep: str = ',',
    drop_duplicates: bool = False,
    reset_index: bool = True
):
    """
    合并多个CSV文件并按指定列排序
    
    参数:
        file_paths: CSV文件路径列表（支持绝对路径和相对路径）
        output_path: 输出文件路径
        sort_columns: 排序的列名列表，例如 ['时间', '流量']
        ascending: True为升序，False为降序，也可以传入列表对应每列
        encoding: 文件编码，默认utf-8，中文文件可能需要'gbk'或'gb2312'
        sep: 分隔符，默认逗号
        drop_duplicates: 是否删除重复行
        reset_index: 是否重置索引
    
    返回:
        合并后的DataFrame
    """
    
    dataframes = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 - {file_path}")
            continue
        
        try:
            # 尝试读取CSV文件
            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
            df['_source_file'] = os.path.basename(file_path)  # 添加来源文件标记
            dataframes.append(df)
            print(f"成功读取: {file_path}, 行数: {len(df)}")
        except UnicodeDecodeError:
            # 如果编码错误，尝试其他编码
            for enc in ['gbk', 'gb2312', 'utf-8-sig', 'latin1']:
                try:
                    df = pd.read_csv(file_path, encoding=enc, sep=sep)
                    df['_source_file'] = os.path.basename(file_path)
                    dataframes.append(df)
                    print(f"成功读取: {file_path} (编码: {enc}), 行数: {len(df)}")
                    break
                except:
                    continue
            else:
                print(f"错误: 无法读取文件 - {file_path}")
        except Exception as e:
            print(f"错误: 读取文件失败 - {file_path}, 原因: {e}")
    
    if not dataframes:
        print("没有成功读取任何文件！")
        return None
    
    # 合并所有DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"\n合并完成，总行数: {len(merged_df)}")
    
    # 删除重复行
    if drop_duplicates:
        original_len = len(merged_df)
        merged_df = merged_df.drop_duplicates()
        print(f"删除重复行: {original_len - len(merged_df)} 行")
    
    # 按指定列排序
    if sort_columns:
        try:
            merged_df = merged_df.sort_values(by=sort_columns, ascending=ascending)
            print(f"按列排序: {sort_columns}")
        except KeyError as e:
            print(f"警告: 排序列不存在 - {e}")
            print(f"可用列: {list(merged_df.columns)}")
    
    # 重置索引
    if reset_index:
        merged_df = merged_df.reset_index(drop=True)
    
    # 保存文件
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        merged_df.to_csv(output_path, index=False, encoding=encoding)
        print(f"\n文件已保存: {output_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")
    
    return merged_df


def find_csv_files(directory: str, pattern: str = "*.csv", recursive: bool = True):
    """
    在指定目录中查找所有CSV文件
    
    参数:
        directory: 目录路径
        pattern: 文件匹配模式
        recursive: 是否递归搜索子目录
    
    返回:
        CSV文件路径列表
    """
    path = Path(directory)
    if recursive:
        files = list(path.rglob(pattern))
    else:
        files = list(path.glob(pattern))
    
    return [str(f) for f in files]


# ============ 使用示例 ============
if __name__ == "__main__":
    
    # 示例1: 指定文件列表合并
    # --------------------------
    # 定义要合并的CSV文件路径
    csv_files = [
        r"data/flowpress_202501-202507.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202401_202508051443.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202402_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202403_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202404_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202405_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202406_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202407_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202408_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202409_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202410_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202411_202508051444.csv",
        r"G:\工作-ZGHL\广水智控项目\数据资料\流量数据\flowpress_202412_202508051444.csv",
        # 添加更多文件路径...
    ]
    
    
    # 示例3: 按多列排序
    # --------------------------
    merged_df = merge_csv_files(
        file_paths=csv_files,
        output_path="data/flowpress_202401-202507.csv",
        sort_columns=['DateDay', 'DateHour'],  # 多列排序
        ascending=[True, True],  # 分别指定每列的排序方向
        encoding='gbk'  # 中文Windows可能需要gbk编码
    )
    print(merged_df.tail())
    
    print("\n脚本执行完成！")
    print("请根据需要修改上面的参数，然后取消注释相应的代码块来执行合并操作。")
