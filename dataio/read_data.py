import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict


_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from dataio.data_factory import load_agg_data
from utils.config_loader import load_alum_config

def read_data(model_name='optimized_dose', config_file='configs/alum_dosing.yaml'):
    """
    测试读取聚合数据
    
    参数：
        model_name: 模型名称，可选值：
                   - 'effluent_turbidity': 沉淀池出水浊度预测模型
                   - 'optimized_dose': 投加药耗优化模型（默认）
        config_file: 配置文件路径
    """
    # 加载配置文件
    print(f"加载配置文件: {config_file}")
    config = load_alum_config(config_file)
    
    # 构建测点列表和列名映射字典
    model_config = config['input_point'][model_name]
    attributes = []
    column_mapping = {}  # 用于存储 AttributeCode -> friendly_name 的映射
    
    for asset_code, point_config in model_config.items():
        for attribute_code, friendly_name in point_config.items():
            attributes.append({
                "AssetCode": asset_code,
                "AttributeCode": attribute_code
            })
            # 构建列名映射：AttributeCode -> friendly_name
            column_mapping[attribute_code] = friendly_name
    
    
    # 读取数据
    data = load_agg_data(attributes, minutes=60 * 5, interval="5min")
    
    # 重命名列名：将 AttributeCode 替换为 friendly_name
    if data is not None and not data.empty:
        
        # 重命名列（只重命名存在于映射字典中的列）
        data = data.rename(columns=column_mapping)

    return data
def transform_data_by_pool(df: pd.DataFrame, num_pools: int = 4) -> Dict[str, np.ndarray]:
    """
    将DataFrame数据按池子分组转换为 Dict[str, np.ndarray]，不包含时间特征。
    
    Args:
        df: 包含时间列和各池子数据的DataFrame
            列名格式示例: time, FTJZ_Turb_1_Instant, FTJZ_Turb_2_Instant, ...
        num_pools: 池子数量，默认为4
    
    Returns:
        Dict[str, np.ndarray]：
        - key: "pool_1", "pool_2", "pool_3", "pool_4"
        - value: shape为 (n_timesteps, 6) 的数组
        - 6个特征列的固定顺序：dose, turb_chushui, turb_jinshui, flow, pH, temp_shuimian
    
    Example:
        输入: DataFrame with columns [time, dose_1, turb_chushui_1, ...]
        输出: {"pool_1": ndarray[60, 6], "pool_2": ndarray[60, 6], ...}
    """
    # 固定的特征顺序（不包含时间）
    FEATURE_ORDER = ['dose', 'turb_chushui', 'turb_jinshui', 'flow', 'pH', 'temp_shuimian']
    
    # 如果 time 在列中，去掉它（我们不需要时间特征）
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
    
    # 如果 time 是索引，重置索引但不保留时间列
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name == 'time':
        df = df.reset_index(drop=True)
    
    result: Dict[str, np.ndarray] = {}
    
    # 为每个池子构建数据
    for pool_id in range(1, num_pools + 1):
        pool_key = f"pool_{pool_id}"
        pool_data_list = []
        
        # 按固定顺序提取每个特征
        for feature_name in FEATURE_ORDER:
            # 尝试匹配列名：可能的格式包括 feature_name_池子号 或包含池子号的其他格式
            matched_col = None
            
            # 遍历所有列，找到属于当前池子和当前特征的列
            for col in df.columns:
                col_lower = col.lower()
                feature_lower = feature_name.lower()
                
                # 检查列名是否包含特征名和池子编号
                # 例如: dose_1, turb_chushui_1, FTJZ_dose_1_Instant, pH_1 等
                has_pool_id = f'_{pool_id}' in col or f'_{pool_id}_' in col
                has_feature = feature_lower in col_lower
                
                if has_pool_id and has_feature:
                    matched_col = col
                    break
            
            # 如果找到匹配的列，提取数据；否则填充 NaN
            if matched_col is not None:
                pool_data_list.append(df[matched_col].values)
            else:
                # 如果某个特征在该池子中不存在，填充 NaN
                pool_data_list.append(np.full(len(df), np.nan))
        
        # 将列表转换为 (n_timesteps, 6) 的数组
        pool_array = np.column_stack(pool_data_list)
        result[pool_key] = pool_array
    
    return result
