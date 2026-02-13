import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from datetime import datetime
import re


_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from dataio.data_factory import load_agg_data
from utils.config_loader import load_alum_config


def read_data(
    model_name: str = 'optimized_dose',
    config_file: str = 'configs/alum_dosing.yaml',
    num_pools: int = 4,
    debug: bool = False,
    mode: str = 'remote',
    app_config: Dict[str, Any] = None,
    seed: int = None,
) -> Tuple[Dict[str, np.ndarray], datetime]:
    """
    读取聚合数据并按池子分组转换为字典格式
    
    参数：
        model_name: 模型名称，可选值：
                   - 'effluent_turbidity': 沉淀池出水浊度预测模型
                   - 'optimized_dose': 投加药耗优化模型（默认）
        config_file: 配置文件路径
        num_pools: 池子数量，默认为4
        mode: 数据模式，'remote' 或 'local'
    
    返回：
        Tuple[Dict[str, np.ndarray], datetime]：
        - 第一个元素：Dict[str, np.ndarray]
          - key: "pool_1", "pool_2", "pool_3", "pool_4"
          - value: shape为 (n_timesteps, n_features) 的数组
        - 第二个元素：datetime，数据的最后时间点
    """
    mode_norm = str(mode or 'remote').strip().lower()

    if mode_norm == 'local':
        cfg = app_config or {}
        seq_len = int(cfg.get('seq_len', 60))
        feature_order = cfg.get('features', ['dose', 'turb_chushui', 'turb_jinshui', 'flow', 'pH', 'temp_shuimian'])

        pools_cfg = cfg.get('pools', {})
        enabled_pools = [pid for pid, p_cfg in pools_cfg.items() if p_cfg.get('enabled', False)]
        if not enabled_pools:
            enabled_pools = [f'pool_{i}' for i in range(1, num_pools + 1)]

        rng = np.random.default_rng(seed)
        local_result: Dict[str, np.ndarray] = {}
        for pool_name in enabled_pools:
            pool_data_list = []
            for feature_name in feature_order:
                if feature_name == 'dose':
                    values = rng.uniform(5.0, 30.0, size=seq_len)
                elif feature_name in ('pH', 'ph'):
                    values = rng.uniform(6.8, 7.6, size=seq_len)
                elif feature_name == 'flow':
                    values = rng.uniform(1500.0, 3000.0, size=seq_len)
                elif feature_name in ('turb_chushui', 'turb_jinshui'):
                    values = rng.uniform(0.1, 8.0, size=seq_len)
                elif feature_name == 'temp_shuimian':
                    values = rng.uniform(10.0, 30.0, size=seq_len)
                else:
                    values = rng.uniform(0.0, 1.0, size=seq_len)
                pool_data_list.append(values)

            local_result[pool_name] = np.column_stack(pool_data_list).astype(np.float32)

        if debug:
            print(f"[read_data][local] pools={list(local_result.keys())}, seq_len={seq_len}, n_features={len(feature_order)}")

        return local_result, datetime.now()

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
        # 重命名列
        data = data.rename(columns=column_mapping)
    else:
        raise ValueError("远程读数为空，无法构建模型输入")
    
    # 固定的特征顺序
    FEATURE_ORDER = ['dose', 'turb_chushui', 'turb_jinshui', 'flow', 'pH', 'temp_shuimian']
    
    # 保存最后一个时间点
    last_time = None
    if 'time' in data.columns:
        last_time = pd.to_datetime(data['time'].iloc[-1])
        data = data.drop(columns=['time'])
    elif isinstance(data.index, pd.DatetimeIndex) or data.index.name == 'time':
        last_time = pd.to_datetime(data.index[-1])
        data = data.reset_index(drop=True)
    
    # 如果没有找到时间信息，使用当前时间
    if last_time is None:
        last_time = datetime.now()
    else:
        # 转换为 datetime 对象
        last_time = last_time.to_pydatetime()
    
    result: Dict[str, np.ndarray] = {}

    # 构建列映射：(feature_base_lower, pool_id|None) -> 原始列名
    col_map: Dict[Tuple[str, int | None], str] = {}
    for col in data.columns:
        # 将列名解析为 (feature_base, pool_id)
        # - turb_jinshui_1 -> ("turb_jinshui", 1)
        # - temp_shuimian  -> ("temp_shuimian", None)
        col_lower = str(col).strip()
        m = re.match(r'^(?P<base>.+)_(?P<pool>\d+)$', col_lower)
        if m:
            base, pid = m.group('base'), int(m.group('pool'))
        else:
            base, pid = col_lower, None
        col_map[(base, pid)] = col

    # 为每个池子构建数据
    for pool_id in range(1, num_pools + 1):
        pool_key = f"pool_{pool_id}"
        pool_data_list = []

        if debug:
            print(f"\n处理 {pool_key}:")

        # 按固定顺序提取每个特征：优先取 feature_pool（如 dose_1），否则取共享列（如 temp_shuimian）
        for feature_name in FEATURE_ORDER:
            feature_key = str(feature_name).strip()
            matched_col = col_map.get((feature_key, pool_id)) or col_map.get((feature_key, None))

            if matched_col is not None:
                values = data[matched_col].values
                pool_data_list.append(values)
                if debug:
                    vmin = np.nanmin(values) if len(values) else np.nan
                    vmax = np.nanmax(values) if len(values) else np.nan
                    print(f"  特征 '{feature_name}' -> 列 '{matched_col}' (min={vmin}, max={vmax})")
            else:
                pool_data_list.append(np.full(len(data), np.nan))
                if debug:
                    print(f"  特征 '{feature_name}' -> 未找到列，填充 NaN")
        
        # 将列表转换为 (n_timesteps, 6) 的数组
        pool_array = np.column_stack(pool_data_list)
        result[pool_key] = pool_array
    
    return result, last_time
