#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   preprocess.py
@License :   (C)Copyright Baidu2024, bce-iot

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
2024/11/25        nanyuze     1.0         preprocess
"""
import yaml
import numpy as np
import pandas as pd
import time
from functools import reduce
from tqdm import tqdm
from datetime import datetime
from .data_loader import read_history_value_batch, write_value_batch
# from data.data_mysql import read_data, write_data
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import Logger
from pandas import DataFrame

logger = Logger()
time_format = '%Y-%m-%d %H:%M:%S'

# 配置文件路径 - 支持 alum_dosing.yaml
path = Path(__file__)
# 从 dataio/data_factory.py -> alum/ -> configs/alum_dosing.yaml
data_config_path = path.parent.parent / 'configs' / 'alum_dosing.yaml'
print('data_config_path:', data_config_path)

# 延迟加载配置，避免文件不存在时报错
control_config = {}
try:
    with open(data_config_path, "r", encoding="utf-8") as file:
        control_config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"警告：配置文件不存在: {data_config_path}")


def write_logger_to_mysql(logger_instance):
    """
    将指定 Logger 实例中的日志记录写入 business_mm_model_log 表中。
    :param logger_instance: Logger 实例
    """
    logs = logger_instance.get_logs_and_clear()

    if not logs:
        print("没有新日志需要写入数据库。")
        return

    # 获取模型编码（即 logger.name）
    model_code = logger_instance.logger.name
    # 构造 DataFrame
    df = DataFrame(logs, columns=['log_time', 'level_name', 'message'])
    df['model_code'] = model_code
    df.rename(columns={'level_name': 'level', 'message': 'log'}, inplace=True)
    df = df[['model_code', 'level', 'log']]  # 保留需要的三列

    # 插入语句，只写入 model_code, level, log
    table_name = "business_mm_model_log"
    sql = f"""
    INSERT INTO {table_name} (model_code, level, log)
    VALUES (%s, %s, %s)
    """

    # 调用封装好的 write_data 方法
    write_data(sql, df)


def load_model_data(point_type, product_line, minutes=10, interval="5S", flag='history'):
    """
    根据测点类型和产线获取历史数据，并对数据进行处理。
    Args:
        point_type (str): 数据类型。
        product_line (str): 工艺段。
        minutes (int, optional): 时间范围，以分钟为单位。默认为10分钟。
    Returns:
        pandas.DataFrame: 处理后的历史数据。
    """
    data_config = control_config[point_type][product_line]

    all_data = pd.DataFrame()
    # for point in tqdm(data_config):
    for point in data_config:
        try:
            arg = {"AssetCode": product_line,
                   "AttributeCode": point}
            data = read_history_value_batch([arg], minutes * 60, flag)
            data.columns = [data_config[point]]
            data.reset_index(inplace=True)
            all_data = data if all_data.empty else pd.merge(all_data, data, on='time', how='outer')
        except Exception:
            logger.info(f"{data_config[point]}: 数据获取异常")
            all_data[data_config[point]] = np.nan
    # 数据重新采样
    all_data.reset_index(inplace=True)
    all_data['time'] = pd.to_datetime(
        all_data['time'].apply(lambda s: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(s) / 1000))))
    # all_data = all_data.resample(interval, on='time').last()
    all_data = all_data.set_index('time').resample(interval).last()
    all_data.reset_index(inplace=True, drop=False)

    return all_data


def load_agg_data(attributes, minutes=10, interval="1min"):
    all_data = pd.DataFrame()
    for attr in attributes:
        try:
            data = read_history_value_batch([attr], duration=minutes * 60)
            data.reset_index(inplace=True)
            all_data = data if all_data.empty else pd.merge(all_data, data, on='time', how='outer')
        except Exception as e:
            print(f"{attr['AttributeCode']}: 数据获取异常")
            all_data[attr['AttributeCode']] = np.nan
    # 数据重新采样
    all_data.reset_index(inplace=True)
    all_data['time'] = pd.to_datetime(
        all_data['time'].apply(lambda s: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(s) / 1000))))
    all_data = all_data.set_index('time').resample(interval).last()
    all_data.reset_index(inplace=True, drop=False)
    all_data.drop(['index'], axis=1, inplace=True)
    return all_data


def load_batch_data(data, param_mapping):
    """根据参数映射加载批量数据。"""
    data_list = []
    for param_id, col_name in param_mapping.items():
        temp_df = data[data['parameter_id'] == param_id][['batch_code', 'value']]
        temp_df.rename(columns={'value': col_name}, inplace=True)
        data_list.append(temp_df)
    data1 = reduce(lambda left, right: pd.merge(left, right, on='batch_code', how='outer'), data_list)
    return data1


def load_silo_data(proline_code, process_code, is_live=False):
    """
    获取规整后贮柜数据
    C20-超回松散；R20-润叶加料；H20-叶丝回潮；H32-薄板烘丝；H41-气流烘丝；P10-掺配；J20-混合加香
    J30-储丝 （成品丝储存柜）；P20-混配 （混丝暂存柜）；R30-配叶贮叶（二储）；C30-预配储叶（一储）
    """
    table_name = "business_dm_batch_module_data"
    data = read_data(
        f"select * from {table_name} where proline_code='{proline_code}' and process_code='{process_code}' ")
    required_variables = ['batch_code', 'in_weight', 'in_num', 'in_end_time']
    agg_mapping = {'in_num': 'first',
                   'in_weight': 'sum',
                   'in_end_time': 'max'}
    if not is_live:
        required_variables += ['out_start_time']
        agg_mapping['out_start_time'] = 'min'

    data = data[required_variables].dropna()
    data.groupby('batch_code').agg(agg_mapping).reset_index(inplace=True)
    data['in_num'] = 'MB' + data['in_num'].astype(str)
    return data


def upload_recommend_message(control_map):
    """
    反控推荐值数据
    :param control_map:
    :return:
    """
    send_message = []

    for instance in control_map:
        args = {
            "AssetCode": instance,
            "Attributes": []
        }
        for point in control_map[instance]:
            value = control_map[instance][point]
            # 每个推荐测点为一个实例
            args["Attributes"].append(
                {
                    "Code": str(point),
                    "Value": str(value)
                }
            )
        send_message.append(args)

    response = write_value_batch(send_message)
    logger.raise_if_not(response.status_code == 200,
                        "Failed to upload control messages: {}".format(control_map))


if __name__ == '__main__':
    point_type = "input_point"
    product_line = "ZS12C20"
    data = load_model_data(point_type, product_line, minutes=100, interval="5S")
    print(data)

    # model_data_map = {'model_data': {'control_change_inner': 10, 'total_change_inner': 10}}
    # upload_recommend_message(model_data_map)
    # logger = Logger(name='test_model')
    #
    # logger.info("这是一个测试日志。")
    # logger.warning("这是一个警告日志。")
    #
    # write_logger_to_mysql(logger)
