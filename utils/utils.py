#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@License :   (C)Copyright Baidu2022, bce-iot

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2025/02/22        nanyuze    1.0         utils
"""
import os
from dateutil import tz
import configparser
from utils.logger import Logger

# set time zone info
logger = Logger()
timezone = "Asia/Shanghai"
tz_sh = tz.gettz(timezone)
time_fmt = "%Y-%m-%d"
default_time_fmt = "%Y-%m-%d %H:%M:%S"


def load_config(config_path="./configs/config.ini"):
    """
    加载配置文件
    :param config_path:
    :return:
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    config_dict = {}
    # 首先处理DEFAULT节（如果有）
    if config.defaults():
        config_dict['DEFAULT'] = dict(config.defaults())
    # 处理其他所有节
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            config_dict[section][key] = value

    # 只处理非DEFAULT节
    for section in config.sections():
        config_dict[section] = dict(config[section])

    return config_dict


config = load_config()

if __name__ == '__main__':
    ...
