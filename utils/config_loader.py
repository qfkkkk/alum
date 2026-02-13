# -*- encoding: utf-8 -*-
"""
配置加载工具
功能：统一加载项目 YAML 配置文件
"""
import yaml
from pathlib import Path
from typing import Dict, Union
from .logger import Logger

logger = Logger(name='config_loader')


def load_config(config_path: Union[str, Path] = None) -> Dict:
    """
    加载 YAML 配置文件

    参数：
        config_path: 配置文件路径
            - 如果为 None，默认加载 configs/app.yaml
            - 支持绝对路径或相对于项目根目录的路径

    返回：
        Dict: 配置字典

    异常：
        FileNotFoundError: 文件不存在时抛出
    """
    if config_path is None:
        # 默认路径：项目根目录/configs/app.yaml
        # 假设当前文件在 utils/config_loader.py，即 web_root/alum/utils/config_loader.py
        # 项目根目录 web_root/alum
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'configs' / 'app.yaml'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_alum_config(config_file: str = 'configs/alum_dosing.yaml') -> Dict:
    """
    加载投矾系统配置文件
    
    Args:
        config_file: 配置文件路径（相对于项目根目录）
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML 解析失败
    """
    try:
        # 获取项目根目录
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_file
        
        logger.debug(f"尝试加载配置文件: {config_path}")
        
        # 验证配置文件存在
        logger.raise_if_not(
            config_path.exists(),
            f"配置文件不存在: {config_path}",
            FileNotFoundError
        )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"配置文件加载成功: {config_file}")
        logger.debug(f"配置内容包含的模型: {list(config.get('input_point', {}).keys())}")
        
        return config
        
    except (FileNotFoundError, yaml.YAMLError):
        
        raise
    except Exception as e:
        logger.exception(f"配置文件加载时发生未知错误: {config_file}")
        raise
