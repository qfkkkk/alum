# -*- encoding: utf-8 -*-
"""
配置加载工具
功能：统一加载项目 YAML 配置文件
"""
import yaml
from pathlib import Path
from typing import Dict, Union


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
