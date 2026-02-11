'''
Author: wangzhuoyang wangzhuoyang@ciictec.com
Date: 2026-02-11 16:20:23
LastEditors: wangzhuoyang wangzhuoyang@ciictec.com
LastEditTime: 2026-02-11 17:56:38
FilePath: /alum/optimizers/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from .base_optimizer import BaseOptimizer
from .dummy_optimizer import DummyOptimizer

def create_optimizer(optimizer_type: str = 'dummy', config: dict = None) -> BaseOptimizer:
    """
    创建优化器实例
    
    参数：
        optimizer_type: 优化器类型 ('dummy', 'mpc', 'rl' 等)
        config: 配置字典
        
    返回：
        BaseOptimizer: 优化器实例
    """
    if config is None:
        config = {}
        
    if optimizer_type == 'dummy':
        return DummyOptimizer(config)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
