from .base_optimizer import BaseOptimizer
from .dummy_optimizer import DummyOptimizer
from .dosing_optimizer import (
    DosingOptimizer, 
    create_dosing_optimizer,
    create_multi_pool_optimizers
)

# 保留向后兼容性
MPCController = DosingOptimizer


def create_optimizer(optimizer_type: str = 'dummy', config: dict = None, pool_id: str = None, **kwargs) -> BaseOptimizer:
    """
    创建优化器实例（工厂函数）
    
    参数：
        optimizer_type: 优化器类型 ('dummy', 'mpc', 'dosing' 等)
        config: 配置字典
        pool_id: 池子ID（用于MPC等优化器）
        **kwargs: 其他参数
        
    返回：
        BaseOptimizer: 优化器实例
    """
    if config is None:
        config = {}
    
    # 如果没有提供pool_id，默认使用'pool_1'
    if pool_id is None:
        pool_id = 'pool_1'
        
    if optimizer_type == 'dummy':
        return DummyOptimizer(config, pool_id)
    elif optimizer_type in ['mpc', 'dosing']:
        return DosingOptimizer(config, pool_id)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


__all__ = [
    'BaseOptimizer',
    'DummyOptimizer',
    'DosingOptimizer',
    'MPCController',  # 向后兼容
    'create_optimizer',
    'create_dosing_optimizer',
    'create_multi_pool_optimizers',
]
