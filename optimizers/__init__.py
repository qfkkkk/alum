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
