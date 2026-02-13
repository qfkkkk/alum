"""
预测器模块 (Predictors)

职责：加载训练好的模型权重，执行推理预测
模式：策略模式 + 模板方法模式
  - 策略模式：不同预测算法（xPatch/LSTM等）可互换
  - 模板方法：BasePredictor.predict() 定义通用流程，子类只实现 _infer()

包含：
    - BasePredictor: 预测器抽象基类（单池）
    - TurbidityPredictor: xPatch 出水浊度预测器（单池）
    - TurbidityPredictorManager: 多池预测管理器
"""
from .base_predictor import BasePredictor
from .turbidity_predictor import TurbidityPredictor
from .predictor_manager import TurbidityPredictorManager

# 已注册的预测器类型
_PREDICTOR_REGISTRY = {
    'xPatch': TurbidityPredictor,
}


def create_predictor(predictor_type: str, **kwargs) -> BasePredictor:
    """
    预测器工厂函数（单池）

    参数：
        predictor_type: 预测器类型（对应模型架构名）
            - 'xPatch': xPatch 出水浊度预测器
        **kwargs: 传递给预测器构造函数的参数
            - pool_id: 池子编号
            - config: 配置字典

    返回：
        BasePredictor: 预测器实例

    使用示例：
        predictor = create_predictor('xPatch', pool_id=1, config=cfg)
    """
    if predictor_type not in _PREDICTOR_REGISTRY:
        raise ValueError(
            f"未知的预测器类型: {predictor_type}，"
            f"可用类型: {list(_PREDICTOR_REGISTRY.keys())}"
        )
    return _PREDICTOR_REGISTRY[predictor_type](**kwargs)


def create_manager(config_path: str = None) -> TurbidityPredictorManager:
    """
    创建多池预测管理器

    参数：
        config_path: app.yaml 路径，默认为 configs/app.yaml

    返回：
        TurbidityPredictorManager: 管理器实例

    使用示例：
        manager = create_manager()
        result = manager.predict_all(data_dict, last_datetime)
    """
    return TurbidityPredictorManager(config_path=config_path)
