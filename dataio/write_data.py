import sys
from pathlib import Path
from typing import Any, Dict

try:
    from .data_factory import upload_recommend_message
except ImportError:
    # 直接运行时，将项目根目录加入 sys.path
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from dataio.data_factory import upload_recommend_message


def write_data(model_name:str, result: Any, mode: str = 'remote') -> Dict[str, object]:
    """
    写入推荐投加量数据
    
    :param model_name: 模型名称，可选值：
                      - 'optimized_dose': 投加药耗优化模型
                      - 'effluent_turbidity': 沉淀池出水浊度预测模型
    :param result: 模型输出结果，支持 dict 或 str。
                  dict 格式示例：{"pool_id": {datetime: x}}
                  其中 pool_id 为池子ID，datetime 为时间戳，x 为预测值
    :param mode: 数据模式，'remote' 或 'local'
    :return: Dict，包含写入状态
    """
    mode_norm = str(mode or 'remote').strip().lower()
    result_str = str(result)

    if mode_norm == 'local':
        print("[write_data][local] model_name={}, payload={}".format(
            model_name, result_str
        ))
        return {"success": True, "skipped": True, "mode": "local", "model_name": model_name}

    if model_name == 'optimized_dose':
        upload_recommend_message(
            {
                "ALUM_DOSING_ALG_OUTPUT": {
                    "ALUM_DOSING_ALG_MODEL": result_str
                }
            }
        )
    elif model_name == 'effluent_turbidity':
        upload_recommend_message(
            {
                "ALUM_DOSING_ALG_OUTPUT": {
                    "EFFLUENT_TURBIDITY_FORECAST_MODEL": result_str
                }
            }
        )
    else:
        raise ValueError("模型名称不支持写入数据: {}".format(model_name))

    return {"success": True, "skipped": False, "mode": "remote", "model_name": model_name}
