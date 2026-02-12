import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict

try:
    from .data_factory import load_agg_data, upload_recommend_message
except ImportError:
    # 直接运行时，将项目根目录加入 sys.path
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from dataio.data_factory import load_agg_data, upload_recommend_message


def write_data(model_name, result) -> None:
    """
    写入推荐投加量数据
    
    :param model_name: 模型名称，可选值：
                      - 'optimized_dose': 投加药耗优化模型
                      - 'effluent_turbidity': 沉淀池出水浊度预测模型
    :param result: 模型输出结果
    :return: None
    """
    if model_name == 'optimized_dose':
        upload_recommend_message(
            {
                "ALUM_DOSING_ALG_OUTPUT": {
                    "ALUM_DOSING_ALG_MODEL": '{}'.format(result)
                }
            }
        )
    elif model_name == 'effluent_turbidity':
        upload_recommend_message(
            {
                "ALUM_DOSING_ALG_OUTPUT": {
                    "EFFLUENT_TURBIDITY_FORECAST_MODEL": '{}'.format(result)
                }
            }
        )
    else:
        raise ValueError("模型名称不支持写入数据: {}".format(model_name))


