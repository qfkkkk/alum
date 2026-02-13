# -*- encoding: utf-8 -*-
"""
IO 适配层。

职责：
    - 统一读取输入数据（local/remote）
    - 统一写入推荐结果（local/remote）
"""

from __future__ import annotations

import json
from datetime import datetime
from importlib import import_module
from typing import Any, Dict, Tuple

import numpy as np

from utils.config_loader import load_config
from utils.logger import Logger

logger = Logger()


def _get_io_cfg(config: Dict[str, Any] | None) -> Dict[str, Any]:
    if isinstance(config, dict):
        io_cfg = config.get("dataio", {})
        if isinstance(io_cfg, dict):
            return io_cfg
    return {}


def _extract_time_series_payload(
    payload: Dict[str, Any],
    field_name: str,
    allow_direct_map: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    从 pipeline 结果中提取按池子组织的时序值：
    - 优化结果: field_name='recommendations'
    - 预测结果: field_name='predictions'
    - 纯预测输出: {pool_id: {datetime: value}}（兼容）
    """
    extracted: Dict[str, Dict[str, float]] = {}
    for pool_id, value in (payload or {}).items():
        if not isinstance(value, dict):
            continue

        if isinstance(value.get(field_name), dict):
            series = value[field_name]
        elif allow_direct_map:
            # 兼容纯预测输出：pool 下直接是 {datetime: value}
            series = value
        else:
            continue

        try:
            extracted[pool_id] = {str(k): float(v) for k, v in series.items()}
        except Exception:
            continue
    return extracted


def read_data(config: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], datetime]:
    """
    读取输入数据。

    mode=remote: 调用 dataio.read_data 的远程逻辑
    mode=local : 调用 dataio.read_data 的本地假数据逻辑
    """
    io_cfg = _get_io_cfg(config)
    mode = str(io_cfg.get("mode", "remote")).strip().lower()

    model_name = str(io_cfg.get("read_model_name", "optimized_dose"))
    remote_cfg_file = str(io_cfg.get("remote_config_file", "configs/alum_dosing.yaml"))
    debug = bool(io_cfg.get("debug", False))
    seed = io_cfg.get("local_seed")

    pools_cfg = config.get("pools", {}) if isinstance(config, dict) else {}
    enabled_pool_count = len([k for k, v in pools_cfg.items() if isinstance(v, dict) and v.get("enabled", False)])
    num_pools = enabled_pool_count or 4

    try:
        read_module = import_module("dataio.read_data")
        read_func = getattr(read_module, "read_data")
        return read_func(
            model_name=model_name,
            config_file=remote_cfg_file,
            num_pools=num_pools,
            debug=debug,
            mode=mode,
            app_config=config,
            seed=seed,
        )
    except Exception as exc:
        logger.warning(f"[IO Adapter] dataio.read_data 导入失败，回退本地假数据: {exc}")
        seq_len = int(config.get("seq_len", 60)) if isinstance(config, dict) else 60
        features = config.get("features", []) if isinstance(config, dict) else []
        n_features = max(1, len(features))
        pools = [f"pool_{idx}" for idx in range(1, num_pools + 1)]
        rng = np.random.default_rng(seed)
        data_dict = {pool: rng.random((seq_len, n_features), dtype=np.float32) for pool in pools}
        return data_dict, datetime.now()


def upload_recommend_message(
    payload: Dict[str, Any],
    config: Dict[str, Any] | None = None,
    write_targets: Tuple[str, ...] = ("optimize", "predict"),
) -> Dict[str, Any]:
    """
    推荐结果写入。

    mode=remote: 调用 dataio.write_data 远程写库逻辑
    mode=local : 仅打印，不落库
    """
    cfg = config if isinstance(config, dict) else load_config()
    io_cfg = _get_io_cfg(cfg)
    mode = str(io_cfg.get("mode", "remote")).strip().lower()
    optimize_model_name = str(io_cfg.get("write_model_name_optimize", io_cfg.get("write_model_name", "optimized_dose")))
    predict_model_name = str(io_cfg.get("write_model_name_predict", "effluent_turbidity"))

    payload_dict = payload if isinstance(payload, dict) else {}
    optimize_payload = {}
    predict_payload = {}
    if "optimize" in write_targets:
        optimize_payload = _extract_time_series_payload(payload_dict, "recommendations", allow_direct_map=False)
    if "predict" in write_targets:
        predict_payload = _extract_time_series_payload(payload_dict, "predictions", allow_direct_map=True)

    # 避免重复：优化 payload 中是推荐值，预测 payload 中是预测值；若两者都为空则跳过
    if not optimize_payload and not predict_payload:
        logger.info(f"[IO Adapter] 无可写入结果，write_targets={write_targets}")
        return {"success": True, "skipped": True, "mode": mode, "reason": "no_payload"}

    if optimize_payload:
        logger.info(
            f"[IO Adapter] optimize result payload: "
            f"{json.dumps(optimize_payload, ensure_ascii=False)}"
        )
    if predict_payload:
        logger.info(
            f"[IO Adapter] predict result payload: "
            f"{json.dumps(predict_payload, ensure_ascii=False)}"
        )

    writes = []
    try:
        write_module = import_module("dataio.write_data")
        write_func = getattr(write_module, "write_data")

        if optimize_payload:
            writes.append(write_func(model_name=optimize_model_name, result=optimize_payload, mode=mode))

        if predict_payload:
            writes.append(write_func(model_name=predict_model_name, result=predict_payload, mode=mode))
    except Exception as exc:
        logger.warning(f"[IO Adapter] dataio.write_data 导入失败，跳过上报: {exc}")
        return {"success": True, "skipped": True, "mode": "local", "reason": "write_module_unavailable"}

    if mode == "local":
        logger.info("[IO Adapter] upload_recommend_message local print completed")

    return {
        "success": True,
        "skipped": mode == "local",
        "mode": mode,
        "writes": writes,
    }
