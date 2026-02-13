# -*- encoding: utf-8 -*-
"""
统一 API 响应封装。
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from flask import jsonify

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _now_str(time_format: str = TIME_FORMAT) -> str:
    return datetime.now().strftime(time_format)


def ok_response(
    data: Dict[str, Any],
    message: str = "success",
    code: str = "OK",
    status_code: int = 200,
    timestamp: Optional[str] = None,
) -> Tuple[Any, int]:
    return (
        jsonify(
            {
                "success": True,
                "code": code,
                "message": message,
                "data": data,
                "meta": {"timestamp": timestamp or _now_str()},
            }
        ),
        status_code,
    )


def error_response(
    code: str,
    message: str,
    detail: str,
    status_code: int = 500,
    timestamp: Optional[str] = None,
) -> Tuple[Any, int]:
    return (
        jsonify(
            {
                "success": False,
                "code": code,
                "message": message,
                "error": {"detail": detail},
                "meta": {"timestamp": timestamp or _now_str()},
            }
        ),
        status_code,
    )
