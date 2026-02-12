#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Type

default_logger = "chuxiong_tobacco_optimization"
log_dir = "./log"


class Logger:
    """
    简单的日志处理类，将日志同时输出到控制台和文件
    """

    def __init__(self, name=default_logger, log_dir=log_dir, level=logging.INFO,
                 max_bytes=10 * 1024 * 1024 * 1024, backup_count=1):
        """
        初始化日志处理类
        Args:
            name: 日志记录器名称，也是日志文件名的一部分
            log_dir: 日志文件存储目录
            level: 日志级别
            max_bytes: 单个日志文件的最大大小（字节）
            backup_count: 保留的旧日志文件数量
        """
        # 保存 name 属性
        self.name = name

        # 初始化日志缓存列表
        self.log_records = []

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志记录器
        self.logger = logging.getLogger(name)

        # 防止重复添加处理器
        if not self.logger.handlers:
            # 设置日志级别
            self.logger.setLevel(level)

            # 创建文件处理器
            log_file = os.path.join(log_dir, f'{name}.log')
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)

            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # 创建格式化器并添加到处理器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 添加处理器到日志记录器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def debug(self, message):
        """记录调试级别的日志"""
        self.logger.debug(message)
        self._log('DEBUG', message)

    def info(self, message):
        """记录信息级别的日志"""
        self.logger.info(message)
        self._log('INFO', message)

    def warning(self, message):
        """记录警告级别的日志"""
        self.logger.warning(message)
        self._log('WARNING', message)

    def error(self, message):
        """记录错误级别的日志"""
        self.logger.error(message)
        self._log('ERROR', message)

    def critical(self, message):
        """记录严重错误级别的日志"""
        self.logger.critical(message)
        self._log('CRITICAL', message)

    def exception(self, message):
        """记录异常信息，包含堆栈跟踪"""
        self.logger.exception(message)
        self._log('EXCEPTION', message)

    def _log(self, level, message):
        # 将日志信息记录到内存中，不再调用底层 logger
        self.log_records.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'), level, message))

    def get_logs_and_clear(self):
        """
        获取当前所有已记录的日志，并清空缓存
        """
        logs = self.log_records.copy()
        self.log_records.clear()
        return logs

    def raise_log(self, exception: Exception):
        """
        在抛出异常前先记录异常日志
        Args:
            exception: 要抛出的异常实例
        Raises:
            原样抛出传入的异常
        """
        exception_type = str(type(exception)).split("'")[1]
        message = str(exception)
        self.error(f"{exception_type}: {message}")
        raise exception

    def raise_if_not(
            self,
            condition: bool,
            message: str = "",
            exc_type: Type[Exception] = ValueError
    ):
        """
        条件不满足时记录错误日志并抛出异常
        Args:
            condition: 要检查的条件
            message: 错误消息
            exc_type: 要抛出的异常类型，默认为ValueError
        Raises:
            指定的异常类型
        """
        if not condition:
            self.error(f"{exc_type.__name__}: {message}")
            raise exc_type(message)

    def raise_if(
            self,
            condition: bool,
            message: str = "",
            exc_type: Type[Exception] = ValueError
    ):
        """
        条件满足时记录错误日志并抛出异常
        Args:
            condition: 要检查的条件
            message: 错误消息
            exc_type: 要抛出的异常类型，默认为ValueError
        Raises:
            指定的异常类型
        """
        self.raise_if_not(not condition, message, exc_type)


# 示例用法
if __name__ == "__main__":
    # 创建日志记录器实例
    logger = Logger(name='example')

    # 基本日志功能测试
    logger.debug('这是一个调试信息')
    logger.info('这是一个信息')
    logger.warning('这是一个警告')
    logger.error('这是一个错误')
    logger.critical('这是一个严重错误')

    # # 异常记录测试
    # try:
    #     result = 1 / 0
    # except Exception as e:
    #     logger.exception(f'发生异常: {e}')
    #
    # # raise_log 测试
    # try:
    #     try:
    #         raise ConnectionError("连接超时")
    #     except ConnectionError as e:
    #         logger.raise_log(e)
    # except Exception as e:
    #     print(f"捕获到异常: {type(e).__name__}: {e}")
    #
    # # raise_if_not 测试
    # response = type('obj', (object,), {'status_code': 404, 'text': 'Not Found'})  # 模拟响应对象
    # assetCode = "A001"
    # attributeCode = "B002"
    # try:
    #     logger.raise_if_not(
    #         response.status_code == 200,
    #         f"Failed to upload {assetCode}-{attributeCode}: {response.text}",
    #         RuntimeError  # 可以指定不同的异常类型
    #     )
    # except Exception as e:
    #     print(f"捕获到异常: {type(e).__name__}: {e}")
    #
    # # raise_if 测试
    # try:
    #     logger.raise_if(
    #         response.status_code != 200,
    #         f"上传失败: {assetCode}-{attributeCode}",
    #         RuntimeError
    #     )
    # except Exception as e:
    #     print(f"捕获到异常: {type(e).__name__}: {e}")
