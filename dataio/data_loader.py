# -*- encoding: utf-8 -*-
"""
@File    :   get_dataframe.py
@License :   (C)Copyright Baidu2024, bce-iot

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
2025/02/22        nanyuze      1.0         get dataframe
"""
import hashlib
import time
import hmac
import base64
import requests
import json
from datetime import datetime, timedelta, timezone
from dateutil.tz import tzlocal
import pandas as pd
from utils.utils import load_config
from utils.logger import Logger

logger = Logger()
config = load_config()
print(config)
host = config["PLATFORM CONFIG"]["host_port"]
ak = config["PLATFORM CONFIG"]["ak"]
as_str = config["PLATFORM CONFIG"]["as_str"]


def sign_http_request_with_ak_as(req, ak, as_str):
    """
    生成签名，用于HTTP请求验证
    """
    # 读取并重置请求体，这里假定req是requests.PreparedRequest实例
    if req.body:
        requestBody = bytes(req.body, 'utf-8')
    else:
        requestBody = b''
    # 确保 Accept
    if 'Accept' not in req.headers:
        req.headers['Accept'] = 'application/json'
    # 确保 Content-Type
    if 'Content-Type' not in req.headers:
        req.headers['Content-Type'] = 'application/x-www-form-urlencoded;charset=utf-8'
    # 确保 Date
    if 'Date' not in req.headers:
        req.headers['Date'] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    # 确保 Content-Length
    if 'Content-Length' not in req.headers:
        req.headers['Content-Length'] = str(len(requestBody))
    # print(f"requestBody: {requestBody}")
    # print(f"hashlib.md5(requestBody): {hashlib.md5(requestBody)}")
    # 计算body的MD5值
    bodyMd5 = hashlib.md5(requestBody).hexdigest()
    # print(f"ak: {ak}, as: {as_str}, md5: {bodyMd5}")
    # 添加自定义头部
    req.headers['X-Ca-Key'] = ak
    req.headers['X-Ca-Signature-Headers'] = 'X-Ca-Key'
    # 构造需要签名的字符串
    must_sign_string = "\n".join([
        req.method,
        req.headers.get('Accept', ''),
        bodyMd5,
        req.headers.get('Content-Type', ''),
        req.headers.get('Date', ''),
        ak,
        # req.headers.get('Content-Length', ''),
    ])
    # print(must_sign_string)
    # 使用HMAC SHA256进行签名
    hmac_sha256 = hmac.new(as_str.encode(), must_sign_string.encode(), hashlib.sha256)
    signature = base64.b64encode(hmac_sha256.digest()).decode()
    # 添加签名到头部
    req.headers['X-Ca-Signature'] = signature
    return


# 示例用法
def read_last_value(assetCode, attributeCode):
    """
    获取单个测点的最新值
    :param assetCode:
    :param attributeCode:
    :return:
    """
    url = config["PLATFORM CONFIG"]["get_last_data"].format(host, assetCode, attributeCode)
    # 创建请求
    req = requests.Request('GET', url)
    prepped = req.prepare()
    # 签名请求
    sign_http_request_with_ak_as(prepped, ak, as_str)
    # 实际发送请求
    with requests.Session() as session:
        response = session.send(prepped)
    res = response.text
    res = json.loads(res)['Results'][0]['Series'][0]
    df = pd.DataFrame(res['Values'], columns=res['Columns']).set_index('time')
    # 质量码在63以下表示数据正常
    df = df[df['quality'] <= 63]
    df = df[['value']]
    df.columns = [attributeCode]
    return df.values[0, 0]


# 示例用法
def read_last_value_batch(attributes):
    """
    获取批量测点的最新值
    :param attributes:
    :return:
    """
    url = config["PLATFORM CONFIG"]["get_batch_data"].format(host)
    payload = json.dumps({
        "Attributes": attributes,
        "Type": "last",
    })
    headers = {
        'Content-Type': 'application/json'
    }

    # 创建请求，并包含请求体和请求头headers
    req = requests.Request('POST', url, data=payload, headers=headers)
    prepped = req.prepare()
    # 签名请求
    sign_http_request_with_ak_as(prepped, ak, as_str)
    # 实际发送请求
    with requests.Session() as session:
        response = session.send(prepped)

    res = response.text
    res = json.loads(res)['Results']

    df = pd.DataFrame()
    for series in res:
        for sr in series['Series']:
            attributeCode = sr['AttributeCode']
            tmp = pd.DataFrame(sr['Values'],
                               columns=sr['Columns']).set_index('time')
            tmp = tmp[tmp['quality'] <= 63]
            tmp = tmp[['value']]
            tmp.columns = [attributeCode]
            df = pd.concat([df, tmp], axis=0)

    return df


# 示例用法
def read_history_value_batch(attributes, duration, flag='history'):
    """
    获取批量测点的历史数据
    :param attributes:
    :param duration: 秒
    :return:
    """
    if flag == 'live':
        host = config["PLATFORM CONFIG"]["live_port"]
        url = config["PLATFORM CONFIG"]["get_live_data"].format(host)
        until = datetime.now(tz=tzlocal())
        since = datetime.now(tz=tzlocal()) - timedelta(seconds=duration)
        payload = json.dumps({
            "Attributes": attributes,
            "Since": since.isoformat(),
            "Until": until.isoformat(),
        })
    else:
        host = config["PLATFORM CONFIG"]["host_port"]
        url = config["PLATFORM CONFIG"]["get_batch_data"].format(host)
        time_ago = datetime.now(tz=tzlocal()) - timedelta(seconds=duration)
        payload = json.dumps({
            "Attributes": attributes,
            "Since": time_ago.isoformat(),
        })

    
    headers = {
        'Content-Type': 'application/json'
    }
    # 创建请求，并包含请求体和请求头headers
    req = requests.Request('POST', url, data=payload, headers=headers)
    prepped = req.prepare()
    # 签名请求
    sign_http_request_with_ak_as(prepped, ak, as_str)
    # 实际发送请求
    with requests.Session() as session:
        response = session.send(prepped)
    res = response.text
    # print(res)
    res = json.loads(res)['Results']
    df = pd.DataFrame()
    for series in res:
        for sr in series['Series']:
            attributeCode = sr['AttributeCode']
            tmp = pd.DataFrame(sr['Values'],
                               columns=sr['Columns']).set_index('time')
            tmp = tmp[tmp['quality'] <= 63]
            tmp = tmp[['value']]
            tmp.columns = [attributeCode]
            df = pd.concat([df, tmp], axis=0)

    return df


# 示例用法
def write_value(assetCode, attributeCode, value):
    """
    写入单个测点数据
    :param assetCode:
    :param attributeCode:
    :param value:
    :return:
    """
    url = config["PLATFORM CONFIG"]["write_value"].format(host, assetCode, attributeCode)
    payload = json.dumps({"Value": str(value)})

    headers = {'Content-Type': 'application/json'}

    # 创建请求，并包含请求体和请求头headers
    req = requests.Request('PATCH', url, data=payload, headers=headers)
    prepped = req.prepare()

    # 签名请求
    sign_http_request_with_ak_as(prepped, ak, as_str)

    # 实际发送请求
    with requests.Session() as session:
        response = session.send(prepped)

    logger.raise_if_not(response.status_code == 200,
                        "Failed to upload {}-{}: {}".format(assetCode, attributeCode, response.text))
    return response


# 示例用法
def write_value_batch(args):
    """
    批量写入单个测点数据
    :param args:
    :return:
    """
    url = config["PLATFORM CONFIG"]["write_batch_value"].format(host)
    payload = json.dumps(args)
    headers = {'Content-Type': 'application/json'}

    # 创建请求，并包含请求体和请求头headers
    req = requests.Request('PATCH', url, data=payload, headers=headers)
    prepped = req.prepare()
    # 签名请求
    sign_http_request_with_ak_as(prepped, ak, as_str)
    # 实际发送请求
    with requests.Session() as session:
        response = session.send(prepped)

    logger.raise_if_not(response.status_code == 200, "Failed to upload {}: {}".format(args, response.text))
    return response


if __name__ == '__main__':
    # 读数据测试
    # df = read_history_value_batch([{
    #     "AssetCode": "device2",
    #     "AttributeCode": "double"
    # }, {
    #     "AssetCode": "device2",
    #     "AttributeCode": "int"
    # }], 30 * 60)
    attributes = [{
        "AssetCode": "ZS11C20",
        "AttributeCode": "6251005201ZS10C2000DB4010002"
    },
        {
            "AssetCode": "ZS11C20",
            "AttributeCode": "6251005201ZS10C2000TB1HU0001"
        },
        {
            "AssetCode": "ZS11C20",
            "AttributeCode": "6251005201ZS10C2000TB1010022"
        }
    ]

    duration = 10 * 60

    df = read_history_value_batch(attributes, duration, 'history') # history  live
    print(df)

    # 下控测试
    # response = write_value_batch([
    #     {
    #         "AssetCode": "device5",
    #         "Attributes":
    #             [
    #                 {
    #                     "Code": "bool",
    #                     "Value": "1",
    #                 }, {
    #                 "Code": "int",
    #                 "Value": "14",
    #             }
    #             ]
    #     }, {
    #         "AssetCode": "motor1",
    #         "Attributes":
    #             [
    #                 {
    #                     "Code": "speed",
    #                     "Value": "14.23",
    #                 }
    #             ]
    #     }
    # ])
    # print(response.status_code)
