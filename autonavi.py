"""
高德地图 Web 服务 API 调用接口

目前实现了如下功能
  - autonavi_coordination 根据名称查询经纬度
  - autonavi_distance_from_name 根据名称计算距离
  - autonavi_distance_from_coord 根据经纬度计算距离
  - autonavi_generate_map 生成静态地图

官方文档链接：
https://lbs.amap.com/api/webservice/guide/api/staticmaps
"""

import requests
import json
from typing import Any

KEY = '6624cdfa073cbcb9c6c9dcf687da5468'

def format_dict(input:dict) -> str:

    output = json.dumps(
        input, 
        sort_keys=True, 
        indent=2, 
        separators=(',', ':'),
        ensure_ascii=False)
    
    return output


def get(url:str, params:dict) -> Any:

    for key, val in params.items():
        url += f'{key}={val}&'

    response = requests.get(url)

    try:
        response:dict = json.loads(response.content)
    except:
        response = response.content

    return response


def autonavi_coordination(name:str, flag=None) -> dict:
    
    # 查询起点经纬度
    url = 'https://restapi.amap.com/v3/geocode/geo?'
    param = {
        'key': KEY,
        'address': name,
    }
    response = get(url, param)
    name = response['geocodes'][0]['formatted_address']
    coordination = response['geocodes'][0]['location']

    return {
        'name': name,
        'coordination': coordination,
        'flag': flag
        }


def autonavi_distance_from_name(origin:str, destination:str, type:int = 1) -> dict:

    # 查询起点经纬度
    url = 'https://restapi.amap.com/v3/geocode/geo?'
    param = {
        'key': KEY,
        'address': origin,
    }
    response = get(url, param)
    begin_name = response['geocodes'][0]['formatted_address']
    begin_pos = response['geocodes'][0]['location']

    # 查询终点经纬度
    url = 'https://restapi.amap.com/v3/geocode/geo?'
    param = {
        'key': KEY,
        'address': destination,
    }
    response = get(url, param)
    end_name = response['geocodes'][0]['formatted_address']
    end_pos = response['geocodes'][0]['location']

    # 根据经纬度计算距离
    url = 'https://restapi.amap.com/v3/distance?'
    param = {
        'key': KEY,
        'origins': begin_pos,
        'destination': end_pos,
        'type': str(type),
        'output': 'JSON'
    }
    response = get(url, param)
    distance = response['results'][0]['distance']
    duration = response['results'][0]['duration']

    return {
        'origin': begin_name,
        'destination': end_name,
        'origin_pos': begin_pos,
        'destination_pos': end_pos,
        'distance': distance,
        'duration': duration
    }


def autonavi_distance_from_coord(origin:tuple, destination:tuple, type:int = 1, flag=None) -> dict:

    # 根据经纬度计算距离
    url = 'https://restapi.amap.com/v3/distance?'
    param = {
        'key': KEY,
        'origins': f'{origin[0]},{origin[1]}',
        'destination': f'{destination[0]},{destination[1]}',
        'type': str(type),
        'output': 'JSON'
    }
    response = get(url, param)
    distance = response['results'][0]['distance']
    duration = response['results'][0]['duration']

    return {
        'distance': int(distance),
        'duration': int(duration),
        'flag': flag
    }
        

def autonavi_generate_map(markers:str, paths:str) -> bytes:

    url = 'https://restapi.amap.com/v3/staticmap?'

    params = {
        'key': KEY,
        'zoom': '10',
        'size': '512*512',
        'scale': '2',
        'markers': markers,
        'paths': paths,
    }
 
    response = get(url, params)

    return response

if __name__ == "__main__":


    # 查询路径距离与预计行驶时间：
    result = autonavi_distance_from_name('湖北省博物馆', '武汉大学')
    origin = result['origin']
    destination = result['destination']
    origin_pos = result['origin_pos']
    destination_pos = result['destination_pos']
    distance = result['distance']
    duration = result['duration']

    print(f'起点：{origin} ({origin_pos})')
    print(f'终点：{destination} ({destination_pos})')
    print(f'路径距离：{float(distance)/1000}km')
    print(f'预计行驶时间：{float(duration)/60}min')


    # 生成静态地图
    markers = \
        f'large,0xFF0000,A:{origin_pos}|' + \
        f'large,0xFF0000,B:{destination_pos}'
    
    response = autonavi_generate_map(markers)

    with open('./test.png', 'wb') as f:
        f.write(response)

    
    
