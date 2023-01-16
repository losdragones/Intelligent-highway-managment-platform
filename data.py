#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2020/8/26 14:48
# @Author : way
# @Site : 
# @Describe:

import json
import time
import numpy as np

class SourceDataDemo(object):

    def __init__(self, traj, speed, volume, congestion):
        self.title = '智慧高速交通运行信息平台'
        self.counter1 = {'name': '实时车辆总数(辆)', 'value': '100'}
        self.counter2 = {'name': '车辆平均速度(km/h)', 'value': speed}
        self.counter3 = {'name': '当前交通流量(5min)', 'value': volume[0]}
        self.counter4 = {'name': '当前拥堵指数', 'value': 300}
        self.counter5 = {'name': '占用车道数', 'value': '2   '}
        self.counter6 = {'name': '预计最大排队长度', 'value': str(congestion[0])}
        self.counter7 = {'name': '事故持续时间', 'value': '30分钟'}
        self.counter8 = {'name': '预计拥堵时间', 'value': congestion[1]}
        self.echart1_data = {
            'title': '平均车速',
            'data': speed
        }
        self.echart2_data = {
            'title': '车辆运行轨迹展示',
            'series': traj
        }
        self.echart4_data = {
            'title': '交通流量变化/5分钟',
            'xAxis': list(volume[2]),
            'data': [
                {"name": "实际流量", "value": list(volume[0])},
                {"name": "预测流量", "value": list(volume[1])}
            ],
            'day': volume[3]
            # 'xAxis': list(volume['timestamp'].values),
            # 'xAxis': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
            #           '18', '19', '20', '21', '22', '23', '24'],
        }
        self.echart5_data = {
            'title': '拥堵里程变化',
            'data': [
                {"name": "5分钟", "value": congestion[2][0]},
                {"name": "10分钟", "value": congestion[2][1]},
                {"name": "15分钟", "value": congestion[2][2]},
                {"name": "20分钟", "value": congestion[2][3]},
                {"name": "25分钟", "value": congestion[2][4]},
                {"name": "30分钟", "value": congestion[2][5]}
            ]
        }
        self.echart6_data = {
            'title': '一线城市情况',
            'data': [
                {"name": "浙江", "value": 80, "value2": 20, "color": "01", "radius": ['59%', '70%']},
                {"name": "上海", "value": 70, "value2": 30, "color": "02", "radius": ['49%', '60%']},
                {"name": "广东", "value": 65, "value2": 35, "color": "03", "radius": ['39%', '50%']},
                {"name": "北京", "value": 60, "value2": 40, "color": "04", "radius": ['29%', '40%']},
                {"name": "深圳", "value": 50, "value2": 50, "color": "05", "radius": ['20%', '30%']},
            ]
        }

    @property
    def echart1(self):
        data = self.echart1_data
        v = data['data']
        tep = np.arange(len(v))
        echart = {
            'title': data.get('title'),
            'series': [[tep[i],v[i]] for i in range(len(v))]
        }
        return echart

    @property
    def echart2(self):
        data = self.echart2_data
        echart = {
            'title': data.get('title'),
            # 'xAxis': [i.get("name") for i in data.get('data')],
            'series': data.get('series')
        }
        return echart;

    @property
    def echart4(self):
        data = self.echart4_data
        echart = {
            'title': data.get('title'),
            'names': [i.get("name") for i in data.get('data')],
            'xAxis': data.get('xAxis'),
            'data': data.get('data'),
            'day': data.get('day')
        }
        return echart

    @property
    def echart5(self):
        data = self.echart5_data
        echart = {
            'title': data.get('title'),
            'xAxis': [i.get("name") for i in data.get('data')],
            'series': [i.get("value") for i in data.get('data')],
            'data': data.get('data'),
        }
        return echart

    @property
    def echart6(self):
        data = self.echart6_data
        echart = {
            'title': data.get('title'),
            'xAxis': [i.get("name") for i in data.get('data')],
            'data': data.get('data'),
        }
        return echart

    # @property
    # def map_1(self):
    #     data = self.map_1_data
    #     echart = {
    #         'symbolSize': data.get('symbolSize'),
    #         'data': data.get('data'),
    #     }
    #     return echart


class SourceData(SourceDataDemo,object):

    def __init__(self,traj,speed,volume,congestion):
        """
        按照 SourceDataDemo 的格式覆盖数据即可
        """
        super().__init__(traj,speed,volume,congestion)
        self.title = '智慧高速交通运行信息平台'

class CorpData(SourceDataDemo):

    def __init__(self):
        """
        按照 SourceDataDemo 的格式覆盖数据即可
        """
        super().__init__()
        with open('corp.json', 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
        self.title = data.get('title')
        self.counter1 = data.get('counter1')
        self.counter2 = data.get('counter2')
        self.echart1_data = data.get('echart1_data')
        self.echart2_data = data.get('echart2_data')
        self.echarts3_1_data = data.get('echarts3_1_data')
        self.echarts3_2_data = data.get('echarts3_2_data')
        self.echarts3_3_data = data.get('echarts3_3_data')
        self.echart4_data = data.get('echart4_data')
        self.echart5_data = data.get('echart5_data')
        self.echart6_data = data.get('echart6_data')
        # self.map_1_data = data.get('map_1_data')

class JobData(SourceDataDemo):

    def __init__(self):
        """
        按照 SourceDataDemo 的格式覆盖数据即可
        """
        super().__init__()
        with open('job.json', 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
        self.title = data.get('title')
        self.counter1 = data.get('counter1')
        self.counter2 = data.get('counter2')
        self.echart1_data = data.get('echart1_data')
        self.echart2_data = data.get('echart2_data')
        self.echarts3_1_data = data.get('echarts3_1_data')
        self.echarts3_2_data = data.get('echarts3_2_data')
        self.echarts3_3_data = data.get('echarts3_3_data')
        self.echart4_data = data.get('echart4_data')
        self.echart5_data = data.get('echart5_data')
        self.echart6_data = data.get('echart6_data')
        # self.map_1_data = data.get('map_1_data')